// Byte-dot decode GEMV for T2 row-split weights (gfx1102): one quantize wavefront per
// 128-group of x, then the row-quad dp4a kernel (t2_gemv_dp4a.cuh). The quantized activation
// plane lives in a lazily allocated process buffer sized for the widest registered shape.
// HIP-only: the byte dots that make it win do not exist on NVIDIA (mma.cuh gates the same way).

#ifdef NINFER_BACKEND_HIP

#include "ops/linear/t2/t2_gemv_dp4a.cuh"
#include "ops/linear/t2/t2_gemv_dp4a.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Groups per quantize CTA (quantize kernel blockDim.y).
constexpr int kDp4aQuantizeWarps = 4;

// Quantized-activation scratch: the widest registered shape (k = 17408: 17408 code bytes +
// 136 binary16 scales) plus alignment slack, one cudaMalloc on first use. Same-stream calls
// serialize on it (a later call's quantize cannot pass this call's GEMV).
struct Dp4aScratch {
    DeviceBuffer backing{32u * 1024u};
    std::uint8_t* const xq  = static_cast<std::uint8_t*>(backing.p);
    std::uint16_t* const xs = reinterpret_cast<std::uint16_t*>(xq + 17408u + 16u);
};

Dp4aScratch& dp4a_scratch() {
    static Dp4aScratch scratch;
    return scratch;
}

} // namespace

__global__ void t2_gemv_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                          std::uint8_t* __restrict__ xq,
                                          std::uint16_t* __restrict__ xs, std::int32_t k) {
    const int group = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) +
                        static_cast<int>(threadIdx.y);
    if (group * T2RowSplitStorage::kGroupK >= k) { return; }
    const __nv_bfloat16* base = x + static_cast<std::int64_t>(group) * T2RowSplitStorage::kGroupK;
    const int l               = static_cast<int>(threadIdx.x);
    float v[4];
    v[0] = __bfloat162float(base[l]);
    v[1] = __bfloat162float(base[l + 32]);
    v[2] = __bfloat162float(base[l + 64]);
    v[3] = __bfloat162float(base[l + 96]);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = std::fabs(v[i]);
        amax            = amax > a ? amax : a;
    }
    // The shfl fold of the group amax; keep the temp form (the inline ternary of two
    // shfl calls lowers to a hanging kernel on this generation).
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_xor_sync(kFullWarpMask, amax, off);
        amax            = amax > o ? amax : o;
    }
    const float scale = amax * (1.0f / 127.0f);
    if (amax > 0.0f) {
        xs[group] = static_cast<std::uint16_t>(__half_as_ushort(__float2half_rn(scale)));
    }
    const float s   = amax > 0.0f ? __half2float(__ushort_as_half(xs[group])) : 0.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    std::uint8_t* ob = xq + static_cast<std::int64_t>(group) * T2RowSplitStorage::kGroupK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = static_cast<int>(__float2int_rn(v[i] * inv));
        ob[l + i * 32] = static_cast<std::uint8_t>(q);
    }
}

namespace {

struct Dp4aShape {
    std::int32_t n;
    std::int32_t k;
    std::int32_t tile; // groups per warp tile of the GEMV schedule
};

// The registered decode shapes (t2_dispatch's switch), with the cal12 tile winner.
constexpr Dp4aShape kDp4aShapes[] = {
    {34816, 5120, 16},   {6144, 5120, 32},   {1024, 5120, 32},  {4096, 5120, 32},
    {7168, 5120, 32},    {12288, 5120, 32}, {14336, 5120, 32}, {16384, 5120, 32},
    {131072, 5120, 16}, {248320, 5120, 16}, {5120, 6144, 32},  {5120, 17408, 32},
};

const Dp4aShape* find_dp4a_shape(std::int32_t n, std::int32_t k) {
    for (const auto& entry : kDp4aShapes) {
        if (entry.n == n && entry.k == k) { return &entry; }
    }
    return nullptr;
}

template <class Schedule>
void launch_dp4a_gemv(const Dp4aShape& shape, const Weight& w, Tensor& out,
                       const std::uint8_t* xq, const std::uint16_t* xs, std::int32_t k,
                       cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const dim3 grid(static_cast<unsigned>((rows + Schedule::kRowsPerCta - 1) /
                                            Schedule::kRowsPerCta));
    const dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    t2_gemv_dp4a_kernel<Schedule><<<grid, block, 0, stream>>>(
        xq, xs, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows,
        k);
    (void)shape;
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace {
void invalid(const char* what) { throw std::invalid_argument(what); }
} // namespace

bool t2_gemv_dp4a_supported(std::int32_t n, std::int32_t k) {
    return find_dp4a_shape(n, k) != nullptr;
}

void t2_gemv_dp4a_run(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (t != 1 || k % T2RowSplitStorage::kGroupK != 0) {
        invalid("t2 dp4a: expects a single contiguous BF16 column");
    }
    const Dp4aShape* shape = find_dp4a_shape(w.n, w.k);
    if (shape == nullptr) { invalid("t2 dp4a: unregistered shape"); }
    if (k > 17408) { invalid("t2 dp4a: k beyond the scratch"); }

    Dp4aScratch& scratch = dp4a_scratch();
    auto* xq             = scratch.xq;
    auto* xs             = scratch.xs;

    const std::int32_t groups = k / T2RowSplitStorage::kGroupK;
    const dim3 qblock(32u, static_cast<unsigned>(kDp4aQuantizeWarps));
    const dim3 qgrid(
        static_cast<unsigned>((groups + kDp4aQuantizeWarps - 1) / kDp4aQuantizeWarps));
    t2_gemv_quantize_kernel<<<qgrid, qblock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), xq, xs, k);
    CUDA_CHECK(cudaGetLastError());

    if (shape->tile == 16) {
        launch_dp4a_gemv<T2GemvDp4aWord16Schedule>(*shape, w, out, xq, xs, k, stream);
    } else {
        launch_dp4a_gemv<T2GemvDp4aWord32Schedule>(*shape, w, out, xq, xs, k, stream);
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_BACKEND_HIP
