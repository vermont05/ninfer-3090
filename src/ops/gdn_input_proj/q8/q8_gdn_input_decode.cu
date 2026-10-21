#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/q8/q8_gemv_launch.cuh"

#ifdef NINFER_BACKEND_HIP
#include "core/arena.h"
#include "ops/linear/q8/q8_gemv_dp4a.cuh"

#include <cuda_fp16.h>

#include <cstdlib>
#include <string>
#endif

namespace ninfer::ops::detail {
namespace {

using Output = LinearBf16SegmentedOutput<8192, 4096>;

struct Q8GdnDecodeConvEpilogue {
    GdnConvEpilogue<SnapshotHistoryPublish> conv;
    __nv_bfloat16* z;

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output&, int row, int,
                                              const float (&projected)[1], int) const {
        if (row < 8192)
            conv.store(row, projected);
        else
            z[row - 8192] = __float2bfloat16_rn(projected[0]);
    }
};

GdnConvEpilogue<SnapshotHistoryPublish>
make_conv_epilogue(const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
                   const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query,
                   Tensor& key, Tensor& value) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        8192,
        2048,
        2048,
        4096,
        0,
        1,
        0,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data), 8192},
    };
}

#ifdef NINFER_BACKEND_HIP
// Byte-dot decode route (cal14: 1.49x over the a16 SIMT bar on 12288x2048, cal14-calibrated
// pipeline exact against f64 mirrors). NINFER_Q8_DP4A=off falls back to the SIMT schedules.
constexpr int kDp4aGroupK = 128;
constexpr int kDp4aStaticK = 2048;

struct Dp4aScratch {
    DeviceBuffer backing{8192u};
    std::uint8_t* const xq = static_cast<std::uint8_t*>(backing.p);
    std::uint16_t* const xs = reinterpret_cast<std::uint16_t*>(xq + kDp4aStaticK + 16u);
};

Dp4aScratch& dp4a_scratch() {
    static Dp4aScratch scratch;
    return scratch;
}

// One warp per 128-group of x: group amax -> binary16 scale -> s8 codes. Identical
// semantics to the T2 quantizer (t2_gemv_dp4a.cu), retargeted to this width.
__global__ void dp4a_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                       std::uint8_t* __restrict__ xq, std::uint16_t* __restrict__ xs,
                                       std::int32_t k) {
    const int group =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (group * kDp4aGroupK >= k) { return; }
    const __nv_bfloat16* base = x + static_cast<std::int64_t>(group) * kDp4aGroupK;
    const int l = static_cast<int>(threadIdx.x);
    float v[4];
    v[0] = __bfloat162float(base[l]);
    v[1] = __bfloat162float(base[l + 32]);
    v[2] = __bfloat162float(base[l + 64]);
    v[3] = __bfloat162float(base[l + 96]);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = std::fabs(v[i]);
        amax = amax > a ? amax : a;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_xor_sync(0xFFFFFFFFu, amax, off);
        amax = amax > o ? amax : o;
    }
    const float scale = amax * (1.0f / 127.0f);
    if (amax > 0.0f) {
        xs[group] = static_cast<std::uint16_t>(__half_as_ushort(__float2half_rn(scale)));
    }
    const float s = amax > 0.0f ? __half2float(__ushort_as_half(xs[group])) : 0.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    std::uint8_t* ob = xq + static_cast<std::int64_t>(group) * kDp4aGroupK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = static_cast<int>(__float2int_rn(v[i] * inv));
        ob[l + i * 32] = static_cast<std::uint8_t>(q);
    }
}

bool dp4a_decode_enabled() {
    static const bool value = [] {
        const char* value = std::getenv("NINFER_Q8_DP4A");
        return value == nullptr || std::string(value) != "off";
    }();
    return value;
}

template <class Output, class Epilogue>
void launch_dp4a_gemv(const Tensor& x, const Weight& weight, Output output,
                       const Epilogue& epilogue, cudaStream_t stream) {
    const Q8LinearOperands operands = q8_linear_operands(x, weight);
    validate_q8_operands(operands);
    if (operands.tokens != 1 || operands.k != kDp4aStaticK || operands.padded_k != kDp4aStaticK) {
        throw std::invalid_argument("Q8 GDN dp4a decode: unsupported operands");
    }
    Dp4aScratch& scratch = dp4a_scratch();

    constexpr int kQuantizeWarps = 4;
    constexpr int kGroups = kDp4aStaticK / kDp4aGroupK;
    const dim3 qblock(32u, kQuantizeWarps);
    const dim3 qgrid((kGroups + kQuantizeWarps - 1) / kQuantizeWarps);
    dp4a_quantize_kernel<<<qgrid, qblock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), scratch.xq, scratch.xs, kDp4aStaticK);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(static_cast<unsigned>((operands.rows + 7) / 8));
    if (operands.rows % 8 == 0) {
        q8_gemv_dp4a_kernel<8, kDp4aStaticK, true, Output, Epilogue>
            <<<grid, 256, 0, stream>>>(scratch.xq, scratch.xs, operands, output, epilogue);
    } else {
        q8_gemv_dp4a_kernel<8, kDp4aStaticK, false, Output, Epilogue>
            <<<grid, 256, 0, stream>>>(scratch.xq, scratch.xs, operands, output, epilogue);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif // NINFER_BACKEND_HIP

} // namespace

void q8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {

    constexpr int kRowsPerCta = 8;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
#ifdef NINFER_BACKEND_HIP
    if (dp4a_decode_enabled()) {
        launch_dp4a_gemv(x, weight, output, LinearIdentityEpilogue{}, stream);
        return;
    }
#endif
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {

    constexpr int kRowsPerCta = 8;
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const Q8GdnDecodeConvEpilogue epilogue{
        make_conv_epilogue(conv_weight, conv_states, valid_columns, initial_slot,
                           snapshot_base_slot, query, key, value),
        static_cast<__nv_bfloat16*>(z.data),
    };
#ifdef NINFER_BACKEND_HIP
    if (dp4a_decode_enabled()) {
        launch_dp4a_gemv(x, weight, ignored_output, epilogue, stream);
        return;
    }
#endif
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
