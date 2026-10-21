#pragma once

// Q8_G32 RowSplit x BF16 integer-dot decode GEMV (T = 1), RDNA byte-dot variant.
//
// Same per-row code/scale layout the a16 SIMT GEMV (q8_a16_gemv.cuh) reads; the
// activation is different: a one-launch quantize pass turns x into an s8 vector plus
// one binary16 scale per 128-element group, and the per-element dequant + FMA partial
// becomes hardware byte dots -- one v_dot4_i32_iu8 per four codes. On gfx1102 this
// measured 1.49x over the a16 SIMT bar on the gdn-proj shape 12288x2048 (cal14, idle
// GPU; the cal14 mirrors verify the pipeline exactly against f64 references).
//
// Only backends with byte dots get this header (t2_gemv_dp4a.cuh gates the same way);
// the CUDA build keeps the fp32 SIMT path of q8_a16_gemv.cuh.

#if defined(__HIPCC__)

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Signed-operand byte dot: s8 weight codes against s8 activations, four per word.
// Plain sdot4 is rejected on gfx1102 (needs dot1-insts); sudot4 lowers to V_SUDOT4.
__device__ __forceinline__ std::int32_t q8_gemv_dot4(std::int32_t a, std::int32_t b,
                                                     std::int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// Byte-dot GEMV: one wavefront per row, StaticK/512 passes of one uint4 per lane.
// A 16B lane span lies inside a single 32-code weight group (512 % 32 == 0), so the
// pass's 16 group scales load in lane order (lane < 16) and broadcast with src
// lane >> 1; the 128-element x groups likewise broadcast with src lane >> 3. The
// row's product is (weight scale x x scale) -- both binary16 -- applied once per
// pass. Requires StaticK % 512 == 0.
template <int BlockRows, int StaticK, bool FullRows, class Output, class Epilogue>
__global__ __launch_bounds__(BlockRows * 32, 2)
    void q8_gemv_dp4a_kernel(const std::uint8_t* __restrict__ xq,
                              const std::uint16_t* __restrict__ xs,
                              Q8LinearOperands operands, Output output, Epilogue epilogue) {
    static_assert(StaticK > 0 && StaticK % 512 == 0);
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * BlockRows + warp;
    float acc = 0.0f;
    if (FullRows || row < operands.rows) {
        const auto* code_row = operands.codes + static_cast<std::int64_t>(row) * StaticK;
        const auto* scale_row =
            operands.scales + static_cast<std::int64_t>(row) * (StaticK / 32) * 2;
#pragma unroll
        for (int pass = 0; pass < StaticK / 512; ++pass) {
            const int off = pass * 512 + lane * 16;
            const uint4 codes = load_vec<uint4>(code_row + off);
            const uint4 xv = load_vec<uint4>(xq + static_cast<std::int64_t>(pass) * 512 +
                                                static_cast<std::int64_t>(lane) * 16);
            unsigned bits = 0;
            if (lane < 16) __builtin_memcpy(&bits, scale_row + (pass * 16 + lane) * 2, 2);
            const unsigned wbits = __shfl_sync(kFullWarpMask, bits, lane >> 1);
            unsigned xbits = 0;
            if (lane < 4) __builtin_memcpy(&xbits, xs + pass * 4 + lane, 2);
            const unsigned xgroup = __shfl_sync(kFullWarpMask, xbits, lane >> 3);
            std::int32_t dot = 0;
            dot = q8_gemv_dot4(static_cast<std::int32_t>(codes.x), static_cast<std::int32_t>(xv.x), dot);
            dot = q8_gemv_dot4(static_cast<std::int32_t>(codes.y), static_cast<std::int32_t>(xv.y), dot);
            dot = q8_gemv_dot4(static_cast<std::int32_t>(codes.z), static_cast<std::int32_t>(xv.z), dot);
            dot = q8_gemv_dot4(static_cast<std::int32_t>(codes.w), static_cast<std::int32_t>(xv.w), dot);
            const float scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(wbits))) *
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(xgroup)));
            acc = fmaf(static_cast<float>(dot), scale, acc);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            const float values[1]{acc};
            linear_finish_row(linear_output_tile<BlockRows>(output, static_cast<int>(blockIdx.x) * BlockRows),
                                epilogue, row, 0, values, 1);
        }
    }
}

} // namespace ninfer::ops::detail

#endif // defined(__HIPCC__)
