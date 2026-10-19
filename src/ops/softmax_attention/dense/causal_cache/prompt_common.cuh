#pragma once

// Shared causal-attention dimensions and leaf PTX helpers used by the independently tuned
// quantized-cache prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptHeadDim = 256;

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalPromptHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void causal_prompt_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                               int row_begin, int row_end, int tid,
                                                               int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kCausalPromptHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kCausalPromptHeadDim;
        const int d   = element - (row - row_begin) * kCausalPromptHeadDim;
        out[causal_prompt_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int causal_prompt_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_prompt_store_byte_swizzled(Byte* tile, int row, int d,
                                                                  Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalPromptHeadDim / 2) + causal_prompt_swz(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

#if defined(__HIPCC__)
// CUDA's named-barrier rendezvous (`bar.sync id, 128`) has no AMD spelling: a GCN workgroup
// joins at one barrier and there is no subset form, so the prompt kernels' 4-wave producer
// rendezvous is this. Every wave of the group holds a lane-0 store ahead of a wave barrier
// and a block fence, then every thread spins the four stamps until each carries the call's
// generation. Callers pass a strictly increasing generation and storage that was zeroed
// before a block barrier, so the first wait cannot read a stamp before its owner zeroes it.
__device__ __forceinline__ void causal_prompt_subset_barrier(unsigned* slots, int wid,
                                                              unsigned gen) {
    if ((threadIdx.x & 31) == 0)
        slots[wid] = gen;
    __syncwarp();
    volatile unsigned* const stamps = reinterpret_cast<volatile unsigned*>(slots);
    for (;;) {
        if (stamps[0] >= gen && stamps[1] >= gen && stamps[2] >= gen && stamps[3] >= gen) {
            break;
        }
    }
    __syncwarp();
}
#endif

template <int Columns>
__device__ __forceinline__ int causal_prompt_p_swz(int row, int col) {
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return causal_prompt_swz(row, col);
}

} // namespace ninfer::ops
