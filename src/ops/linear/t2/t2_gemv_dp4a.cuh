#pragma once

// T2G128_F16S RowSplit x BF16 integer-dot GEMV (T = 1), RDNA byte-dot variant.
//
// Same RowSplit weights and PackedWord16 lane mapping as t2_rowsplit_gemv.cuh; the
// activation is different: a one-launch quantize pass turns x into an s8 vector plus one
// binary16 scale per 128-element group, and the fp32 sign-mask partial becomes hardware
// byte dots -- one v_dot4_i32_iu8 per four codes. On gfx1102 this measured 1.40x (gate_up
// 34816x5120), 1.91x (down 5120x17408) and 1.79x (attn 6144x5120) against the bit-trick word
// partial at the same bar (cal12, idle GPU; cal13q proved the pipeline matches the float64
// reference exactly at both k).
//
// Only backends with byte dots get this header (mma.cuh gates its dot_s8w8 the same
// way); the CUDA build keeps the fp32 word partials of t2_rowsplit_gemv.cuh.

#if defined(__HIPCC__)

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/t2/t2_rowsplit_gemv.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One code byte -> four sign-extended bytes for the byte dot: byte j of the result is
// 0x00 for codes 0 and 2, 0x01 for code 1 and 0xFF for code 3 (the -1 byte the dot negates).
// The selector carries 4 bits per result byte and indexes the two table operands; both
// operands are the same 4-entry table {0x00, 0x01, 0x00, 0xFF} split across two sources,
// so index 0/4 -> 0x00, 1/5 -> 0x01, 2/6 -> 0x00, 3/7 -> 0xFF.
__device__ __forceinline__ std::uint32_t t2_gemv_sym4(std::uint32_t code_byte) {
    const std::uint32_t y = (code_byte & 0x33u) | ((code_byte & 0xCCu) << 6);
    const std::uint32_t z = (y & 0x0303u) | ((y & 0x3030u) << 12);
    return static_cast<std::uint32_t>(__builtin_amdgcn_perm(0xFF000100u, 0xFF000100u, z));
}

// Byte-dominant word schedules. The fp32 word schedule tiles 8 groups; the byte path
// measures best one tile wider (cal12: 16 groups on n=34816, 32 on the low-row shapes).
using T2GemvDp4aWord16Schedule =
    T2RowSplitGemvSchedule<4, 1, 16, T2GemvLaneMapping::PackedWord16, 0, 2>;
using T2GemvDp4aWord32Schedule =
    T2RowSplitGemvSchedule<4, 1, 32, T2GemvLaneMapping::PackedWord16, 0, 2>;

// x [k] bf16 -> xq [k] s8 + xs [k/128] binary16: the largest-absolute value of each
// 128-group sets its scale (amax/127, rounded to binary16), and the group's 128 elements
// divide by it. One warp per group (blockDim.y picks the group inside the CTA), so the
// amax reduce is a single shfl fold and the whole vector goes in one wavefront.
// Requires k % 128 == 0 (every registered T2 row-split width qualifies).
__global__ void t2_gemv_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                          std::uint8_t* __restrict__ xq,
                                          std::uint16_t* __restrict__ xs, std::int32_t k);

// Word-mapped byte-dot partial: one wavefront per row, staging each group's codes in
// LDS (shared with the fp32 path's schedule type), then one byte dot per code word.
// Scale per group is the weight scale x activation scale product, both binary16.
template <class Schedule>
__device__ __forceinline__ float
t2_gemv_dot_word_dp4a(T2GemvTileStorage<Schedule>& tiles, int cta_warp,
                      const std::uint8_t* __restrict__ activation_bytes,
                      const std::uint16_t* __restrict__ activation_scales,
                      const std::uint8_t* __restrict__ code_row,
                      const std::uint8_t* __restrict__ scale_row, int group_begin, int group_end,
                      int lane) {
    static_assert(Schedule::kLaneMapping == T2GemvLaneMapping::PackedWord16);
    constexpr int kGroupsPerTile = Schedule::kGroupsPerWarpTile;

    uint4* shared_codes   = tiles.codes[cta_warp];
    const int lane_group  = lane >> 3;
    const int lane_in_group = lane & 7;
    float accumulator     = 0.0f;

    for (int tile_begin = group_begin; tile_begin < group_end; tile_begin += kGroupsPerTile) {
        const int active_groups = min(kGroupsPerTile, group_end - tile_begin);
        const int active_vectors  =
            active_groups * T2RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
        const auto* global_codes =
            reinterpret_cast<const uint4*>(code_row + static_cast<std::int64_t>(tile_begin) *
                                                          T2RowSplitStorage::kCodeBytesPerGroup);
        for (int vector = lane; vector < active_vectors; vector += 32) {
            shared_codes[vector] = load_ldg<uint4>(&global_codes[vector]);
        }
        std::uint16_t lane_scale_bits = 0;
        if (lane < active_groups) {
            lane_scale_bits =
                load_vec<std::uint16_t>(scale_row + static_cast<std::int64_t>(tile_begin + lane) *
                                                        T2RowSplitStorage::kScaleBytesPerGroup);
        }
        __syncwarp();

        const auto* words = reinterpret_cast<const std::uint32_t*>(shared_codes);
#pragma unroll
        for (int base = 0; base < kGroupsPerTile; base += 4) {
            const int local_group = base + lane_group;
            // The shuffle is warp-collective; only the use below is predicated.
            const std::uint16_t scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(kFullWarpMask, lane_scale_bits, min(local_group, 31)));
            if (local_group < active_groups) {
                const std::uint32_t packed = words[base * 8 + lane];
                const std::uint16_t act_bits = load_vec<std::uint16_t>(
                    activation_scales + static_cast<std::int64_t>(tile_begin + local_group));
                const uint4 xv      = load_vec<uint4>(
                    activation_bytes +
                    static_cast<std::int64_t>(tile_begin + local_group) * T2RowSplitStorage::kGroupK +
                    lane_in_group * T2RowSplitStorage::kCodesPerWord);
                const std::uint32_t q0 = t2_gemv_sym4(packed & 0xFFu);
                const std::uint32_t q1 = t2_gemv_sym4((packed >> 8) & 0xFFu);
                const std::uint32_t q2 = t2_gemv_sym4((packed >> 16) & 0xFFu);
                const std::uint32_t q3 = t2_gemv_sym4((packed >> 24) & 0xFFu);
                int acc                = 0;
                acc                      = __dp4a(static_cast<int>(q0), static_cast<int>(xv.x), acc);
                acc                      = __dp4a(static_cast<int>(q1), static_cast<int>(xv.y), acc);
                acc                      = __dp4a(static_cast<int>(q2), static_cast<int>(xv.z), acc);
                acc                      = __dp4a(static_cast<int>(q3), static_cast<int>(xv.w), acc);
                const float scale      = __half2float(__ushort_as_half(scale_bits)) *
                                       __half2float(__ushort_as_half(act_bits));
                accumulator = fmaf(static_cast<float>(acc), scale, accumulator);
            }
        }
        __syncwarp();
    }
    return accumulator;
}

// Row-quad GEMV over the quantized vector: row = blockIdx.x * 4 + warp.
template <class Schedule>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::kLaunchBoundsMinBlocks) void t2_gemv_dp4a_kernel(
    const std::uint8_t* __restrict__ xq, const std::uint16_t* __restrict__ xs,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k) {
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;

    __shared__ T2GemvTileStorage<Schedule> tiles;

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp = static_cast<int>(threadIdx.x) >> 5;
    const int row      = static_cast<int>(blockIdx.x) * kRowsPerCta + cta_warp;

    const int groups_per_row = k / T2RowSplitStorage::kGroupK;
    float accumulator          = 0.0f;
    if (row < rows) {
        const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * groups_per_row *
                                                    T2RowSplitStorage::kCodeBytesPerGroup;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row *
                                                     T2RowSplitStorage::kScaleBytesPerGroup;
        accumulator = t2_gemv_dot_word_dp4a<Schedule>(tiles, cta_warp, xq, xs, code_row, scale_row,
                                                        0, groups_per_row, lane);
    }
    accumulator = warp_reduce_sum(accumulator);
    if (lane == 0 && row < rows) { out[row] = __float2bfloat16(accumulator); }
}

} // namespace ninfer::ops::detail

#endif // defined(__HIPCC__)
