#pragma once

// T2G128 RowSplit prefill GEMM, 4-bit-activation twin of t2_prefill_wmma1_kernel: the served
// (1,2) tile, its panel raster, its magic-bits fold and its epilogue, fed by
// v_wmma_i32_16x16x16_iu4_w32 instead of the iu8 form. Built behind NINFER_T2_A8_IU4 (default
// off) so the shipped i8 route is untouched until a served ABBA set decides.
//
// WHY ASK FOR IT (measured, cal42 peak-issue battery, .qual/wmma/cal42.*):
//   iu8  14.18 / 14.49 / 14.91 TMAC/s at NACC = 2 / 8 / 16
//   iu4  25.32 / 27.62 / 28.07 TMAC/s at NACC = 2 / 8 / 16   (~1.9x the iu8 issue rate)
// and cal42's fragment geometry: iu4 A and B are 8 B/lane (int2) against iu8's 16 B/lane (int4),
// C identical (int8v = 8 x i32). One instruction consumes a 16-code k-window per lane either
// way, so a 128-k group costs EIGHT s-steps in both routes: this kernel's instruction count per
// group equals the served kernel's. The win is issue rate plus half the activation operand bytes
// (and the register relief that comes with them: cal42's iu4 probes held 52 / 64 / 80 VGPRs at
// NACC 2 / 8 / 16 against iu8's 152 at NACC 8).
//
// QUALITY COST (measured CPU-side, .qual/wmma/proof-iu4-act.cpp -> proof-iu4-act.log, TOTAL
// FAILURES = 0): activations in 4 bits per 128-k group are 13-19x worse in rms relative error
// than the served i8 route in every healthy regime (gaussian sigma=1: i8 0.00401 vs iu4 0.07096,
// cos 0.999992 vs 0.997486; heavy tail x30: 0.02091 vs 0.28314; sigma=30: 0.00408 vs 0.07550).
// Weights are lossless in nibbles (proof check B, and proof-iu4-m5.log for the builder), so the
// cost is entirely on the activation side. iu4 is therefore a PURE THROUGHPUT BET and ships only
// on a served measurement, per the project's rule that the served request is the only verdict.
//
// ACTIVATION CONTRACT (the producer below): codes [Tpad][K/2] packed nibbles, scales
// [K/128][Tpad] binary16 with scale = amax / 7 (the i8 producer's shape with 127 replaced by 7),
// Tpad a multiple of kColumns, padded columns zero, a zero group keeps a zero scale and zero
// codes. The plane is THE i8 PLANE, byte-paired: nibble byte n = i8-plane bytes 2n (low nibble)
// and 2n+1 (high nibble), so nibble position p carries the same code the served plane carries at
// byte p -- namely perm16^-1(p) (proof check A: a nibble byte always holds a same-parity k pair
// {16Q+r*2, 16Q+r*2+2}, a byte's two nibbles never straddle parities). Consequences:
//   * the fragment window for s-step s is the 8 bytes at (gBase + s*16) / 2 = gBase/2 + 8 s,
//     whole-byte and 8-byte aligned (proof check A's alignment clause);
//   * the producer's shuffle network is the served one VERBATIM: the k-set lane l must gather is
//     identical in both routes (l even -> {4l, 4l+1, 4l+4, 4l+5}, l odd -> {4l-2, 4l-1, 4l+2,
//     4l+3}), so only the store changes -- 32 lanes x 2 bytes = 64 B = K/2 per group;
//   * the epilogue, the scale plane and every caller-side geometry are shared unchanged.
//
// PAIRING LAW: the served i8 kernel's verified semantics are "plane byte p is multiplied with
// weight fragment byte p", where wmma_code_frag lays weights out in k order (byte p = code p) and
// the plane is permuted so byte p holds activation code perm16^-1(p). This route keeps that law
// nibble-wise: wmma_code_frag4 (below) lays nibble p = code p, and the plane's nibble p holds
// perm16^-1(p) by construction above. What is NOT inherited is the pairing INSIDE a byte: i8 pairs
// whole bytes, iu4 pairs nibbles at the same nibble index. That is hypothesis H5, and it is the
// one open ship gate: cal77 (.qual/wmma, hardware) must confirm the slot mapping under the
// instruction against the served i8 tile on identical quantised data before this route's served
// ABBA leg is worth a card minute. proof-iu4-act.log carries the same note.
//
// WEIGHT SIDE IS UNCHANGED: T2 payloads stay 2-bit packed (32 code bytes per row per group =
// eight 16-code words, the same bytes the shipped tile cp_async's in the same order). Each word
// decodes to ONE 8-byte nibble fragment (four perms, the same perm count as wmma_code_frag's four)
// and drives one s-step, so 8 words = 8 s-steps per group, matching the activation plane's eight
// 8-byte windows.
//
// BUILDER (closed ship gate, proof-iu4-m5.log TOTAL mismatches = 0 over 2^32 words, corroborated
// on hardware by cal76): out[j] = lo | (hi << 4) with
//   lo = PERM(Tlo, Thi, spread3(w >> 16 j)) & 0x0F0F0F0F,  hi = same at (w >> 16 j + 2),
//   Tlo = 0x0F0E0100 (kF[0..3]), Thi = 0x1F1E1110 (kF[4..7]), kF = {00, 01, 0E, 0F}, j = 0..1.
// Selector bytes 4..7 (Thi's bytes) are exercised here and nowhere else in the served kernels;
// cal76 measured them, and M4 (src0-only 4-byte table) -- the only reading that breaks this
// builder -- is excluded on hardware. M5 (the pair with src1 as the LOW half) is the measured
// semantics and is what the proof covers.
//
// __launch_bounds__ deliberately has NO minBlocks. The served kernel's (32, 8) pins the allocator
// at 64 VGPRs; this kernel's live set (accf, the hoisted xs row, the MF row pointers, the 32-byte
// weight line) is comparable to iu8's and the register relief here comes from the 8-byte fragments,
// not from a smaller accumulator. cal21/cal22 measured what a spilled tile costs (~2.4x), so
// occupancy is left to the allocator and read back from the build's VGPR report; the panel grid
// saturates the card without a minBlocks hint.

#if defined(__HIPCC__)

#include "ops/common/memory.cuh"
#include "ops/linear/t2/t2_prefill_i8.cuh"
#include "ops/linear/t2/t2_prefill_wmma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Four code bytes (byte j in bits 8 j, only its low nibble meaningful) packed to two nibble bytes:
// result byte n = (byte 2n) | (byte 2n+1 << 4), the pairing the nibble plane's layout law asks for.
// Four masked shifts, no shuffle. The 12-bit mask form is deliberate: shifting the 4-bit fields
// DOWN to bits 0..15 (not widening in place) is what keeps the two output bytes adjacent, and it is
// checked by the round-trip in .qual/wmma/proof-iu4-act.cpp (check C).
__device__ __forceinline__ std::uint16_t nibble_pack4(unsigned w) {
    const unsigned lo = w & 0x0000000Fu;             // byte 0 -> nibble 0
    const unsigned m1 = (w >> 4) & 0x000000F0u;      // byte 1 -> nibble 1
    const unsigned m2 = (w >> 8) & 0x00000F00u;      // byte 2 -> nibble 2
    const unsigned hi = (w >> 12) & 0x0000F000u;     // byte 3 -> nibble 3
    return static_cast<std::uint16_t>(lo | m1 | m2 | hi);
}

namespace wmma_iu4_detail {

using int2v = int __attribute__((__vector_size__(8)));

// Selector splay for the nibble builder: byte q of the result carries code 2q of w (three low bits
// -- code 2q's value plus the sign-extension bit the 16-entry table needs), so every selector byte
// lands in 0..15 and indexes Tlo/Thi's sixteen entries directly. Its first four terms are the
// served spread4's shape at 4-bit granularity; the builder's 0x0F0F0F0F mask clears bytes 4..7.
__device__ __forceinline__ unsigned spread3(unsigned w) {
    return (w & 7u) | (((w >> 4) & 7u) << 8) | (((w >> 8) & 7u) << 16) | (((w >> 12) & 7u) << 24);
}

// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) widened to the 8-byte
// nibble fragment, nibble j = signed-4-bit view of code j. Four perms per word, matching
// wmma_code_frag's four (proof-iu4-m5.log: builder vs the independent 16-code reference, 0
// mismatches over 2^32 words).
__device__ __forceinline__ int2v wmma_code_frag4(unsigned word) {
    constexpr unsigned kTlo = 0x0F0E0100u; // kF[0..3] = {00, 01, 0E, 0F}
    constexpr unsigned kThi = 0x1F1E1110u; // kF[4..7] = {10, 11, 1E, 1F}
    const unsigned lo0 = static_cast<unsigned>(__builtin_amdgcn_perm(kTlo, kThi, spread3(word))) & 0x0F0F0F0Fu;
    const unsigned hi0 = static_cast<unsigned>(__builtin_amdgcn_perm(kTlo, kThi, spread3(word >> 2))) & 0x0F0F0F0Fu;
    const unsigned lo1 = static_cast<unsigned>(__builtin_amdgcn_perm(kTlo, kThi, spread3(word >> 16))) & 0x0F0F0F0Fu;
    const unsigned hi1 = static_cast<unsigned>(__builtin_amdgcn_perm(kTlo, kThi, spread3(word >> 18))) & 0x0F0F0F0Fu;
    return {static_cast<int>(lo0 | (hi0 << 4)), static_cast<int>(lo1 | (hi1 << 4))};
}

} // namespace wmma_iu4_detail

// Producer for the nibble plane, mirroring t2_prefill_i8_quantize_kernel's shape verbatim: one
// block per padded column, a warp per 128-k group, a lane per four k, padding columns zero. The
// amax butterfly, the shuffle gather and its source/shift derivation are the served producer's;
// only the scale divisor (7, not 127), the clamp (+-7), the quantiser's divisor source and the
// 2-byte store differ.
//
// The scale floor (2^-24 = binary16's smallest subnormal) fixes a defect of the SHIPPED producer
// that the iu4 route exposes: the stored scale is __float2half(amax / d), so with
// amax < d * 127 * 2^-24 the division rounds to ZERO, the group's codes collapse to zero and the
// whole 128-k group contributes nothing. Measured (proof-iu4-act.log, arm "i8 + binary16
// subnormal floor"): a group with amax ~ 1e-3 costs the served i8 route rms-rel 1.00000 (total
// collapse) and the floored route 0.02977 (x0.03, 128 groups floored); the near-zero band
// (amax 3e-7 ... 1e-5) goes 0.38558 -> 0.04903 (x0.13, 48 groups floored); every healthy regime
// is a wash (x0.96 ... x1.00, no group floored). It is a compare and a select per group, no layout
// change, and dots stay inside the served accumulator window because the codes stay in +-d.
template <std::int32_t kCols>
__global__ __launch_bounds__(T2PrefillI8::kQuantThreads) void t2_prefill_iu4_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t tokens, std::int32_t t_pad,
    std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    using C              = T2PrefillI8;
    constexpr int kGroups = kCols / C::kGroupK;
    constexpr int kWarps  = C::kQuantThreads / 32;
    constexpr float kMinSub = 5.9604644775390625e-8F; // binary16 min subnormal = 2^-24
    const int t            = static_cast<int>(blockIdx.x);
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    std::int8_t* const dst = codes + static_cast<std::size_t>(t) * (kCols / 2);
    if (t >= tokens) {
        for (int i = static_cast<int>(threadIdx.x); i < kCols / 32; i += C::kQuantThreads) {
            store_vec(dst + i * 16, make_uint4(0u, 0u, 0u, 0u)); // 16 B x kCols/32 = kCols/2 bytes
        }
        for (int g = static_cast<int>(threadIdx.x); g < kGroups; g += C::kQuantThreads) {
            scales[static_cast<std::size_t>(g) * t_pad + t] = __float2half(0.0F);
        }
        return;
    }
    const __nv_bfloat16* const src = x + static_cast<std::size_t>(t) * kCols;
    // Identical to the served producer's derivation: lane l's four nibble bytes are the i8-plane
    // bytes [4 l, 4 l + 4), gathered from the even/odd code pairs of lanes 4q + (l & 2).
    const int source = lane & ~1;
    const int shift  = (lane & 1) * 16;
    for (int g = warp; g < kGroups; g += kWarps) {
        const uint2 raw = load_vec<uint2>(src + g * C::kGroupK + lane * 4);
        const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float f[4];
        float amax = 0.0F;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[j] = __bfloat162float(v[j]);
            amax = fmaxf(amax, fabsf(f[j]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        __half scale = __float2half(amax / 7.0F);
        if (amax > 0.0F && __half2float(scale) == 0.0F) { scale = __float2half(kMinSub); }
        if (lane == 0) { scales[static_cast<std::size_t>(g) * t_pad + t] = scale; }
        const float s   = __half2float(scale);
        const float inv = s > 0.0F ? 1.0F / s : 0.0F; // the producer's own divisor, as shipped
        unsigned q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = static_cast<unsigned>(max(-7, min(7, __float2int_rn(f[j] * inv)))) & 0xffu;
        }
        const unsigned packed = q[0] | (q[2] << 8) | (q[1] << 16) | (q[3] << 24);
        const unsigned a      = __shfl_sync(0xffffffffu, packed, source);
        const unsigned b      = __shfl_sync(0xffffffffu, packed, source + 1);
        const unsigned word   = ((a >> shift) & 0xffffu) | (((b >> shift) & 0xffffu) << 16);
        store_vec(dst + g * (C::kGroupK / 2) + lane * 2, nibble_pack4(word));
    }
}

// The (16*MF) x (16*NF) one-warp tile of t2_prefill_wmma1_kernel, 4-bit twin: same grid and
// raster conventions (panel mode, token-fast panel mode, rows_fast, callers size grids
// identically), same per-group
// structure, same hoisted xs row, same magic-bits fold, same epilogue call sequence -- the
// fragments are int2 (8 B/lane) instead of int4 (16 B/lane), so the group runs eight s-steps on
// half-width operands and the activation plane is read at k_cols / 2 bytes per token.
//
// The fold's integer headroom is strictly wider here: proof-iu4-act.log measures max |dot| per
// group at 1792 for iu4 against 32512 for i8 (18.1x smaller), against a fold that resolves
// integers up to 8388608 with the accumulator window [4194304, 20971520] = [0x400000, 0x1400000].
template <int MF, int NF, int Panel, class Epilogue>
__global__ __launch_bounds__(32) void t2_prefill_wmma1_iu4_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t k_cols,
    std::int32_t t_pad, std::int32_t tokens, bool rows_fast, Epilogue epilogue) {
    constexpr int kGroupK  = 128; // one weight scale group, as in the shipped tile
    constexpr float kMagic = 12582912.0F; // 1.5 * 2^23

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    // Raster decode, identical to t2_prefill_wmma1_kernel's (if constexpr so the token-fast arm's
    // % panels_y is discarded, not just dead, in the served Panel >= 0 instantiations).
    int col0;
    int row0;
    if constexpr (Panel > 0) {
        col0 = static_cast<int>(blockIdx.y) * (16 * NF);
        row0 = (static_cast<int>(blockIdx.z) * Panel + static_cast<int>(blockIdx.x)) * (16 * MF);
    } else if constexpr (Panel < 0) {
        const int panels_y = static_cast<int>(gridDim.z) * (-Panel) * (16 * NF) / t_pad;
        col0 = (static_cast<int>(blockIdx.x) + (static_cast<int>(blockIdx.z) / panels_y) * (-Panel)) * (16 * NF);
        row0 = (static_cast<int>(blockIdx.y) + (static_cast<int>(blockIdx.z) % panels_y) *
                                                static_cast<int>(gridDim.y)) * (16 * MF);
    } else {
        col0 = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * (16 * NF);
        row0 = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * (16 * MF);
    }

    const int colWin = lane >> 1; // A-window column within the warp tile
    const int rowWin = lane & 15; // B-window row within the warp tile

    wmma_detail::int8v acc[MF][NF];
    float accf[MF][NF][8];
    const wmma_detail::int8v magic = {0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                                           0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
#pragma unroll
    for (int m = 0; m < MF; ++m) {
#pragma unroll
        for (int n = 0; n < NF; ++n) {
            acc[m][n] = magic;
#pragma unroll
            for (int e = 0; e < 8; ++e) { accf[m][n][e] = 0.0F; }
        }
    }

    // Row bases: weight lines (unchanged 2-bit payload, 32 code bytes per row per group), weight
    // scales (unchanged), activation windows at HALF the i8 stride.
    const std::size_t kRowBytes  = static_cast<std::size_t>(k_cols) / 4;
    const std::size_t kScaleCols = static_cast<std::size_t>(k_cols) / kGroupK;
    const std::size_t kHalf      = static_cast<std::size_t>(k_cols) / 2;
    const std::uint8_t* pA[MF];
    const __half* pS[MF];
    const std::int8_t* pB = x + static_cast<std::size_t>(col0 + colWin) * kHalf;
#pragma unroll
    for (int m = 0; m < MF; ++m) {
        pA[m] = w_codes + (static_cast<std::size_t>(row0) + static_cast<std::size_t>(rowWin) +
                            static_cast<std::size_t>(m) * 16) *
                               kRowBytes;
        pS[m] = w_scales + (static_cast<std::size_t>(row0) + static_cast<std::size_t>(rowWin) +
                             static_cast<std::size_t>(m) * 16) *
                                kScaleCols;
    }

    const int kGroups = k_cols / kGroupK;
    for (int g = 0; g < kGroups; ++g) {
        const std::size_t gBase  = static_cast<std::size_t>(g) * kGroupK;
        const std::size_t gBase4 = gBase / 2; // nibble-plane group base
#pragma unroll
        for (int m = 0; m < MF; ++m) {
#pragma unroll
            for (int n = 0; n < NF; ++n) { acc[m][n] = magic; }
        }

        // Weight lines for this group: MF lines of 32 bytes, one per B-window row -- the served
        // load, unchanged. Eight code words per line, one per s-step.
        wmma_detail::int8v line[MF];
#pragma unroll
        for (int m = 0; m < MF; ++m) { line[m] = *reinterpret_cast<const wmma_detail::int8v*>(pA[m] + gBase / 4); }

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            wmma_iu4_detail::int2v av[NF];
            wmma_iu4_detail::int2v bv[MF];
#pragma unroll
            for (int n = 0; n < NF; ++n) {
                av[n] = *reinterpret_cast<const wmma_iu4_detail::int2v*>(
                    pB + static_cast<std::size_t>(n) * (16 * kHalf) + gBase4 + s * 8);
            }
#pragma unroll
            for (int m = 0; m < MF; ++m) {
                int lw[8];
                *reinterpret_cast<wmma_detail::int8v*>(lw) = line[m];
                bv[m] = wmma_iu4_detail::wmma_code_frag4(static_cast<unsigned>(lw[s]));
            }
#pragma unroll
            for (int m = 0; m < MF; ++m) {
#pragma unroll
                for (int n = 0; n < NF; ++n) {
                    acc[m][n] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, av[n], true,
                                                                           bv[m], acc[m][n], true);
                }
            }
        }

        // Fold this group's dots: identical to the served kernel's sequence, xs hoisted ahead of
        // the m-loop once per group and shared across every m.
        float xsv[NF][8];
#pragma unroll
        for (int n = 0; n < NF; ++n) {
            const int cb = col0 + n * 16 + (lane >= 16 ? 8 : 0);
#pragma unroll
            for (int e = 0; e < 8; ++e) { xsv[n][e] = __half2float(xs[static_cast<std::size_t>(g) * t_pad + cb + e]); }
        }
#pragma unroll
        for (int m = 0; m < MF; ++m) {
            const float wv = __half2float(pS[m][g]);
#pragma unroll
            for (int n = 0; n < NF; ++n) {
                const int cBase = col0 + n * 16 + (lane >= 16 ? 8 : 0);
                const wmma_detail::int8v v = acc[m][n];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = cBase + e;
                    if (col < tokens) {
                        const float dot = __int_as_float(v[e]) - kMagic;
                        accf[m][n][e] = fmaf(dot, wv * xsv[n][e], accf[m][n][e]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MF; ++m) {
        const int row = row0 + m * 16 + rowWin;
#pragma unroll
        for (int n = 0; n < NF; ++n) {
            const int cBase = col0 + n * 16 + (lane >= 16 ? 8 : 0);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = cBase + e;
                if (col < tokens) { epilogue(row, col, accf[m][n][e]); }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // defined(__HIPCC__)
