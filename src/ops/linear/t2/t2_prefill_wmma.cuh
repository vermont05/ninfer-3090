#pragma once

// T2G128 RowSplit x int8 prefill GEMM, gfx1102 WMMA variant: the shipped t2_prefill_i8.cuh
// tile's operand contract and fold sequence, with one hardware v_wmma_i32_16x16x16_iu8 per
// 16x16 output tile in place of the mma.cuh shuffle-gather emulation. cal16 pinned the
// instruction's fragment semantics on this card (cal16c/e/f, .qual/wmma batteries pass) and
// cal18 turned them into a tiled GEMM. This kernel is that loader with the weight side
// decoded on the fly from the T2 payload -- one warp per block, cal23's fastest shape.
//
// Why one warp per block (cal21/22/23, all bit-identical batteries vs the i8 anchor):
// the 4-warp block-tile version (cal18's loader, 128 threads, per-group hoisted av)
// allocated past the 256-VGPR ceiling and spilled ~470 VGPRs to scratch, running ~2.4x
// slower than the same loader standalone. Splitting the block tile into one-warp tiles
// AND hoisting the fold's xs activation-scale loads ahead of the m-loop keeps every
// used tile spill-free at <= 223 VGPRs and is 1.8-4x faster than the 4-warp block on the
// engine's three shapes (cal22, cal23). The xs hoist is what buys it: without it even
// (2,2) and (4,2) one-warp tiles spill.
//
// Fragment semantics (cal16c/e/f, reused by cal18's loader):
//   Awin slot s = activation token (s >> 1)'s 16-code k-window (slot s == lane s's register;
//                 the odd and 16 slots load duplicates that are never read).
//   Bwin slot s = weight row (s & 15)'s decoded 16-code window (the s >= 16 slots duplicate).
//   C(l, w)     = dot(Awin[(l < 16 ? 0 : 17) + 2w], Bwin[l]) = (W * X)(l & 15, (l >= 16 ? 8 : 0) + w).
// A weight row's group line is the 32 code bytes at w_codes + row * (K / 4) + g * 32 -- the
// same bytes the shipped tile cp_async's, in the same order: its eight 2-bit words carry
// the group's 128 codes, word i holds k in [16 i, 16 i + 16) in the fragment byte order,
// and t2_prefill_decode(word_i) yields that 16-code window as one 16-byte register. The
// i32 accumulator starts at the bits of 1.5 * 2^23 (the shipped magic-bits idiom): as a
// float it reads that plus the dot over the whole group, so one bitcast and a subtract
// recover (float)dot, and the fold is the shipped kernel's sequence exactly -- group dot
// -> float -> fma(dot, w_scale * x_scale, acc) with both scales as the stored binary16.
// Per-group operands equal the shipped kernel's, so outputs are bit-identical.
//
// Activations and scales use t2_prefill_i8.cuh's contract verbatim (codes [Tpad][K] s8 in
// the permuted order, scales [K/128][Tpad] binary16, padded columns zero), so the
// quantiser and every epilogue are shared unchanged. Loads are plain per-lane 16B/32B
// reads from global -- the shipped staging's address pattern without the shared round
// trip (RDNA3 has no async copy to hide it anyway).

#if defined(__HIPCC__)

#include "ops/linear/t2/t2_prefill_i8.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace wmma_detail {

using int4v = int __attribute__((__vector_size__(16)));
using int8v = int __attribute__((__vector_size__(32)));

// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) widened to the
// 16-byte fragment register byte j = table byte for code j (table bytes 0, +1, -2, -1;
// nibble < 4 selects a table byte, nibble >= 4 a zero byte).
__device__ __forceinline__ int4v wmma_code_frag(unsigned word) {
    constexpr unsigned kTable = 0xFFFE0100u;
    const unsigned even       = word & 0x33333333u;
    const unsigned odd        = (word >> 2) & 0x33333333u;
    const int a                 = static_cast<int>(__byte_perm(kTable, 0u, even));
    const int b                 = static_cast<int>(__byte_perm(kTable, 0u, odd));
    const int c                 = static_cast<int>(__byte_perm(kTable, 0u, even >> 16));
    const int d                 = static_cast<int>(__byte_perm(kTable, 0u, odd >> 16));
    return {a, b, c, d};
}

} // namespace wmma_detail

// One warp computes one (16 * MF) x (16 * NF) tile; the block is that warp (cal23's
// fastest family). (MF, NF) = (1, 2): 16 rows x 32 columns, 128 VGPRs, beats the
// 4-warp block ~3.9x on the engine shapes (gate_up 28.7 ms vs 112.7). Grid and raster
// follow t2_prefill_i8_kernel's convention: one block index per tile, (column blocks,
// row blocks) or the reverse under rows_fast. Callers must size grids as
// (columns / (16 * NF), rows / (16 * MF)) or the swapped pair.
//
// Panel mode (Panel > 0) reorders the same tiles for L2 residency (cal41's G32, the
// cal39/cal41 winner at +6-8% on every engine shape, bit-exact on all battery shapes):
// grid = (Panel, column blocks, row blocks / Panel), token block = blockIdx.y, and the
// fast axis walks `Panel` row tiles inside one panel, so row0 = (z*Panel + x) * 16 * MF.
// Consecutive blocks then share a token block and sweep a Panel*16*MF-row weight panel:
// at Panel=32/MF=1 the in-flight set is one 1024-row weight panel (1.31 MB at K=5120)
// plus the hot 32-token activation block, ~1.47 MB inside the 2 MB L2. rows_fast is
// unused in panel mode; geometry must satisfy rows % (16 * MF * Panel) == 0.
template <int MF, int NF, int Panel, class Epilogue>
__global__ __launch_bounds__(32, 8) void t2_prefill_wmma1_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t k_cols,
    std::int32_t t_pad, std::int32_t tokens, bool rows_fast, Epilogue epilogue) {
    using namespace wmma_detail;
    constexpr int kGroupK = 128; // one weight scale group, as in the shipped tile
    constexpr float kMagic = 12582912.0F; // 1.5 * 2^23

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int col0 = Panel > 0
                         ? static_cast<int>(blockIdx.y) * (16 * NF)
                         : static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * (16 * NF);
    const int row0 = Panel > 0
                         ? (static_cast<int>(blockIdx.z) * Panel + static_cast<int>(blockIdx.x)) * (16 * MF)
                         : static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * (16 * MF);

    const int colWin = lane >> 1; // Awin window column within the warp tile
    const int rowWin = lane & 15; // Bwin slot row within the warp tile

    int8v acc[MF][NF];
    float accf[MF][NF][8];
    const int8v magic = {0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
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

    // Row bases: weight lines, weight scales, activation windows (cal18's loader; the
    // odd and 16 activation slots duplicate rows the instruction never reads).
    const std::size_t kRowBytes  = static_cast<std::size_t>(k_cols) / 4;
    const std::size_t kScaleCols = static_cast<std::size_t>(k_cols) / kGroupK;
    const std::uint8_t* pA[MF];
    const __half* pS[MF];
    const std::int8_t* pB = x + static_cast<std::size_t>(col0 + colWin) * k_cols;
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
        const std::size_t gBase = static_cast<std::size_t>(g) * kGroupK;
#pragma unroll
        for (int m = 0; m < MF; ++m) {
#pragma unroll
            for (int n = 0; n < NF; ++n) { acc[m][n] = magic; }
        }

        // Weight lines for this group: MF lines of 32 bytes, one per Bwin row.
        int8v line[MF];
#pragma unroll
        for (int m = 0; m < MF; ++m) { line[m] = *reinterpret_cast<const int8v*>(pA[m] + gBase / 4); }

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int4v av[NF];
            int4v bv[MF];
#pragma unroll
            for (int n = 0; n < NF; ++n) {
                av[n] = *reinterpret_cast<const int4v*>(pB + static_cast<std::size_t>(n) * (16 * k_cols) +
                                                          gBase + s * 16);
            }
#pragma unroll
            for (int m = 0; m < MF; ++m) {
                int lw[8];
                *reinterpret_cast<int8v*>(lw) = line[m];
                bv[m] = wmma_detail::wmma_code_frag(static_cast<unsigned>(lw[s]));
            }
#pragma unroll
            for (int m = 0; m < MF; ++m) {
#pragma unroll
                for (int n = 0; n < NF; ++n) {
                    acc[m][n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av[n], true,
                                                                           bv[m], acc[m][n], true);
                }
            }
        }

        // Fold this group's dots. The xs column loads are hoisted ahead of the m-loop
        // once per group (cal22: this is what keeps the tile off the spill heap) and
        // shared across every m. Then bits -> float, scale by ws[row] * xs[col], fma
        // into the float accumulator (the shipped fold's sequence).
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
                const int8v v   = acc[m][n];
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

    // Epilogue: every tile's eight fragment words, columns < tokens (padded columns are
    // zero and contribute nothing).
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
