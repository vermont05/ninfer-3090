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
// Written as the hardware permute (V_PERM_B32) directly, not as __byte_perm. RDNA3's PRMT selector
// is immediate-encoded, so __byte_perm with a data-derived selector cannot become PRMT; what LLVM
// does instead is decided **per kernel**, and that is measured, not assumed: this kernel's byte_perm
// form paid 32 ds_store_b32 + 128 ds_load_u8, the address adds feeding them and 111 s_waitcnt in a
// body of 1365 instructions (census below), while the *same idiom* in t2_prefill_i8.cuh's
// t2_prefill_decode - same kTable, same 0x33333333 masks, same translation unit - lowers to 16
// v_perm_b32 with only 19 ds_load / 9 ds_store in t2_prefill_i8_kernel. So LDS staging is LLVM's
// choice for this kernel, not what the idiom forces; asking for the permute explicitly is what
// removes it. V_PERM_B32 takes its selector from a register, so the lookup is one VALU op per
// fragment register.
// The table is passed as both source operands: for selectors 0..3 the selected byte is a table byte
// whichever operand the ISA treats as the low half, so the result does not depend on operand order.
// Selector layout: V_PERM_B32 consumes ONE SELECTOR BYTE per output byte (PRMT consumes one nibble),
// so each code has to sit alone in its byte lane before the permute, and the selector byte must stay
// in 0..3. The nibble-mask form (`word & 0x0F0F0F0F`) puts nibble j - code 2j plus 4 * code 2j+1 -
// into byte j, which packs values 8..15; measured exhaustively over all 2^32 code words, 4253024256
// of them mismatch the shipped widening under the ISA's strictest reading (index = selector_byte &
// 0x07, force 0 when selector_byte & 0xF8). The splay below masks each code pair to its own byte
// lane's low two bits, so every selector byte is a code value: bits 2..7 clear, index = code under
// both the 0x07 and 0x0F index readings, no forced zero under either the 0xF0 or the 0xF8 reading,
// and with the table in both source operands the result cannot depend on which operand the ISA
// treats as the low half. Proven exhaustively (proof-splay2.c, TOTAL mismatches = 0 over
// 4294967296 words x 4 fragment registers x 4 readings) under all four readings at once.
//
// What LLVM actually emits for spread4 (gfx1102, -O3): each shl+and term lowers to a shift plus a
// mask, and the last term folds the mask into the merge as v_and_or_b32 - v_bfm is NOT used, contrary
// to the first guess when this was written.
//
// Census of the served kernel body (t2_prefill_wmma1_kernel<1,2,Panel=32,SplitStoreEpilogue>), shipped
// form -> this form, counted by ONE parser over the body terminated at its .Lfunc_end label
// (/home/vermont/tmp/t2-ab/census-splay.py -> census-base.txt / census-cand.txt):
//   instructions        1365 -> 1034   (-331, -24.3%)
//   ds_load              128 -> 0      ds_store   32 -> 0     (all 160 DS ops gone)
//   s_waitcnt            111 -> 16     v_add      209 -> 81   (the LDS address math goes with them)
//   v_bfe                130 -> 40     v_perm       0 -> 32
//   v_and                  9 -> 104    v_lshl 123 -> 75   v_lshr 32 -> 72
//   v_wmma 16, global_load 24, global_store 16 - unchanged; VGPR 128 -> 123, sgpr 43,
//   private segment 36 B, scratch load/store 2+2 - no new spills.
// v_perm is 32, not the 64 a site count predicts (16 v_wmma sites x 4 B-fragment registers): LLVM
// common-subexpressions repeated permutes across the K-steps. That is the measured count; the
// mechanism is inferred. The five sibling instantiations in this TU move the same way (1301 -> 970,
// 1301 -> 969, 1385 -> 1051, 1385 -> 1052, 1365 -> 1035; DS -> 0 in every one).
// An earlier version of this comment quoted 2056 -> 1325 body lines with 256/64 DS ops and 64 perms.
// That came from an ad-hoc diff script whose region started outside the kernel (its DS counts were
// exactly 2x the verified ones); it is recorded here only as a warning: a census parser must
// terminate at .Lfunc_end and must not anchor on a bare label match, because GCN labels carry a
// trailing " ; @mangled-name", and must let instruction suffixes match (a \bv_perm\b regex matches
// nothing at all - the instruction is v_perm_b32).
//
// Served-route qualification (measured, 4-leg ABBA B,C,C,B in one flock hold, 1024-token lighthouse
// prompt, temperature 0, 128 max tokens, tag perm2, 21:43-21:47): prefill 178.1 / 175.2 tok/s shipped
// vs 203.4 / 202.5 tok/s with this widening - mean 176.65 -> 202.95, +14.9%, with base-leg spread
// 2.9 tok/s. Generated text is identical on all four legs (sha1 77f1779eb831, 574 chars, 93 words /
// 59 distinct, max 8-gram repeat 1), decode 13.0 tok/s both arms. Details: docs/remaining-work.md
// section 15b.
__device__ __forceinline__ unsigned spread4(unsigned w) {
    return (w & 0x00000003u) | ((w << 4) & 0x00000300u) | ((w << 8) & 0x00030000u) | ((w << 12) & 0x03000000u);
}

__device__ __forceinline__ int4v wmma_code_frag(unsigned word) {
    constexpr unsigned kTable = 0xFFFE0100u; // 0, +1, -2, -1
    const int a                 = static_cast<int>(__builtin_amdgcn_perm(kTable, kTable, spread4(word)));
    const int b                 = static_cast<int>(__builtin_amdgcn_perm(kTable, kTable, spread4(word >> 2)));
    const int c                 = static_cast<int>(__builtin_amdgcn_perm(kTable, kTable, spread4(word >> 16)));
    const int d                 = static_cast<int>(__builtin_amdgcn_perm(kTable, kTable, spread4(word >> 18)));
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
//
// Token-fast panel mode (Panel < 0, let BC = -Panel) is the same panelisation with the two
// axes swapped, which is what the DRAM traffic model says the served raster is missing.
// grid = (BC, RC, panels_y * column_blocks / BC) with panels_y = row_blocks / RC, and
//   col0 = (x + (z / panels_y) * BC) * 16 * NF,   row0 = (y + (z % panels_y) * RC) * 16 * MF
// so the fast axis walks BC *token* blocks inside one panel while y sweeps RC row tiles:
// every block dispatched at the same z shares one RC-tile weight panel and streams it over
// BC token blocks. Weight bytes are re-streamed column_blocks/BC times (not column_blocks
// times) and the activation plane row_blocks/RC times, i.e. traffic
//   w.n*K/4 * (column_blocks/BC) + T*K * (row_blocks/RC)
// against the served raster's w.n*K/4 * column_blocks. At gate_up's 2176x5120x1024 (32
// column blocks, 2176 row tiles) the served 1.43 GB -> BC=8/RC=34: 178 + 166 = 344 MB.
// The in-flight set is BC*160 KB of activations + RC*20.5 KB of weight, so the pair must
// fit the 2 MB L2 (BC=8,RC=34 -> 1.98 MB; BC=8,RC=17 -> 1.63 MB; BC=6,RC=32 -> 1.62 MB).
// Coverage laws: BC | column_blocks and RC | row_blocks; z indexes
// (col panel, row panel) = (z / panels_y, z % panels_y), so each tile is visited exactly
// once. panels_y is recovered from gridDim.z * BC / column_blocks -- exact under the
// coverage law, and it needs no new kernel argument. Zero register cost (two scalar
// divisions per block, none per group); warp parallelism is untouched, which is the
// failure mode the LDS-staged loader family (w1c/wlds/wcoop) hit.
template <int MF, int NF, int Panel, class Epilogue>
__global__ __launch_bounds__(32, 8) void t2_prefill_wmma1_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t k_cols,
    std::int32_t t_pad, std::int32_t tokens, bool rows_fast, Epilogue epilogue) {
    using namespace wmma_detail;
    constexpr int kGroupK = 128; // one weight scale group, as in the shipped tile
    constexpr float kMagic = 12582912.0F; // 1.5 * 2^23

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    // if constexpr (not a ternary) so the token-fast decode is discarded — not merely dead — in
    // every other mode: a ternary leaves a % panels_y with panels_y = 0 in the discarded arm, which
    // the backend warns about (-Wdivision-by-zero) in every served Panel >= 0 instantiation.
    int col0;
    int row0;
    if constexpr (Panel > 0) {
        col0 = static_cast<int>(blockIdx.y) * (16 * NF);
        row0 = (static_cast<int>(blockIdx.z) * Panel + static_cast<int>(blockIdx.x)) * (16 * MF);
    } else if constexpr (Panel < 0) {
        // panels_y = gridDim.z / (column_blocks / BC), exact under the coverage law BC | column_blocks.
        const int panels_y = static_cast<int>(gridDim.z) * (-Panel) * (16 * NF) / t_pad;
        col0 = (static_cast<int>(blockIdx.x) + (static_cast<int>(blockIdx.z) / panels_y) * (-Panel)) * (16 * NF);
        row0 = (static_cast<int>(blockIdx.y) + (static_cast<int>(blockIdx.z) % panels_y) *
                                                static_cast<int>(gridDim.y)) * (16 * MF);
    } else {
        col0 = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * (16 * NF);
        row0 = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * (16 * MF);
    }

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

// Column-split variant (cal64/cal64b): a 64-thread block whose two warps split the token
// columns of a (16*MF) x (16*NF*2) block tile and share its row panel. Each warp computes one
// (16*MF) x (16*NF) tile exactly as t2_prefill_wmma1_kernel does, fed by WIDE loads: one
// 32 B/lane int8v load supplies two consecutive s-slots and is byte-identical per lane to
// that kernel's two 16 B loads, so fragment pairing — and every output bit — is preserved.
// The win is layout (cal60: requests are sector-bound, so only layout moves them): the two
// warps touch DISJOINT token windows (columns are tokens), halving per-block x sectors, and
// the panel raster keeps them hot (P=32: 975M L2 hits against 24M misses).
//
// ROW COVERAGE LAW (pinned by cal64b's verify runs): a block covers 16*MF rows, NOT 16*MF*2.
// Callers must size the grid as (rows / (16*MF*Panel), columns / (16*NF*2), Panel) — tile
// stride 16*MF, grid.x = column blocks — and the shape must satisfy rows % (16*MF*Panel) == 0
// and columns % (16*NF*2) == 0. cal64 measured a 22.8-50.4% coverage bug with a 16*MF*2
// stride; its timings are void.
//
// Verified BIT-EXACT against t2_prefill_wmma1_kernel on identical buffers — 35,651,584 /
// 35,651,584 outputs, both passes, gate_up 34816x5120x1024 — for all five shapes swept
// ((2,1) at P=8/16/32, (4,1) at P=8/16). The gate must be the engine kernel itself: T2
// activations are stored in quantiser-permuted order, so a scalar identity-pairing reference
// on synthetic data diverges from BOTH kernels by design (cal64b: 1/35,651,584 match).
// (4,1) shapes spill 112 B — do not pursue. 86 VGPR, 0 private segment, occupancy 8.
//
// VERDICT (cal65/cal66, closed 2026-10-23): the cal64b probe win was a coverage artifact — its
// timing grid carried cal63/64's overlap-trick extra /2 on grid.x, measuring half the rows, so
// 23.14 ms is half-work. Full-coverage timing (cal66, grid.x = N/(16*MF*P), flush per rep, same
// epilogue both arms): (2,1) P=32 loses to the shipped (1,2) G32 raster at every engine shape —
// gate_up 45.95 vs 28.77 ms, down 24.08 vs 14.61, o_proj 6.94 vs 4.19, flat across P=8/16/32 —
// reproducing the engine's -27% ABBA (104.2 vs 142.4 tok/s, bit-exact text). Mechanism: halving
// x windows keeps warp count identical, so each weight row is read by 2 warps — weight traffic
// doubles, and at 2.125 bpw weights dominate. The coverage law applies to TIMING grids too.
// Routed by NINFER_T2_A8_COLSPLIT in t2_a8.cu, default off; the lane is closed.
template <int MF, int NF, int Panel, class Epilogue>
__global__ __launch_bounds__(64, 8) void t2_prefill_wmma2_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t k_cols,
    std::int32_t t_pad, std::int32_t tokens, bool rows_fast, Epilogue epilogue) {
    using namespace wmma_detail;
    constexpr int kGroupK = 128; // one weight scale group, as in the shipped tile
    constexpr float kMagic = 12582912.0F; // 1.5 * 2^23

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    (void)rows_fast; // panel raster only; grid is the caller's (see the coverage law above)
    const int col0 = static_cast<int>(blockIdx.y) * (16 * NF * 2) + warp * (16 * NF);
    const int row0 = Panel > 0
                         ? (static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x)) * (16 * MF)
                         : static_cast<int>(blockIdx.x) * (16 * MF);

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

    // Row bases, walked per group (the cal59 D22 diet — no reindexed address math): weight
    // lines, weight scales, activation windows. pBOff is the NF token-window row offset.
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
    std::size_t pBOff[NF];
#pragma unroll
    for (int n = 0; n < NF; ++n) { pBOff[n] = static_cast<std::size_t>(n) * 16 * k_cols; }

    const int kGroups = k_cols / kGroupK;
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
        for (int m = 0; m < MF; ++m) {
#pragma unroll
            for (int n = 0; n < NF; ++n) { acc[m][n] = magic; }
        }

        // Weight lines for this group: MF lines of 32 bytes, one per Bwin row.
        int8v line[MF];
#pragma unroll
        for (int m = 0; m < MF; ++m) { line[m] = *reinterpret_cast<const int8v*>(pA[m]); pA[m] += kGroupK / 4; }

        // Four wide-load steps, each feeding the s-pair (2*sp, 2*sp+1): the int8v read is
        // 32 B/lane, its two int4v halves are the engine kernel's av[s] bytes verbatim.
#pragma unroll
        for (int sp = 0; sp < 4; ++sp) {
            int4v av[NF][2];
#pragma unroll
            for (int n = 0; n < NF; ++n) {
                const int8v wide = *reinterpret_cast<const int8v*>(pB + pBOff[n] + sp * 32);
                av[n][0] = {wide[0], wide[1], wide[2], wide[3]};
                av[n][1] = {wide[4], wide[5], wide[6], wide[7]};
            }
            int4v bv[MF][2];
#pragma unroll
            for (int m = 0; m < MF; ++m) {
                int lw[8];
                *reinterpret_cast<int8v*>(lw) = line[m];
                bv[m][0] = wmma_code_frag(static_cast<unsigned>(lw[2 * sp]));
                bv[m][1] = wmma_code_frag(static_cast<unsigned>(lw[2 * sp + 1]));
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int m = 0; m < MF; ++m) {
#pragma unroll
                    for (int n = 0; n < NF; ++n) {
                        acc[m][n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av[n][j],
                                                                               true, bv[m][j], acc[m][n], true);
                    }
                }
            }
        }
        pB += kGroupK;

        // Fold this group's dots — the shipped sequence: xs hoisted per group, bits ->
        // float, scale by ws[row] * xs[col], fma. Shared xs columns make the column split
        // free on the scale side too.
        float xsv[NF][8];
#pragma unroll
        for (int n = 0; n < NF; ++n) {
            const int cb = col0 + n * 16 + (lane >= 16 ? 8 : 0);
            const __half* xp = xs + static_cast<std::size_t>(g) * t_pad + cb;
#pragma unroll
            for (int e = 0; e < 8; ++e) { xsv[n][e] = __half2float(*xp); xp += 1; }
        }
#pragma unroll
        for (int m = 0; m < MF; ++m) {
            const float wv = __half2float(*pS[m]); pS[m] += 1;
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
