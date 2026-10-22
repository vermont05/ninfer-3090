#pragma once

// GDN chunked-recurrence sites 2/3 on hardware WMMA (gfx1102), replacing the
// mma.cuh SIMT-emulated ldmatrix + mma_tf32_bits/mma_bf16 chains on the
// NT >= kWarps shape (DV = 64: eight warps, warp w owning one 8-column
// stripe; matrix_delta likewise runs per-warp on its own stripe).
//
// The instruction path and fragment contract are the shipped t2_prefill_wmma.cuh
// family's, pinned on this card by cal25/cal26/cal32/cal33/cal34/cal35
// (memory line gfx1102-wmma-f16-bf16-layout):
//   builtin(a, b, acc): the FIRST operand slot s is a 16-element column vector of
//     the right matrix, read at slots {0,2,..14,17,..31} (F-side map); the SECOND
//     operand slot s is row s of the left matrix, all 32 slots (slot s + 16
//     duplicates it) -> two aligned 16-byte chunks per lane (cal33 S-side map).
//     A column gather on the second operand yields the TRANSPOSED left matrix
//     (cal34 D: M[l][e] = block[e][cb + (l & 15)] = block^T slice).
//   C(l, q) = dot(Awin_slot[(l < 16 ? 0 : 17) + 2q], Bwin[l]) = (M * N)(l & 15, (l >= 16 ? 8 : 0) + q).
//   With an 8-column right window the second m16n8 tile mirrors the first
//   (cal35 H1/H2: mirror = 1): every lane's 32-byte result chunk carries row
//   (lane & 15) columns [cb, cb + 8), so the whole tile is one flat 32 x 32B
//   block read back with one aligned 32-byte LDS per lane.
//
// Every block-shaped value here (prediction, history, delta stripe, state
// tile) is exactly that: one 32-byte chunk per lane, lane l owning row
// (l & 15) -- the C fragment's own layout, so phase outputs feed the next
// phase's operand loaders with no shuffles. The existing staged layouts
// (vector_index qk blocks, value_index snapshot/delta, square_index
// solve/mqk) equal the probe-pinned loader layouts, so operands are read in
// place. Narrow (8-column) F windows use the cal35 H map j = (lane & 15) >> 1.
//
// Results are not bit-identical to the emulation: deterministic 1-3 ULP
// instruction noise plus TF32 -> BF16 on the prediction dot (cal25/cal26/cal33/
// cal34/cal35 batteries: 0 mismatches vs the dot models with tolerance; the
// emulation itself is exact TF32). Gate: NINFER_GDN_REC_WMMANATIVE.

#if defined(__HIPCC__)

#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace wmma_detail {

using i16v16 = short __attribute__((__vector_size__(32)));
using f8v32  = float __attribute__((__vector_size__(32)));

// Staged-block layouts (element (row, col) offset). value_index<DV> in
// recurrence.cu equals LayV64 / LayV32 / LayV16 for the matching DV, and
// vector_index / square_index are the QK and solve/mqk layouts verbatim.
struct LayV128 {
    static __device__ __forceinline__ int index(int row, int col) { return vector_index(row, col); }
};
struct LayV64 {
    static __device__ __forceinline__ int index(int row, int col) { return row * 64 + (col ^ ((row & 3) * 8)); }
};
struct LayV32 {
    static __device__ __forceinline__ int index(int row, int col) { return row * 32 + (col ^ ((row & 3) * 8)); }
};
struct LaySq {
    static __device__ __forceinline__ int index(int row, int col) { return square_index(row, col); }
};

__device__ __forceinline__ constexpr int narrow_j(int lane) { return (lane & 15) >> 1; }

// ---------------- primitives (cal33/34/35 loader shapes) ----------------

// NOTE: all loaders take PLAIN loads (the probe bodies do) — the operands live
// in __shared__ here, and __ldg is a global read-only-path load.
__device__ __forceinline__ unsigned short load_bits(const __nv_bfloat16* p, int idx) {
    return reinterpret_cast<const unsigned short*>(p)[idx];
}
__device__ __forceinline__ unsigned short load_bits(const float* p, int idx) {
    return __builtin_bit_cast(std::uint16_t,
                               __float2bfloat16_rn(p[idx])); // RN-to-bf16; 16-bit raw
}

// S-side (second argument): row (rowbase + (lane & 15)) of the left matrix,
// columns [colbase, colbase + 16) as two aligned 16-byte chunks (cal33 B).
template <class Lay>
__device__ __forceinline__ i16v16 s_window(const __nv_bfloat16* p, int rowbase, int colbase,
                                           int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    *reinterpret_cast<int4*>(h) =
        *reinterpret_cast<const int4*>(p + Lay::index(rowbase + (lane & 15), colbase));
    *reinterpret_cast<int4*>(h + 8) =
        *reinterpret_cast<const int4*>(p + Lay::index(rowbase + (lane & 15), colbase + 8));
    return out;
}
// Square (16-wide) block row window: one 32-byte piece per lane, scalar-filled
// (cal34 G shape; the square_index row swizzle is inside the 8-block).
__device__ __forceinline__ i16v16 s_window_sq(const float* p, int rowbase, int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
    for (int e = 0; e < 16; ++e) h[e] = load_bits(p, LaySq::index(rowbase + (lane & 15), e));
    return out;
}
// S-side column gather (cal34 D): slot l holds column (colbase + (lane & 15)) rows
// [rowbase, rowbase + 16), i.e. the TRANSPOSED left matrix (block^T slice).
__device__ __forceinline__ i16v16 s_colongather(const __nv_bfloat16* p, int rowbase, int colbase,
                                                int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int col = colbase + (lane & 15);
#pragma unroll
    for (int r = 0; r < 16; ++r) h[r] = load_bits(p, LayV128::index(rowbase + r, col));
    return out;
}

// F-side (first argument): column j = (lane & 15) >> 1 of rows [rowbase, rowbase + 16)
// (cal35 H narrow map -- the right matrix's 8-column window; slots 16..31 mirror).
template <class Lay>
__device__ __forceinline__ i16v16 f_narrow(const __nv_bfloat16* p, int rowbase, int colbase,
                                           int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int j = narrow_j(lane);
#pragma unroll
    for (int r = 0; r < 16; ++r) h[r] = load_bits(p, Lay::index(rowbase + r, colbase + j));
    return out;
}
template <class Lay>
__device__ __forceinline__ i16v16 f_narrow(const float* p, int rowbase, int colbase, int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int j = narrow_j(lane);
#pragma unroll
    for (int r = 0; r < 16; ++r) h[r] = load_bits(p, Lay::index(rowbase + r, colbase + j));
    return out;
}
// Site-3 right operand: the delta stripe column scaled rowwise by row_scale[row]
// (diag(suffix) Delta -- the SIMT path's delta_b packing, moved into the loader).
template <class Lay>
__device__ __forceinline__ i16v16 f_narrow_scaled(const float* p, int colbase, int lane,
                                                  const float* row_scale) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int j = narrow_j(lane);
#pragma unroll
    for (int r = 0; r < 16; ++r)
        h[r] = __builtin_bit_cast(
            std::uint16_t,
            __float2bfloat16_rn(row_scale[r] * p[Lay::index(r, colbase + j)]));
    return out;
}

// Full-width F-side window (cal33/34's F loader shape): slots {0,2,..14} carry
// columns 0..7's column-vectors and {17,..31} columns 8..15's, one per lane pair.
template <class Lay>
__device__ __forceinline__ i16v16 f_square(const __nv_bfloat16* p, int rowbase, int colbase,
                                           int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int j = (lane < 16) ? (lane >> 1) : (8 + ((lane - 16) >> 1));
#pragma unroll
    for (int r = 0; r < 16; ++r) h[r] = load_bits(p, Lay::index(rowbase + r, colbase + j));
    return out;
}
template <class Lay>
__device__ __forceinline__ i16v16 f_square(const float* p, int rowbase, int colbase, int lane) {
    i16v16 out;
    auto* h = reinterpret_cast<unsigned short*>(&out);
    const int j = (lane < 16) ? (lane >> 1) : (8 + ((lane - 16) >> 1));
#pragma unroll
    for (int r = 0; r < 16; ++r) h[r] = load_bits(p, Lay::index(rowbase + r, colbase + j));
    return out;
}

__device__ __forceinline__ f8v32 mma_bf16(const i16v16& a, const i16v16& b, const f8v32& acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
}

// C fragment round-trip: slot l = lane l's 32-byte chunk (row l & 15). The mirror
// half (16..31) writes the same bytes as its twin, so guarded stores to a
// 16-slot block are the block contract (cal33/35).
__device__ __forceinline__ void c_store(float* block, int lane, const f8v32& v) {
    if (lane < 16) reinterpret_cast<f8v32*>(block)[lane] = v;
}
__device__ __forceinline__ f8v32 c_load(const float* block, int lane) {
    f8v32 out{0, 0, 0, 0, 0, 0, 0, 0};
    if (lane < 16) out = reinterpret_cast<const f8v32*>(block)[lane];
    return out;
}
// The same chunk written at its swizzled slot (the staged-block layouts):
// Lay::index(row, colbase) keeps 32-byte alignment for 8-column colbase, so
// a fragment chunk goes straight into a Lay block for the next phase.
template <class Lay>
__device__ __forceinline__ void store_chunk8(float* block, int rowbase, int colbase, int lane,
                                              const f8v32& v) {
    if (lane < 16) *reinterpret_cast<f8v32*>(block + Lay::index(rowbase + (lane & 15), colbase)) = v;
}
template <class Lay>
__device__ __forceinline__ f8v32 load_chunk8(const float* block, int rowbase, int colbase,
                                              int lane) {
    f8v32 out{0, 0, 0, 0, 0, 0, 0, 0};
    if (lane < 16)
        out = *reinterpret_cast<const f8v32*>(block + Lay::index(rowbase + (lane & 15), colbase));
    return out;
}

template <int DV>
struct LayFor;
template <>
struct LayFor<64> {
    using type = LayV64;
};
template <>
struct LayFor<32> {
    using type = LayV32;
};

} // namespace wmma_detail

namespace wmma_detail {

// ---------------- site phases (NT >= kWarps shapes) ----------------
// A "block" is 16 slots x 32B (16 rows x 8 columns, slot = lane); c_store/c_load
// move chunks. Warp w owns stripe column base w * 8 (n_start = warp % NT).

// Site 2 (NT >= kWarps branch): prediction = K . S and history = Q . S chained
// over the eight 16-key slices of the [128 keys][DV] snapshot stripe
// [colbase, colbase + 8). Mirrors the SIMT loop's two A windows + B snapshot.
template <int DV>
__device__ __forceinline__ void site2_pred_history(const QkChunk& qk, const float* snapshot,
                                                   int colbase, int lane, f8v32& prediction,
                                                   f8v32& history) {
    using Lay = typename LayFor<DV>::type;
    f8v32 ap{0, 0, 0, 0, 0, 0, 0, 0};
    f8v32 ah{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < kStateDim; k += 16) {
        const i16v16 sk = s_window<LayV128>(qk.k, 0, k, lane);
        const i16v16 sq = s_window<LayV128>(qk.q, 0, k, lane);
        const i16v16 sn = f_narrow<Lay>(snapshot, k, colbase, lane);
        ap = mma_bf16(sn, sk, ap);
        ah = mma_bf16(sn, sq, ah);
    }
    prediction = ap;
    history    = ah;
}

// Site 2 consumption (WMMA store_residual): lane owns the prediction chunk for
// row (lane & 15); the delta chunk value[row][colbase..+8) - prefix[row] * pred
// goes back to the block slot at Lay::index(row, colbase) (32-byte aligned).
template <class Lay, int DV>
__device__ __forceinline__ void store_residual_wmma(float* delta, const __nv_bfloat16* value,
                                                    const f8v32& pred, int colbase, int lane,
                                                    const float* prefix) {
    if (lane >= 16) return;
    const int row = lane;
    const float scale = prefix[row];
    i16v16 v;
    *reinterpret_cast<int4*>(&v) = *reinterpret_cast<const int4*>(value + row * DV + colbase);
    const auto* vh = reinterpret_cast<const __nv_bfloat16*>(&v);
    f8v32 out;
    auto* oh = reinterpret_cast<float*>(&out);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(load_vec<__nv_bfloat162>(vh + 2 * q));
        oh[2 * q] = f.x - scale * pred[2 * q];
        oh[2 * q + 1] = f.y - scale * pred[2 * q + 1];
    }
    *reinterpret_cast<f8v32*>(delta + Lay::index(row, colbase)) = out;
}

// matrix_delta's shape: a square_index 16x16 matrix (left; cal34 G row
// window) times the [16 rows][8 cols] stripe block (right, Lay window).
// Both matrix_delta call sites (solve x residual, mqk x delta) are this
// single tile, accumulated into acc.
template <class Lay>
__device__ __forceinline__ f8v32 dot_square(const float* square, const float* stripe,
                                            int colbase, int lane, const f8v32& acc) {
    return mma_bf16(f_narrow<Lay>(stripe, 0, colbase, lane), s_window_sq(square, 0, lane), acc);
}

// matrix_delta's WMMA shape for an F-side matrix laid out in Lay (square_index for
// the solve matrix, vector_index for the MQK matrix): one 16x16x16 dot, left = the
// matrix's 16x16 row window, right = the 16x8 stripe column-gather at colbase. The
// result block (16x8) round-trips through c_store/c_load, so chained results feed
// dot_square's next call with no shuffles.
template <class Lay, class T>
__device__ __forceinline__ f8v32 dot_block(const T* matrix, const float* stripe, int colbase,
                                           int lane, const f8v32& acc) {
    return mma_bf16(f_narrow<Lay>(stripe, 0, colbase, lane), s_window_sq(matrix, 0, lane), acc);
}

// Site-3 tile: state += K^T[keybase, keybase + 16) . (suffix . Delta). The K^T
// slice is the column-gather transposed left matrix (cal34 D / cal35 H2);
// the right window folds the per-row suffix into the delta stripe load
// (the SIMT delta_b packing). Chained over keybase slices.
template <class Lay>
__device__ __forceinline__ f8v32 state_tile(const __nv_bfloat16* k, int keybase,
                                             const float* delta, int colbase, int lane,
                                             const float* suffix, const f8v32& acc) {
    return mma_bf16(f_narrow_scaled<Lay>(delta, colbase, lane, suffix),
                    s_colongather(k, 0, keybase, lane), acc);
}

// Output store, the block-shape mirror of store_output: lane owns the row
// (lane & 15) chunk (8 columns), one guarded 16-byte global store per active
// lane (the SIMT helper does two float2 stores covering the same row).
__device__ __forceinline__ void store_chunk_row(const Arguments& args, const f8v32& acc,
                                                 int chunk, int head, int value_start,
                                                 int colbase, int lane) {
    const int row = lane & 15;
    const int t0  = chunk * kChunkSize + row;
    if (lane >= 16 || t0 >= args.tokens) return;
    const auto o0 = (static_cast<std::int64_t>(t0) * args.value_heads + head) * kStateDim +
                      value_start + colbase;
    unsigned words[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        words[q] = pack_bf16x2(acc[2 * q] * args.scale, acc[2 * q + 1] * args.scale);
    *reinterpret_cast<int4*>(args.out + o0) = *reinterpret_cast<const int4*>(words);
}

} // namespace wmma_detail

} // namespace ninfer::ops::detail::gated_delta_net::chunked

#endif // defined(__HIPCC__)
