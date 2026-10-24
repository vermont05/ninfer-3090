#pragma once
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"

namespace ninfer::ops::detail {

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_cache_index(int page, int head, int d, int offset) {
    return paged_kv_element_offset<G::kHeadDim, G::KVHeads>(page, head, offset, d);
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
template <typename Geometry, class Schedule, typename Element>
__device__ __forceinline__ void bf16_kv_stage_tile(Element* dst, const Element* cache, int kv_head,
                                                   int k0, int max_query_abs, int physical_page,
                                                   int tid) {
    constexpr int D         = Geometry::kHeadDim;
    constexpr int Bc        = Schedule::kKeyRows;
    constexpr int Threads   = Schedule::kThreads;
    constexpr int VecPerRow = D / 8; // 8 two-byte elements per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const Element* cache_block =
        cache + paged_kv_element_offset<Geometry::kHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l = chunk / VecPerRow;       // / VecPerRow
            const int d     = (chunk % VecPerRow) * 8; // feature offset
            Element* p      = &dst[key_l * D + causal_swizzle(key_l, d)];
            cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l = chunk / VecPerRow;       // / VecPerRow
            const int d     = (chunk % VecPerRow) * 8; // feature offset
            Element* p      = &dst[key_l * D + causal_swizzle(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// Transposed V staging for the native-wmma PV route (see tiled_mma.cuh): dst is a
// [D][Bc] tile -- one row per head dim, the Bc keys contiguous -- so a w32 B-fragment
// (1 N-row x 16 contiguous contraction keys) is two LDS128 reads with no transpose.
// cp.async cannot write this layout (it moves 16 contiguous bytes and the transpose
// needs 2-byte granularity), so the transpose lives in the stores: each store merges a
// key pair (2*pair, 2*pair+1) into one 4B-aligned u32, which is also the cheapest
// store width available. Staging reads 16B per key (its own row in the cache) and
// writes 8 words per item, so the tile needs 2 int4 loads per 8 stores -- the real cost
// of the layout swap, paid on the store side.
//
// Swizzle inside a 32-key row: the 16B (8-key) granule g maps to g ^ (dim & 3), the
// causal_probability_swizzle<32> shape. That keeps every u32 store 4B-aligned (pairs
// never straddle a granule) and makes the WMMA reads conflict-free: the 8-lane quarter
// wavefront of an int4 read covers 4 distinct dim rows whose swizzled granules are
// distinct, so the 16B accesses land on 16 distinct banks.
//
// Layout contract: element (dim, key) lands at
//   dst[dim * Bc + (((key >> 3) ^ (dim & 3)) << 3) + (key & 7)]
template <int Bc>
__device__ __forceinline__ int bf16_vt_index(int dim, int key) {
    static_assert(Bc == 32, "transposed V staging assumes a 32-key row (four 16B granules)");
    return dim * Bc + ((((key >> 3) ^ (dim & 3)) << 3) + (key & 7));
}

template <typename Geometry, class Schedule, typename Element>
__device__ __forceinline__ void bf16_kv_stage_vt(Element* dst, const Element* cache, int kv_head,
                                                 int k0, int max_query_abs, int physical_page,
                                                 int tid) {
    constexpr int D       = Geometry::kHeadDim;
    constexpr int Bc      = Schedule::kKeyRows;
    constexpr int Threads = Schedule::kThreads;
    constexpr int Pairs   = Bc / 2; // key pairs per row = u32 stores per dim column
    constexpr int Chunks  = D / 8;  // 8 dims per 16B load
    static_assert(Bc == 32, "transposed V staging is written for Bc == 32");
    const bool full_tile = (k0 + Bc - 1) <= max_query_abs;
    const Element* cache_block =
        cache + paged_kv_element_offset<Geometry::kHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    // item = (key pair, 8-dim chunk): 16 x 32 = 512 items over 128 threads, 4 rounds.
    // Each item transposes a 2-key x 8-dim block: read the two keys as their own 16B
    // loads (each key is contiguous in the cache), write the 8 merged words.
#pragma unroll
    for (int item = tid; item < Pairs * Chunks; item += Threads) {
        const int pair = item % Pairs;
        const int d0   = (item / Pairs) * 8;
        const int key0 = 2 * pair;
        const int key1 = key0 + 1;
        // Zero the tail per key, not per pair: key0 is live whenever it is <= max_query_abs,
        // and zeroing a live key would feed a real query row a probability of 0.
        const bool keep0 = full_tile || (k0 + key0) <= max_query_abs;
        const bool keep1 = full_tile || (k0 + key1) <= max_query_abs;
        // Plain load_vec, not load_ldg: on HIP the existing tile staging's cp_async<16,cg>
        // already lowers to this exact global load, so the two routes stay comparable in the
        // machine-code census.
        const uint4 v0 = keep0 ? load_vec<uint4>(cache_block + key0 * D + d0) : make_uint4(0, 0, 0, 0);
        const uint4 v1 = keep1 ? load_vec<uint4>(cache_block + key1 * D + d0) : make_uint4(0, 0, 0, 0);
        const unsigned* w0 = reinterpret_cast<const unsigned*>(&v0);
        const unsigned* w1 = reinterpret_cast<const unsigned*>(&v1);
        // Word e of the destination row (dim = d0+e) = (V[key0][d0+e], V[key1][d0+e]).
        // A source u32 word holds dims (d0+2j, d0+2j+1), so each output word picks the
        // low or high half of source word e>>1 from each key: one byte-permute per word.
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const unsigned sel = (e & 1) ? 0x7632u : 0x5410u;
            // Deduce both template arguments (store_vec is <class T, class V>): T is the tile's
            // element pointer, V is the u32 being planted at that address.
            store_vec(&dst[bf16_vt_index<Bc>(d0 + e, key0)],
                      __byte_perm(w0[e >> 1], w1[e >> 1], sel));
        }
    }
}

} // namespace ninfer::ops::detail
