#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

#if defined(__HIPCC__)
// --- SIMT reference path ------------------------------------------------------------------
// GCN has neither ldmatrix nor an mma.sync tensor core. The helpers and definitions below
// evaluate the fragment layouts of PTX ISA section 9.7.16.5 (groupID = laneid >> 2 and
// threadID_in_group = laneid % 4, formulas restated at each definition) and the product-sum
// each instruction computes. A gathered fragment decodes to the same matrix a CUDA warp
// loads and the reference sums the same products; only the partial-sum order differs from
// the tensor-core result. gfx1102 selects wave32, so lane == threadIdx.x % 32 as on CUDA,
// and every fetch below is one wave32 shuffle.
namespace mma_detail {

__device__ __forceinline__ unsigned lane() {
    return static_cast<unsigned>(
        __builtin_amdgcn_mbcnt_lo(~0u, __builtin_amdgcn_mbcnt_hi(~0u, 0u)));
}

// Broadcast another lane's value to the wave; the references read the other lanes'
// fragment words with one shuffle per word, addressed by lane.
__device__ __forceinline__ unsigned bcast_u32(unsigned v, unsigned src) {
    return __shfl_sync(0xFFFFFFFFu, v, static_cast<int>(src));
}

__device__ __forceinline__ std::uintptr_t bcast_ptr(std::uintptr_t addr, unsigned src) {
    return static_cast<std::uintptr_t>(
        __shfl_sync(0xFFFFFFFFu, static_cast<unsigned long long>(addr), static_cast<int>(src)));
}

// The provider group's row/column as eight words: first register's four words in lane
// order, then the second register's — in the m16n8k16/m16n8k32 fragments that order is
// exactly the element sequence of one row (A) or one column (B).
__device__ __forceinline__ void gather8(unsigned rA, unsigned rB, unsigned base, unsigned* out) {
    out[0] = bcast_u32(rA, base);
    out[1] = bcast_u32(rA, base + 1u);
    out[2] = bcast_u32(rA, base + 2u);
    out[3] = bcast_u32(rA, base + 3u);
    out[4] = bcast_u32(rB, base);
    out[5] = bcast_u32(rB, base + 1u);
    out[6] = bcast_u32(rB, base + 2u);
    out[7] = bcast_u32(rB, base + 3u);
}

// --- shared-memory matrix reads -----------------------------------------------------------
// ldmatrix.m8n8 takes 8 row addresses per matrix from 8 consecutive lanes (matrix 0 from
// lanes 0..7, each further matrix from the next 8), and gives every lane the pair
// (row lane >> 2, col (lane & 3) * 2) of that matrix.

__device__ __forceinline__ unsigned pair_at(std::uintptr_t addr, unsigned base, unsigned row,
                                             unsigned col) {
    return load_vec<unsigned>(
        reinterpret_cast<const std::uint16_t*>(bcast_ptr(addr, base + row)) + (col & ~1u));
}

// .trans: the same provider rows, but the fragment is of the transposed matrix — lane l's
// pair is (M^T)[g][2t], (M^T)[g][2t+1], i.e. provider rows 2t and 2t+1 read at column g.
__device__ __forceinline__ unsigned bcast_pair_t(std::uintptr_t addr, unsigned base,
                                                  unsigned row_pair) {
    const std::uintptr_t r0 = bcast_ptr(addr, base + row_pair);
    const std::uintptr_t r1 = bcast_ptr(addr, base + row_pair + 1u);
    const unsigned col = lane() >> 2;
    return static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(r0) + col) |
           (static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(r1) + col) << 16);
}

// --- element codecs ---------------------------------------------------------------------------
__device__ __forceinline__ void unpack_bf2w(unsigned w, float& lo, float& hi) {
    const __nv_bfloat162 bf = load_vec<__nv_bfloat162>(&w);
    const float2 f = __bfloat1622float2(bf);
    lo = f.x;
    hi = f.y;
}

__device__ __forceinline__ void unpack_h2w(unsigned w, float& lo, float& hi) {
    const __half2 h = load_vec<__half2>(&w);
    lo = __half2float(__low2half(h));
    hi = __half2float(__high2half(h));
}

// E4M3 per the F16X2/E4M3X2 conversion table of the PTX ISA: both NaN codes (0x7F, 0xFF)
// decode to the canonical NaN, everything else to its exact value (binary32 holds every
// e4m3 value exactly).
__device__ __forceinline__ float e4m3_to_f32(unsigned byte) {
    if (byte == 0xFFu || byte == 0x7Fu) {
        return __uint_as_float(0x7FC00000u);
    }
    const unsigned sign = (byte >> 7) & 1u;
    const unsigned exponent = (byte >> 3) & 0xFu;
    const unsigned mant = byte & 7u;
    if (exponent == 0u) {
        return (sign != 0u ? -1.f : 1.f) * (static_cast<float>(mant) * 0.001953125f);
    }
    const unsigned bits = (sign << 31u) | ((exponent + 120u) << 23u) | (mant << 20u);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float e2m1_nibble(unsigned nib) {
    const float mag = nib < 4u ? static_cast<float>(nib) * 0.5F
                      : nib == 4u ? 1.5F
                      : nib == 5u ? 2.5F
                      : nib == 6u ? 3.5F
                                   : 5.5F;
    return (nib & 8u) != 0u ? -mag : mag;
}

__device__ __forceinline__ float tf32_of(unsigned bits) {
    // The 10-mantissa-bit value a tf32 operand carries; callers pass cvt.rna'd words.
    return __uint_as_float(bits & 0xFFFFE000u);
}

// --- dots over gathered words -----------------------------------------------------------------
// Both vectors are one operand row and one operand column of one instruction, word i of
// each covering the same element positions.

__device__ __forceinline__ float dot_bf8(const unsigned* a, const unsigned* b) {
    float s = 0.f;
    float lo, hi, blo, bhi;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unpack_bf2w(b[i], blo, bhi);
        unpack_bf2w(a[i], lo, hi);
        s += blo * lo;
        s += bhi * hi;
    }
    return s;
}

__device__ __forceinline__ float dot_h8(const unsigned* a, const unsigned* b) {
    float s = 0.f;
    float lo, hi, blo, bhi;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unpack_h2w(b[i], blo, bhi);
        unpack_h2w(a[i], lo, hi);
        s += blo * lo;
        s += bhi * hi;
    }
    return s;
}

__device__ __forceinline__ float dot_b4e4m3w8(const unsigned* a, const unsigned* b) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned x = a[i], y = b[i];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            s += e4m3_to_f32((x >> (e * 8)) & 0xFFu) * e4m3_to_f32((y >> (e * 8)) & 0xFFu);
        }
    }
    return s;
}

// RDNA2+ ships V_DOT_INSTS-style byte dot products; gfx1102 (RDNA3) decodes
// __builtin_amdgcn_sudot4 into V_SUDOT4, and one call replaces the eight sign-extended
// byte products of one word pair. The hardware sum order matches the loop's word order;
// like every reference here the partial order only differs from the tensor-core order.
__device__ __forceinline__ int dot_s8w8(const unsigned* a, const unsigned* b) {
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s = __builtin_amdgcn_sudot4(true, a[i], true, b[i], s, false);
    }
    return s;
}

__device__ __forceinline__ int dot_s8w4(const unsigned* a, const unsigned* b) {
    int s = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        s = __builtin_amdgcn_sudot4(true, a[i], true, b[i], s, false);
    }
    return s;
}

__device__ __forceinline__ float dot_tf1w4(const unsigned* a, const unsigned* b) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        s += tf32_of(a[i]) * tf32_of(b[i]);
    }
    return s;
}

__device__ __forceinline__ float dot_tf1w8(const unsigned* a, const unsigned* b) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s += tf32_of(a[i]) * tf32_of(b[i]);
    }
    return s;
}

// The e2m1 fragment words hold 8 nibbles; a 16-element chunk is two words.
__device__ __forceinline__ float dot_b4nib2(const unsigned* a, const unsigned* b) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned x = a[i], y = b[i];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            s += e2m1_nibble((x >> (e * 4)) & 0xFu) * e2m1_nibble((y >> (e * 4)) & 0xFu);
        }
    }
    return s;
}

} // namespace mma_detail
// --- ldmatrix and mma references ------------------------------------------------------------
// The definitions below re-implement, for one wave32 of lanes, the instructions of PTX ISA
// 9.7.16.5 that the CUDA branch spells in inline asm. Fragment positions (groupID g =
// laneid >> 2, threadID_in_group t = laneid % 4): ldmatrix.m8n8 gives each lane the pair
// (row lane >> 2, col (lane & 3) * 2) of each 8x8 tile, the tile addressed by the row
// pointer of lane 8i of tile i. The mma fragments follow the same rule: operand words are
// read at their providers' lanes, then the products summed. Accumulators sit at
// c0,c1 = (g, 2t),(g, 2t+1) and c2,c3 = (g+8, 2t),(g+8, 2t+1); the fp16-accumulator form packs
// (g, 2t..2t+1) into c0 and (g+8, 2t..2t+1) into c1.

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, std::uintptr_t addr) {
    const unsigned l = mma_detail::lane();
    const std::uintptr_t a = addr;
    r0 = mma_detail::pair_at(a, 0u, l >> 2, (l & 3u) * 2u);
    r1 = mma_detail::pair_at(a, 8u, l >> 2, (l & 3u) * 2u);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                            unsigned& r3, std::uintptr_t addr) {
    const unsigned l = mma_detail::lane();
    const std::uintptr_t a = addr;
    r0 = mma_detail::pair_at(a, 0u, l >> 2, (l & 3u) * 2u);
    r1 = mma_detail::pair_at(a, 8u, l >> 2, (l & 3u) * 2u);
    r2 = mma_detail::pair_at(a, 16u, l >> 2, (l & 3u) * 2u);
    r3 = mma_detail::pair_at(a, 24u, l >> 2, (l & 3u) * 2u);
}

// .trans inverts the tile: lane's pair of the transpose is provider rows 2 * (lane & 3)
// and +1, read at column lane >> 2.
__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, std::uintptr_t addr) {
    const unsigned l = mma_detail::lane();
    const std::uintptr_t a = addr;
    r0 = mma_detail::bcast_pair_t(a, 0u, (l & 3u) * 2u);
    r1 = mma_detail::bcast_pair_t(a, 8u, (l & 3u) * 2u);
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, std::uintptr_t addr) {
    const unsigned l = mma_detail::lane();
    const std::uintptr_t a = addr;
    r0 = mma_detail::bcast_pair_t(a, 0u, (l & 3u) * 2u);
    r1 = mma_detail::bcast_pair_t(a, 8u, (l & 3u) * 2u);
    r2 = mma_detail::bcast_pair_t(a, 16u, (l & 3u) * 2u);
    r3 = mma_detail::bcast_pair_t(a, 24u, (l & 3u) * 2u);
}

// m16n8k16 .bf16: A rows g and g+8 as two half-rows from the group's registers a0/a2 and
// a1/a3; B columns 2t and 2t+1 from b0/b1 of the column's group.
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    c0 += dot_bf8(ar, bc0);
    c1 += dot_bf8(ar, bc1);
    c2 += dot_bf8(ar8, bc0);
    c3 += dot_bf8(ar8, bc1);
}

// m16n8k16 .f16: the same shape with half inputs.
__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    c0 += dot_h8(ar, bc0);
    c1 += dot_h8(ar, bc1);
    c2 += dot_h8(ar8, bc0);
    c3 += dot_h8(ar8, bc1);
}

// m16n8k16 .f16 with FP16 accumulator. The CUDA instruction rounds every product and
// partial sum to FP16; this reference accumulates in FP32 and rounds once at the end
// with the same RNE conversion, so results can differ in low bits.
__device__ __forceinline__ void mma_f16_acc16(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    float lo0, hi0, lo8, hi8;
    unpack_h2w(c0, lo0, hi0);
    unpack_h2w(c1, lo8, hi8);
    const __half2 p0 = __floats2half2_rn(lo0 + dot_h8(ar, bc0), hi0 + dot_h8(ar, bc1));
    const __half2 p1 = __floats2half2_rn(lo8 + dot_h8(ar8, bc0), hi8 + dot_h8(ar8, bc1));
    c0 = load_vec<unsigned>(&p0);
    c1 = load_vec<unsigned>(&p1);
}

// m16n8k32 .s8: rows and columns as eight 4-byte words (32 values each).
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    c0 += dot_s8w8(ar, bc0);
    c1 += dot_s8w8(ar, bc1);
    c2 += dot_s8w8(ar8, bc0);
    c3 += dot_s8w8(ar8, bc1);
}

// m16n8k16 .s8: A is {a0,a1} = rows g and g+8 (4 bytes each of the group's lanes), B the
// single b0 word of the column's group; rows and columns are four 4-byte words.
__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[4], ar8[4], bc0[4], bc1[4];
    ar[0] = bcast_u32(a0, gbase);
    ar[1] = bcast_u32(a0, gbase + 1u);
    ar[2] = bcast_u32(a0, gbase + 2u);
    ar[3] = bcast_u32(a0, gbase + 3u);
    ar8[0] = bcast_u32(a1, gbase);
    ar8[1] = bcast_u32(a1, gbase + 1u);
    ar8[2] = bcast_u32(a1, gbase + 2u);
    ar8[3] = bcast_u32(a1, gbase + 3u);
    bc0[0] = bcast_u32(b0, cb);
    bc0[1] = bcast_u32(b0, cb + 1u);
    bc0[2] = bcast_u32(b0, cb + 2u);
    bc0[3] = bcast_u32(b0, cb + 3u);
    bc1[0] = bcast_u32(b0, cb + 4u);
    bc1[1] = bcast_u32(b0, cb + 5u);
    bc1[2] = bcast_u32(b0, cb + 6u);
    bc1[3] = bcast_u32(b0, cb + 7u);
    c0 += dot_s8w4(ar, bc0);
    c1 += dot_s8w4(ar, bc1);
    c2 += dot_s8w4(ar8, bc0);
    c3 += dot_s8w4(ar8, bc1);
}

// m16n8k32 .e4m3 (kind::mxf8f6f4, unit scale): 4 bytes per word, each byte one e4m3 value.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    c0 += dot_b4e4m3w8(ar, bc0);
    c1 += dot_b4e4m3w8(ar, bc1);
    c2 += dot_b4e4m3w8(ar8, bc0);
    c3 += dot_b4e4m3w8(ar8, bc1);
}

// m16n8k8 .tf32: rows and columns as the group's eight single-value words.
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb = (l & 3u) << 3;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb, bc0);
    gather8(b0, b1, cb + 4u, bc1);
    c0 += dot_tf1w8(ar, bc0);
    c1 += dot_tf1w8(ar, bc1);
    c2 += dot_tf1w8(ar8, bc0);
    c3 += dot_tf1w8(ar8, bc1);
}
// m16n8k64 .e2m1 with UE4M3 block scales (kind::mxf4nvf4, scale_vec::4X). A row is eight
// 8-nibble words; chunks of 16 elements are two words. The row's chunk scales are the
// four bytes of its provider lanes' sfa words (the thread-pair lanes the ISA's
// thread-id-a = 0 selection reads), the column's from its provider lane's sfb word.
// scale_vec::4X scales every 16-element chunk with its own byte.
__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    using namespace mma_detail;
    const unsigned l = lane();
    const unsigned gbase = (l >> 2) << 2;
    const unsigned cb0 = (l & 3u) << 3;
    const unsigned cb1 = cb0 + 4u;
    unsigned ar[8], ar8[8], bc0[8], bc1[8];
    gather8(a0, a2, gbase, ar);
    gather8(a1, a3, gbase, ar8);
    gather8(b0, b1, cb0, bc0);
    gather8(b0, b1, cb1, bc1);
    const unsigned sa0 = bcast_u32(sfa, gbase);
    const unsigned sa8 = bcast_u32(sfa, gbase + 1u);
    const unsigned sb0 = bcast_u32(sfb, cb0);
    const unsigned sb1 = bcast_u32(sfb, cb1);
    for (int ch = 0; ch < 4; ++ch) {
        const float r0 = dot_b4nib2(ar + 2 * ch, bc0 + 2 * ch);
        const float r8 = dot_b4nib2(ar8 + 2 * ch, bc0 + 2 * ch);
        const float r1 = dot_b4nib2(ar + 2 * ch, bc1 + 2 * ch);
        const float r9 = dot_b4nib2(ar8 + 2 * ch, bc1 + 2 * ch);
        const float wa = e4m3_to_f32((sa0 >> (ch * 8)) & 0xFFu);
        const float wa8 = e4m3_to_f32((sa8 >> (ch * 8)) & 0xFFu);
        const float wb0 = e4m3_to_f32((sb0 >> (ch * 8)) & 0xFFu);
        const float wb1 = e4m3_to_f32((sb1 >> (ch * 8)) & 0xFFu);
        c0 += wa * wb0 * r0;
        c1 += wa * wb1 * r1;
        c2 += wa8 * wb0 * r8;
        c3 += wa8 * wb1 * r9;
    }
}

#else


__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// FP16 x FP16 with an FP16 accumulator: full rate on GA10x GeForce parts, where the FP32-accumulate
// form above runs at half rate. c0 holds the accumulator pair (row g, cols 2t..2t+1) and c1 the pair
// (row g+8, cols 2t..2t+1) as packed __half2 bits; every product and partial sum rounds to FP16, so
// callers bound the partial range themselves.
__device__ __forceinline__ void mma_f16_acc16(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The k16 int8 shape: A is {a0,a1} (rows g and g+8, k = 4 (lane % 4) + 0..3), B is {b0}.
__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}


#endif

} // namespace ninfer::ops
