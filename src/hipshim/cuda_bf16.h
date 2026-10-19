#pragma once

// Shadow of <cuda_bf16.h>. The HIP header carries the same surface (conversions, arithmetic,
// round-to-nearest constructors); only the vendor spelling differs, and CUDA's _rn aliases of
// the two entry-point conversions are absent here.

#include <hip/hip_bf16.h>

// CUDA's storage types under the spelling the sources use.
using nv_bfloat16  = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using __nv_bfloat16  = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// HIP spells the round-to-nearest helpers without the suffix: every conversion is RN.
__attribute__((always_inline)) inline __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

__attribute__((always_inline)) inline __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __hip_bfloat162{__float2bfloat16(a), __float2bfloat16(b)};
}
