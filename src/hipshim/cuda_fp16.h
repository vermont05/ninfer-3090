#pragma once

// Shadow of <cuda_fp16.h>: the HIP header carries the whole CUDA surface (__half, __half2,
// the conversions and arithmetic the sources use) under the same names.

#include <hip/hip_fp16.h>

// HIP spells __hmax for __half and __hmax2 for __hip_bfloat162, but not __hmax2 for
// __half2 — CUDA's header has it, and the int8 FA2 prompt kernel folds its V absmax
// with it. The two-lane component form matches the bf16 one above. Device pass only:
// a plain C++ TU including this header sees only the standalone fp16 stubs, where the
// bf16 header's __hmax is the sole declaration of the name and this body cannot parse.
#if defined(__HIP__)
__attribute__((always_inline)) inline __device__ __half2 __hmax2(__half2 a, __half2 b) {
    return __half2{__hmax(a.x, b.x), __hmax(a.y, b.y)};
}
#endif
