#pragma once

// Shadow of <cuda_fp4.h>: the vendor FP4 types the nvfp4 codecs declare, on top of
// <hip/hip_fp4.h> (same struct shape and float2 constructor, __hip_* spelling).

#include <hip/hip_fp4.h>

using __nv_fp4_storage_t = __hip_fp4_storage_t;
using __nv_fp4x2_e2m1    = __hip_fp4x2_e2m1;
