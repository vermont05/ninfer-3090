#pragma once

// Shadow of <cuda_fp8.h>: the vendor FP8 types and the three conversion entry points the
// codecs call, on top of <hip/hip_fp8.h> (same structs, same enum values, __hip_* spelling).

#include <hip/hip_fp8.h>

// The spellings the codecs use.
using __nv_fp8_storage_t  = __hip_fp8_storage_t;
using __nv_fp8_e4m3       = __hip_fp8_e4m3;
using __nv_fp8x2_e4m3     = __hip_fp8x2_e4m3;

// The enumerator values the codecs pass to the conversions.
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3

// __nv_cvt_float_to_fp8(f, sat, interp) and the float2 variant.
#define __nv_cvt_float_to_fp8(f, sat, interp) __hip_cvt_float_to_fp8(f, sat, interp)
#define __nv_cvt_float2_to_fp8x2(f, sat, interp) __hip_cvt_float2_to_fp8x2(f, sat, interp)
