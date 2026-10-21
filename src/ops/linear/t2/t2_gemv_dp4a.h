#pragma once

// Byte-dot (dp4a) decode GEMV for T2G128 RowSplit weights: one quantize wavefront turns
// x into s8 with one binary16 scale per 128-group, then each row's fp32 partial is a chain
// of hardware byte dots (t2_gemv_dp4a.cuh). Exists only on the HIP backend, where the
// wavefront dot4 lowers to v_dot4_i32_iu8; NINFER_T2_DP4A=off falls back to the bit-trick
// word partial.

#ifdef NINFER_BACKEND_HIP

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// True when the dp4a route registers this (n, k) for the T = 1 decode shape.
[[nodiscard]] bool t2_gemv_dp4a_supported(std::int32_t n, std::int32_t k);

// out [n,1] = w [n,k] x x [k,1], single wavefront quantize + dp4a GEMV. Takes the shapes
// t2_gemv_dp4a_supported admits; throws otherwise.
void t2_gemv_dp4a_run(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail

#endif // NINFER_BACKEND_HIP
