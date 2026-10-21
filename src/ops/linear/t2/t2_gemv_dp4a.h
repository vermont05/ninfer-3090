#pragma once

// Byte-dot (dp4a) decode GEMV for T2G128 RowSplit weights: one quantize wavefront turns
// x into s8 with one binary16 scale per 128-group, then each row's fp32 partial is a chain
// of hardware byte dots (t2_gemv_dp4a.cuh). Exists only on the HIP backend, where the
// wavefront dot4 lowers to v_dot4_i32_iu8; NINFER_T2_DP4A=off falls back to the bit-trick
// word partial.

#ifdef NINFER_BACKEND_HIP

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The NINFER_T2_DP4A environment gate ("off" disables the route). One site so the
// dispatch hook and the wrapper that folds a pair into one quantize agree on whether
// the route is on.
[[nodiscard]] bool t2_gemv_dp4a_enabled();

// True when the dp4a route registers this (n, k) for the T = 1 decode shape.
[[nodiscard]] bool t2_gemv_dp4a_supported(std::int32_t n, std::int32_t k);

// Bytes of the quantised-activation scratch one k-wide call takes from the workspace:
// one s8 code per element plus one binary16 scale per 128-group (the quantize kernel
// writes the two planes from one wavefront, so they travel together).
[[nodiscard]] std::size_t t2_gemv_dp4a_scratch_bytes(std::int32_t k);

// One wavefront quantization of x into the s8 code plane + binary16 scale plane of a
// t2_gemv_dp4a_run call, taken standalone so several GEMVs can share it: the pair wrapper
// prepares once per projection instead of once per leg (three legs, one quantize). The
// plane is bump-allocated from the caller's open scope and lives for it; passing it to
// run_prepared is exactly what run() does internally, split across two calls.
struct T2Dp4aActivation {
    std::uint8_t* xq = nullptr;
    std::uint16_t* xs = nullptr;
    std::int32_t k   = 0;
};

[[nodiscard]] T2Dp4aActivation t2_gemv_dp4a_prepare(const Tensor& x, WorkspaceArena& workspace,
                                                     cudaStream_t stream);

// The t2_gemv_dp4a_run GEMV for an already-quantized vector: w must be a registered shape
// with the activation's k. The destination is a stride-1 run of at least w.n elements;
// pass a slice at the wanted offset (the row-split layout's row-major planes make a row
// range of a wider destination a valid target).
void t2_gemv_dp4a_run_prepared(const T2Dp4aActivation& activation, const Weight& w, Tensor& out,
                                cudaStream_t stream);

// out [n,1] = w [n,k] x x [k,1], single wavefront quantize + dp4a GEMV. Takes the shapes
// t2_gemv_dp4a_supported admits; throws otherwise. The quantised x plane is allocated
// from workspace (in a nested scope), so concurrent streams must get distinct arenas.
void t2_gemv_dp4a_run(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail

#endif // NINFER_BACKEND_HIP
