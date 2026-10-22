#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

// Upper bound of the quantised-x plane the byte-dot decode route bumps from a call's
// arena scope (including the alignment slack a bump can sit behind); 0 in builds with
// no byte-dot route. Planning adds it to the arena capacity of the ops whose decode
// launches allocate from it, so those launches can never fail to bump.
[[nodiscard]] std::size_t q8_gdn_input_dp4a_scratch_bytes() noexcept;

// Whether the byte-dot decode route is live in this build: a HIP build with
// NINFER_Q8_DP4A not "off" (the same predicate the decode launches gate on). The
// capacity queries add the scratch above only while this is true, so a declined
// route is never reserved for and an enabled one never undersized.
[[nodiscard]] bool q8_gdn_input_dp4a_route_live() noexcept;

// The decode launches take a nullable workspace: the byte-dot route (HIP, enabled) bumps
// its quantised-x plane from a scope of this arena, so a null arena keeps them on the
// SIMT schedules. The conv form additionally feeds the fused conv/snapshot epilogue.
// `allow_integer_route` carries the call's policy: the byte-dot route quantises the
// activation to s8, so it runs only for a policy that admits integer-A8 (linear.h's
// allows_a8_int); without it the launches stay on the A16 SIMT schedules.
void q8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 bool allow_integer_route, WorkspaceArena* workspace,
                                 cudaStream_t stream);
void q8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, bool allow_integer_route,
    WorkspaceArena* workspace, cudaStream_t stream);
void q8_gdn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void q8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void q8_gdn_input_splitk_conv_record_launch(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, cudaStream_t stream);
void q8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
