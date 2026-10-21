#pragma once

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

// `workspace` backs the byte-dot decode route's quantised-x plane; a null arena keeps
// the T = 1 decode on the word schedules (the dp4a route is not admitted without one).
void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
