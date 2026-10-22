#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8GdnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR64C128,
};

enum class Q8GdnInputConvScheduleId {
    DecodeFused,
    SplitKMmaFused,
    Materialized,
};

struct Q8GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8GdnInputPlan {
    Q8GdnInputScheduleId schedule;
};

struct Q8GdnInputConvPlan {
    Q8GdnInputConvScheduleId schedule;
};

const char* q8_gdn_input_schedule_name(Q8GdnInputScheduleId schedule) noexcept;
const char* q8_gdn_input_conv_schedule_name(Q8GdnInputConvScheduleId schedule) noexcept;
bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept;
Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem);
Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size);

// Routes one Q8_G32 GDN input projection. `allow_integer_route` gates the byte-dot decode
// route (it quantises the activation to s8); `workspace` (nullable) is handed to the decode
// launches; see q8_gdn_input_kernels.h.
void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            bool allow_integer_route, WorkspaceArena* workspace,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
