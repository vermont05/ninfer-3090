#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Measured on gfx1102 with bench/ops/linear_schedule_bench.cu (`q4:5120x17408`), cold, median
// of 9. The table that arrived here was n5120_k6144.cu's ladder moved to a deeper K without a
// measurement, and on this card that ladder is wrong in both of its halves:
//
//   T=1    gemv_r4_w1  221.8 vs ksplit8 (the old route) 1529 -- 6.9x
//   T=8    simt_r8_c8   740.6 vs ksplit8                1455 -- the ksplit rungs cost ~400 us of
//                                                       fixed overhead here, every capacity
//   T=128  simt_r8_c8 10757 vs mma_r64_c64t (best mma)  38741 -- the MMA tiers reach 7-72 ms at
//          every column count on gfx1102; the sm_86 ladder that chose them never applied.
//
// The eight-row SIMT tile wins every interval from 6 through 512; the four-row one leads at
// 2 and 4 by 4-6% and is not worth a second compiled instance next to the GEMV at 1.
Q4Launch select_q4_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    return launch_q4_a16_simt_r8_t8_w1_g16_s2;
}

} // namespace ninfer::ops::detail
