#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"
#include <type_traits>

#if defined(__HIPCC__)
namespace ninfer::ops::detail {

// RDNA3 gives a thread group at most 64 KiB of LDS (gfx1102: sharedPerBlock ==
// sharedPerBlockOptin == 65536), while the sm_8x/sm_120 tables were tuned against a 99 KiB
// carve-out. An over-ceiling tiled tile cannot be configured at all on this device family,
// so rebuild it smaller before the launch configures it: first halve KeyRows (which keeps the
// query-row geometry, and therefore the warp mapping, of the tier intact), and only then
// halve QueryRows. Table KeyRows values are 64, so every descent step is one table step.
inline constexpr int kBf16CausalLdsCeilingBytes = 64 * 1024;

namespace causal_fit_detail {

template <class Geometry, int Br, int Bc, int MinBlocks>
constexpr auto Bf16KvTiledDescend(int) {
    if constexpr (bf16_kv_tiled_shared_bytes<Geometry, Bf16KvTiledMmaSchedule<Br, Bc, MinBlocks>> <=
                  kBf16CausalLdsCeilingBytes) {
        return Bf16KvTiledMmaSchedule<Br, Bc, MinBlocks>{};
    } else if constexpr (Bc > 16) {
        return Bf16KvTiledDescend<Geometry, Br, Bc / 2, MinBlocks>(0);
    } else {
        static_assert(Br > 16, "causal MMA tile exceeds the RDNA3 LDS ceiling");
        return Bf16KvTiledDescend<Geometry, Br / 2, 64, MinBlocks>(0);
    }
}

} // namespace causal_fit_detail

template <class Geometry, class Schedule>
struct Bf16KvTiledLdsFit {
    using type = Schedule;
};

template <class Geometry, int Br, int Bc, int MinBlocks>
    requires (bf16_kv_tiled_shared_bytes<Geometry, Bf16KvTiledMmaSchedule<Br, Bc, MinBlocks>> >
              kBf16CausalLdsCeilingBytes)
struct Bf16KvTiledLdsFit<Geometry, Bf16KvTiledMmaSchedule<Br, Bc, MinBlocks>> {
    using type =
        decltype(causal_fit_detail::Bf16KvTiledDescend<Geometry, Br, Bc, MinBlocks>(0));
};

} // namespace ninfer::ops::detail
#endif
