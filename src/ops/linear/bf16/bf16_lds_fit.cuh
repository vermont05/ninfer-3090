#pragma once
#include "ops/linear/bf16/bf16_schedule.cuh"
#include <type_traits>

#if defined(__HIPCC__)
namespace ninfer::ops::detail {

// RDNA3 gives a thread group at most 64 KiB of LDS (gfx1102: sharedPerBlock ==
// sharedPerBlockOptin == 65536). The sm_8x shape tables were tuned against the RTX 3090's
// ~99 KiB shared carve-out and instantiate MMA tiles above that ceiling; such a launch
// cannot be configured on this device family. Rebuild an over-ceiling tile smaller before
// the launch configures it: first drop staging stages, and only then halve BlockK, so the
// warp geometry and warp count of a tier are preserved whenever stage depth alone suffices.
inline constexpr int kBf16LdsCeilingBytes = 64 * 1024;

template <class Schedule>
struct Bf16LdsFit {
    using type = Schedule;
};

template <class Schedule, int K, int Capacity, bool ExactTokens>
struct Bf16LdsFit<Bf16ScheduleInstance<Schedule, K, Capacity, ExactTokens>> {
    using type =
        Bf16ScheduleInstance<typename Bf16LdsFit<Schedule>::type, K, Capacity, ExactTokens>;
};

template <class Schedule>
struct Bf16LdsFit<Bf16RowTailSchedule<Schedule>> {
    using type = Bf16RowTailSchedule<typename Bf16LdsFit<Schedule>::type>;
};

template <class Schedule>
struct Bf16LdsFit<Bf16KTailSchedule<Schedule>> {
    using type = Bf16KTailSchedule<typename Bf16LdsFit<Schedule>::type>;
};

namespace lds_fit_detail {

// Widest BlockK at or below BlockK whose two-stage staging footprint fits the ceiling.
// Table BlockK values are multiples of 64, so each descent step is one table step.
template <int RowCount, int BlockK>
constexpr int Bf16MmaStageFit(int) {
    if constexpr (RowCount * BlockK * 4 <= kBf16LdsCeilingBytes) {
        return BlockK;
    } else {
        static_assert(BlockK > 64, "MMA tile staging exceeds the RDNA3 LDS ceiling at BlockK 64");
        return Bf16MmaStageFit<RowCount, BlockK / 2>(0);
    }
}

} // namespace lds_fit_detail

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache,
          Bf16MmaFragmentPipeline FragmentPipeline, Bf16MmaRaster Raster, Bf16MmaSwizzle Swizzle,
          int RasterGroupRows>
struct Bf16LdsFit<Bf16A16MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens,
                                     Stages, MinBlocksPerSm, WeightCache, ActivationCache,
                                     FragmentPipeline, Raster, Swizzle, RasterGroupRows>> {
private:
    static constexpr int kRowCount = BlockRows + BlockTokens;
    static constexpr int kFitBlockK =
        kRowCount * BlockK * 2 * Stages <= kBf16LdsCeilingBytes
            ? BlockK
            : lds_fit_detail::Bf16MmaStageFit<kRowCount, BlockK>(0);
    static constexpr int kFitStages =
        kRowCount * kFitBlockK * 2 * Stages <= kBf16LdsCeilingBytes
            ? Stages
            : kBf16LdsCeilingBytes / (kRowCount * kFitBlockK * 2);

public:
    using type = Bf16A16MmaSchedule<BlockRows, BlockTokens, kFitBlockK, WarpRows, WarpTokens,
                                    kFitStages, MinBlocksPerSm, WeightCache, ActivationCache,
                                    FragmentPipeline, Raster, Swizzle, RasterGroupRows>;
};

template <int BlockRows, int BlockTokens, int KWarps, int WarpK, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm>
struct Bf16LdsFit<
    Bf16A16SlicedKMmaSchedule<BlockRows, BlockTokens, KWarps, WarpK, Stages, WeightCache,
                              ActivationCache, MinBlocksPerSm>> {
private:
    static constexpr int kStageBytes = (BlockRows + BlockTokens) * KWarps * WarpK * 2;
    static constexpr int kPartialBytes = KWarps * BlockRows * BlockTokens * 4;
    static_assert(kPartialBytes <= kBf16LdsCeilingBytes,
                  "sliced-K partials exceed the RDNA3 LDS ceiling");
    static constexpr int kFitStages =
        Stages * kStageBytes <= kBf16LdsCeilingBytes
            ? Stages
            : (kBf16LdsCeilingBytes / kStageBytes > 0 ? kBf16LdsCeilingBytes / kStageBytes : 1);

public:
    static_assert(Stages * kStageBytes <= kBf16LdsCeilingBytes ||
                  kFitStages * kStageBytes <= kBf16LdsCeilingBytes);
    using type = Bf16A16SlicedKMmaSchedule<BlockRows, BlockTokens, KWarps, WarpK, kFitStages,
                                            WeightCache, ActivationCache, MinBlocksPerSm>;
};

} // namespace ninfer::ops::detail
#endif
