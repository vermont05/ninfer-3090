#pragma once

// Shadow of <cub/warp/warp_merge_sort.cuh>: the same class in hipcub's rocprim backend, with
// the same leading <KeyT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS> template parameters the
// tree instantiates it with.

#include <hipcub/warp/warp_merge_sort.hpp>

namespace cub {
using hipcub::WarpMergeSort;
} // namespace cub
