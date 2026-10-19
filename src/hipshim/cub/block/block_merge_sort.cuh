#pragma once

// Shadow of <cub/block/block_merge_sort.cuh>: the same class in hipcub's rocprim backend.
// The tree uses `cub::BlockMergeSort<Key, Threads, ItemsPerThread>` with a TempStorage and
// Sort(keys, comp); hipcub's has all three in the same shapes.

#include <hipcub/block/block_merge_sort.hpp>

namespace cub {
using hipcub::BlockMergeSort;
} // namespace cub
