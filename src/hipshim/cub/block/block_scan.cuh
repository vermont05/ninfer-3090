#pragma once

// Shadow of <cub/block/block_scan.cuh>: the same class in hipcub's rocprim backend, under a
// namespace cub alias. The tree uses `cub::BlockScan<int, kSelectThreads>` and calls
// ExclusiveSum; hipcub's carries both in the same shapes.

#include <hipcub/block/block_scan.hpp>

namespace cub {
using hipcub::BlockScan;
} // namespace cub
