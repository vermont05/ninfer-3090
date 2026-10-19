#pragma once

// Shadow of <cooperative_groups.h>. HIP ships its own cooperative-groups header with the
// same namespace and the grid_group / this_grid() entry point the tree calls; it is active
// only in the HIP compile passes, which is where anything including this is compiled.

#include "detail/api.h"

#include <hip/hip_cooperative_groups.h>
