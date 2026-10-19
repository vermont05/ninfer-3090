#pragma once

// Shadow of <math_constants.h>. The tree includes it for CUDART_INF_F and CUDART_NAN_F,
// which detail/api.h defines; CUDA's own header carries the whole constant table, and
// nothing else in the tree reaches for it.

#include "detail/api.h"
