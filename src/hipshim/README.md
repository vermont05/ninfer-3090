# hipshim — CUDA API names for the HIP backend

`NINFER_BACKEND=hip` compiles the same sources with hipcc. This directory holds the CUDA-named
headers those sources include (`cuda_runtime.h`, `cuda_bf16.h`, `cublas_v2.h`, `cub/...`), each of
which forwards to its ROCm counterpart and carries whatever name translation the CUDA API needs
to land on it. `ninfer_internal_includes()` puts it first on the include path of every target
when the HIP backend is configured; the CUDA build never looks at it.

The point is that no source file, and no kernel body, carries backend conditionals for the API
layer: a kernel that launches with `cudaLaunchKernelEx` compiles on both backends unchanged.
Only real hardware divergence belongs in sources (PTX-only instructions, Blackwell tensor-core
routes, the sm_120 warp-specialized kernels), and that stays per-file and explicit.

Two rules keep the table in `detail/api.h` honest:

* a CUDA name whose HIP counterpart is a **type** becomes an alias (`using`), so the CUDA name
  denotes exactly the HIP type;
* a CUDA name whose HIP counterpart is a **function or an enumerator** becomes an object-like
  macro, because `using` cannot reach a function template or an enum constant.

Anything whose signature or semantics differs goes into a real wrapper function in that header
(`cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaFuncSetAttribute`), never into a rename — a
rename would silently pass the CUDA call's arguments to a HIP function that expects different
ones.

`#include <hip/hip_runtime.h>` is what every shadow header bottoms out at; the `__nv_bfloat16`,
`__nv_fp8_*` and `__nv_fp4_*` spellings the sources use are aliased to the `__hip_*` vendor types
in the same way.
