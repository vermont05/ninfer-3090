# NInfer-3090 → ROCm port analysis (gfx1102 eGPU)

Written 2026-10-18. Source analyzed: `~/git/ninfer-3090` @ clone of
`https://github.com/Don-Chad/ninfer-3090` v0.16.1-rtx3090.
Goal: upstreamable **dual-backend** (CUDA + HIP) port so NInfer runs on the attached
ONEXGPU eGPU, and to replace the PrismML/llama.cpp path currently used for Bonsai-2-27B.

---

## 1. Why

- PrismML (llama.cpp fork) is the only way Bonsai-2-27B runs today, and it is weak:
  on HIP the PTQ1_0 ternary variant falls back to a **scalar base-3 unpack**
  (12.5 tok/s vs 20.6 tok/s for PQ2_0's dp4a path — measured in
  `~/bin/bonsai-rocm.sh` comments). The kernel quality floor is llama.cpp's.
- NInfer is a from-scratch engine with its own quantization codecs — including the
  **Bonsai2-style ternary format** (`T2_G128_FP16`) — and an architecture where a
  second backend is a bounded (if large) project rather than a rewrite.

## 2. Target hardware and toolchain (measured on this box)

| | |
|---|---|
| eGPU | Navi 33 **RX 7600M XT**, PCI `1002:7480`, **gfx1102**, RDNA3, **8 GB VRAM**, Thunderbolt 3 (≈2.5 GB/s effective) |
| iGPU | Radeon 680M, **gfx1035** (RDNA2) — must be hidden (below) |
| ROCm | 7.1 (Ubuntu distro packages, `/usr/lib/rocm`, `hipcc`/`rocminfo` on PATH) |
| BLAS | rocBLAS 7.1 present; **hipBLASLt NOT installed** |

**Machine landmines, already solved once in `~/bin/bonsai-rocm.sh`:**

1. `ROCR_VISIBLE_DEVICES=1` is mandatory: a gfx1102-only HIP binary **segfaults in
   libamdhip64 module registration** when the gfx1035 iGPU is device 0.
2. Ubuntu's `librocblas5` ships **no gfx1102 Tensile kernels**; working kernels were
   extracted from AMD's ROCm 7.2 noble rocblas `.deb` into `~/opt/rocblas-gfx1102/`
   and selected via `ROCBLAS_TENSILE_LIBPATH`. Any rocBLAS use in the port needs
   the same treatment (or avoid rocBLAS — see §6 Phase 1, only one cuBLAS call exists).
3. Do not set `HSA_OVERRIDE_GFX_VERSION`.
4. gfx1102 has **no bf16/fp16 tensor cores** (no MFMA — that's CDNA-only). Wave32
   `wmma.f16` is the only matrix instruction; `mma.sync`, `ldmatrix`, `cp.async`
   have no equivalents. All NInfer PTX must be re-expressed, not translated.

## 3. VRAM budget for 27B on 8 GB — feasible, proven on this exact card

**Corrected after first-draft error:** 27B on 8 GB is not only possible, it is
running on this machine right now. Bonsai-2-27B (PrismML) is ternary {−1,0,+1}
weights with FP16 group scales at **1.76 effective bits/weight**
([prismml.com/news/bonsai-2-27b](https://prismml.com/news/bonsai-2-27b)).

Measured on this eGPU (`llama-server -ngl 999`, all layers on GPU):

| format | weights | note |
|---|---|---|
| PQ2_0  | **7.2 GB** | dp4a fast path on HIP → **20.6 tok/s** |
| PTQ1_0 | 5.6 GB | scalar base-3 fallback on HIP → 12.5 tok/s |

With q4_0 KV cache it sustains 40,960-token context (49,152 dies; treat 40k as the
ceiling with ~1 GB left for KV + state + workspace on the 8 GB card).

**NInfer's equivalent route — and this repo already ships it.** The fork added
Ternary Bonsai 2 27B natively (commit `63bab74f`, v0.16.x): model card
`model-cards/Ternary-Bonsai-2-27B-NInfer-3090/`, artifact
`bonsai2_27b.ninfer` = 8.87 GiB on disk, **6.70 GiB weights on device**
(7.28 GiB with MTP head) as `t2_g128_fp16` — ternary {−1,0,+1} + FP16 scale per
128 weights = 2.125 bits/weight, Hadamard-rotated, imported losslessly from
Prism ML's PQ2_0 GGUF. Launcher profile exists:
`scripts/download-model.sh bonsai2-27b`. RTX 3090 numbers from the card:
plain decode **88 tok/s**, MTP-3 199, DFlash2-7 **297**; perplexity 5.627.
Combined with NInfer's quantized KV storages (`KvCacheStorage`: `Int8Group64`,
`RotatedInt8KeyInt4ValueGroup64`, …) the 8 GB budget closes: 6.7–7.3 GB weights
+ ~0.8 GB quantized KV + state + workspace. Tight but real — exactly how the
llama-prism deployment (7.2 GB weights, q4_0 KV, 40k ctx) fits.
Architecture is `Qwen3_5ForCausalLM` (Qwen3.8 hybrid GDN geometry: most layers
use GDN linear attention, so the KV is cheap — verified via the model card).

**Decode is memory-bandwidth-bound at batch 1**, so a well-written ternary/int8 GEMV
in plain FFMA/`__dp4a` code should land in the same 15–25 tok/s band as llama-prism
despite the missing tensor cores. That is the port's viability argument.

## 4. What NInfer is (analysis of the 3 survey passes)

From-scratch C++20 inference engine, ~110k lines of CUDA (385 `.cu` + 238 `.cuh`,
~244 `__global__` kernels), Apache-2.0, hard-forked to sm_86.
**No CUTLASS. No flash-attention lib. No cuDNN/NCCL/NVRTC. Third-party = zero CUDA.**
Virtually no cuBLAS either (see §5). `AGENTS.md` lays out the ownership boundaries
(Core = physical primitives, Ops = closed math, Models = fixed math + binding,
Program = mutable state + CUDA Graphs, Runtime = contracts) and explicitly wants
"explicit implementations for supported architectures" — a compile-time dual backend
fits the project's own philosophy; plugin/abstraction layers are banned.

| Layer | LOC (measured) | CUDA coupling |
|---|---|---|
| `src/serve` (HTTP/SSE, OpenAI+Anthropic APIs) | ~14.8k, 0 `.cu` | ~5 lines `cudaGetDeviceProperties` in `generation_service.cpp` |
| `src/models` (qwen3_5, qwen4_exp, tokenizer, chat template, tool parsing, speculative decoding state machines) | ~52.9k, **0 `.cu`** | handle passthrough only |
| `src/runtime` (engine, scheduler) | ~12.9k, 0 `.cu` | none |
| `src/text` + `src/media` + `include` | ~11k | tokenizer is hand-rolled BPE — pure C++ |
| `src/artifact` (.ninfer v3 loader) | ~2.8k, 1 stray `.cu` (`permute_row_split.cu`, 95 lines) | upload/materialize on host |
| `src/core` (memory, devices, graphs) | 8.8k: **11 `.cu`**, rest host | arena/VMM/device/`decode_graph`/`stage_link`/`pdl` |
| `src/ops` (all kernels) | 75k: **278 `.cu` + 233 `.cuh`** | everything — hand-written PTX |
| `tests`/`bench`/`tools` | ~99k | ~100 `.cu` (bench + standalone probes) |

Backend-neutral math already in-tree that matters for this port: the AVX2 CPU
evaluation route for offloaded MoE experts (`src/ops/offloaded_sparse_moe/cpu/`,
bit-exact `canonical_ggml`) — zero porting needed if the qwen4_exp offload route is
ever used here.

### Kernel inventory (what actually has to be rewritten)

- **Inline PTX: 35 files / ~150 sites, but ~90% concentrates in 5 headers**:
  `src/ops/common/{mma.cuh, memory.cuh, rowsplit_a8_mma.cuh, mbarrier.cuh, warp.cuh}`.
  Instructions: `mma.sync m16n8k16 bf16/f16`, `m16n8k32 s8`, `m16n8k8 tf32`,
  `ldmatrix`, `cp.async` pipeline, `bar.sync`, `ex2.approx`, `tanh.approx`,
  `cvt.rn.bf16x2`. All Ampere-targeted — none exists on gfx1102.
- **Already excluded from sm8x builds (template for the HIP build!):**
  Blackwell block-scaled MMA (`nvfp4/*_a4.cu`, `fp8/*_a8.cu`) is filtered out in
  `src/ops/CMakeLists.txt:53-58` and replaced by `nvfp4_sm86_stubs.cpp` /
  `fp8_sm86_stubs.cpp`. PDL (`src/core/pdl.cuh`) is `NINFER_SM8X_COMPAT`-guarded.
  The 4 TMA files (`linear/{bf16,fp8,nvfp4}/*_tma*`, `softmax_attention/.../nvfp4/tiled_mma.cuh`
  + host-side `cuTensorMapEncodeTiled`) are sm90+ paths that throw at runtime on
  sm8x — exclude from HIP the same way.
- **cuBLAS: exactly ONE production file** —
  `src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.cu`, a single
  `cublasGemmEx(8I × 8I → 32I)` INT8 prefill GEMM. (cublasLt appears only in a
  bench probe.) Port = one `rocblas_gemm_ex` call, or bypass with a hand kernel
  and dodge the gfx1102-Tensile mess entirely.
- **CUDA Graphs**: whole-program decode capture, single site
  `src/core/decode_graph.cpp` (no node-param patching APIs used). `hipGraph`
  supports capture/instantiate/launch; risk is legality-rule differences.
- **Driver VMM** (`cuMemCreate/cuMemMap` family): 6 files
  (`src/core/{cuda_vmm.h, vmm_region.*, vmm_arena.*, evictable_weight_pool.cu,
  evictable_kv_pool.cu}`). `hipMemCreate` exists in ROCm 7.1; granularity differs
  (2 MB); eviction pools need re-measurement.
- **CUDA API surface is NOT wrapped**: 351 files include `cuda_runtime.h`;
  `cudaStream_t` threads through every entry point in `include/ninfer/ops/*.h`;
  38 host `.cpp` files call `cudaMalloc/Memcpy/Stream*` directly. Counts:
  `cudaStream_t` ~2,491, `cudaGetLastError` 383, `cudaMemcpy` ~215, events ~94,
  graphs ~38-48, `cudaHostAlloc` 31. No unified memory, no dynamic parallelism.
- **Other CUDA-only bits**: `cudaMemcpyBatchAsync` (wrapper
  `src/core/copy_batch.cpp` already has a one-by-one fallback), NVTX
  (`src/core/nvtx.h`, swap for hipTX or no-op), cooperative launch in exactly ONE
  kernel (`src/ops/gdn_gating_proj/bf16/.../gemm_mma.cuh` grid-sync — split into
  two kernels on HIP), P2P in `stage_link` (single-GPU: dormant), NVML power
  polling in `power_throttling.cpp` (dlopen'd; needs a sysfs/amdgpu equivalent or
  disable).
- **Numerics to respect**: ReplaySSM (GDN speculative state) requires the replay
  fold kernel to reproduce the online recurrence **bit-identically**
  (`include/ninfer/ops/gdn_replay.h`). Any HIP GDN kernel must be qualified against
  the repo's own oracle contract (`docs/maintainer/op-development.md`) before use.

### Weight/quant formats (converter-side is already portable)

`.ninfer` v3 artifacts (content-addressed objects + JSON catalog), produced by the
**Python converter** `tools/convert/` (CPU/torch — runs unchanged on this box).
Formats implemented: BF16/FP32; groupwise Q4/Q5/Q6-G64-FP16, Q8-G32-FP16;
NVFP4; FP8-E4M3 per-row; **T2_G128_FP16 ternary (Hadamard-rotated, bonsai2-style)**;
9 GGML block types (Q8_0, Q6_K, IQ2_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q2_0)
with hand-ported ggml MMQ/MMVQ kernels; GGML expert-record formats for MoE banks.
KV-cache formats are a separate enum (`KvCacheStorage`, `include/ninfer/types.h`).
The ternary + GGML-codec dequant logic is mostly integer/table work that ports
directly; the MMA plumbing around it does not.

## 5. What `AGENTS.md` demands of the port (project rules, not mine)

- One resident model, one GPU, 1–8 lanes; multi-GPU pipeline is parked; no TP.
- Every inherited tuning/route constant "is a hypothesis until measured on this
  card" — a third hardware target re-arms every route table in `src/ops`.
- No plugin discovery / string-driven execution → the HIP backend must be a
  compile-time sibling, selected at configure, not a runtime abstraction.
- Numerical changes require FP64/FP32 oracles per Op at production shapes.
- Tests: `tests/cmake/TestGpuClasses.cmake` labels `cpu`/`gpu`; the CPU-labelled
  suite ports as-is and is the regression net for all host code.

## 6. Port strategy — dual backend (upstreamable)

**Phase 0 — build seam (~days).**
Backend switch in the 3 CUDA-aware CMake files: root `CMakeLists.txt:12-29`
(the `80|86|89` FATAL_ERROR gate + `project(... CUDA)`),
`cmake/Dependencies.cmake` (`find_package(CUDAToolkit)`),
`cmake/NinferTargets.cmake` (the two archive helpers).
`-DNINFER_BACKEND=hip`: compile `.cu` via hipcc, keep the sm8x stub-swap pattern to
exclude `*_tma*`, `*_a4*`, `*_a8*`, Blackwell MMA, and (temporarily) everything
kernel-heavy behind per-directory source filters. Set `NINFER_SM8X_COMPAT=1` for HIP.
Gate: configure completes and `ninfer_text`/`ninfer_serve`/`ninfer_runtime` build.

**Phase 1 — API translation (~1–2 weeks).**
New `include/ninfer/device.h`: `ninferStream_t`/`ninferEvent_t`/`NINFER_CHECK`
aliasing `hip*` under the HIP backend, `cuda*` under CUDA; rename the ~2,500
`cudaStream_t` parameter sites and the 38 host-CUDA `.cpp` files to the seam
(mechanical, tree-wide, one commit — matches the project's no-alias style).
Runtime call translation is near 1:1 (`hipMemcpy` etc.). Special cases:
`cudaMemcpyBatchAsync` → existing fallback path; graphs → `hipGraph` with a
`NINFER_HIP_NO_GRAPHS` kill-switch for v1; VMM → `hipMemCreate` (or stub the
eviction pools to plain `hipMalloc` arenas first and re-enable later);
pinned-host mapped memory over Thunderbolt is suspect → prefer explicit-copy
pinned transfers (`hipHostMalloc` without `Mapped`); NVML → amdgpu sysfs or
disable power-throttle hooks; one-file rocBLAS (or hand-kernel) for the q4 cublas
route; NVTX → no-op. Gate: `ninfer_core` + a trivial op + a decode-graph-free
smoke test on gfx1102.

**Phase 2 — SIMT kernel backend (~4–8 weeks; the bulk).**
Do **not** map PTX→MFMA (no MFMA on RDNA3). Write straightforward SIMT fallbacks
behind the same op interfaces, prioritized by the Bonsai-2-27B dense route:
1. `rmsnorm_rope`, `embedding`/gather, `rows`, casts (trivial);
2. **T2_G128_FP16 ternary GEMV** (the route this machine exists to run — decode-bound,
   dequant is table+shift work; `__dp4a` works on gfx10+). Helpfully, `src/ops/linear/t2/`
   already contains a non-MMA SIMT path (`t2_rowsplit_gemm_simt.cu`) and a GEMV
   (`t2_rowsplit_gemv.cu`) that are the closest existing templates to what gfx1102 needs;
   the t2 dispatch table (`t2_dispatch.cpp`) picks between gemv/small-t/mma paths.
3. GGML `mmvq`-style GEMV for Q4/Q5/Q6/Q8 (decode path), groupwise dequant logic
   reused verbatim;
4. `softmax_attention` — start with a naive two-pass FA reference kernel for
   correctness, tile later with LDS; 125 files exist but one correct path first;
5. GDN recurrent + `gdn_replay_fold` — arithmetic-preserving port, **must** meet the
   bit-identical oracle contract (or ship with speculation disabled initially);
6. MoE topk/routing + `linear_swiglu` variants (needed only for the MoE artifact);
7. bf16/fp16 GEMM paths: FP32-accumulation SIMT (RDNA3 has no hw bf16 — software
   bf16 or fp32; prefill will trail the 3090, decode barely cares).
Every kernel inherits the FP64-oracle qualification requirement.

**Phase 3 — memory system (~1–2 weeks).**
Arenas/`hipMalloc`, re-measure VMM granularity for the eviction pools, then enable
`hipGraph` for decode and re-measure against graphs-off. Per AGENTS.md: every route
table in `src/ops` (tile-size/launch-bound selectors) is invalid on this hardware
and needs sweeping for the surviving routes only.

**Phase 4 — artifacts + validation.**
Converter runs unchanged (Python, CPU). Build a T2/Bonsai-class artifact for
Bonsai-2-27B (verify hybrid geometry first). Validation ladder:
(1) CPU-labelled ctest suite (seconds, no GPU);
(2) per-op HIP tests vs FP64 oracles at production shapes;
(3) `ninfer-perplexity` end-to-end vs the llama-prism numbers on the same checkpoint;
(4) serve + OpenAI-API smoke vs the running PrismML server.

**Do first, de-risks everything:** Phase 0 + 1 end-to-end with one trivial kernel —
proves hipcc on the C++20 codebase, the `ROCR_VISIBLE_DEVICES=1` iGPU landmine,
hipGraph behavior, and the CMake seam in days, before any kernel investment.

## 7. Honest cost

- Phase 0–1: **2–3 weeks**. Mechanical but broad (~234–351 files touched).
- Phase 2 to first correct full forward pass: **4–6 weeks**; to a competitive
  decode route for the ternary model: **+4 weeks** of tuning. Call it
  **~2–3 months focused work to first useful token, 4+ to "good"**.
- Ongoing: route-table sweeps on every ops change; upstream fork drift (this fork
  pins sm_86 and rewrites route tables constantly — the HIP backend must be kept
  green or explicitly parked with CMake-level isolation).
- Performance outlook: decode at batch 1 ≈ memory-bound → same band as llama-prism
  (15–25 tok/s), possibly better with NInfer's speculation once ReplaySSM ports.
  Prefill will be **slower** than the 3090 (no tensor cores; SIMT fp32). That is
  the price of RDNA3, not of this codebase.
- Alternative rejected for now: hipify-style whole-tree translation — it produces
  code that compiles but keeps CUDA-shaped assumptions (TMA guards, mma layouts)
  and can't be upstreamed against AGENTS.md's explicit-architecture rule.

## 8. Open questions

1. ~~Does Bonsai-2-27B use the Qwen3.8 hybrid GDN geometry?~~ **Answered: yes** —
   the in-repo artifact is `Qwen3_5ForCausalLM` and downloads via the
   `bonsai2-27b` launcher profile; no converter work needed, the artifact can be
   downloaded and used directly by the ported engine.
2. Does `hipcc` on ROCm 7.1 digest C++20 + the ~150 PTX sites' headers far enough
   that the Phase-0 source-filter approach compiles (Phase 0 answers this in days).
3. hipGraph capture legality for NInfer's `ThreadLocal` full-step capture on
   gfx1102; fallback without graphs is mandatory-measurable.
4. Whether to upstream into this fork (it self-describes as sm_86-tuned; maintainers'
   appetite for a second backend unknown — ask before building CI around it).
5. 40k ctx ceiling seen with llama-prism: verify it's a VRAM wall, not a llama.cpp
   quirk, since NInfer's KV/page math differs.

## 9. Repro facts / pointers

- Analysis snapshot: `~/git/ninfer-3090` (clone of Don-Chad/ninfer-3090 @ v0.16.1).
- eGPU landmines + working Bonsai invocation: `~/bin/bonsai-rocm.sh`.
- gfx1102 Tensile fix: `~/opt/rocblas-gfx1102/library` (from ROCm 7.2 noble deb).
- Key files: root `CMakeLists.txt:8-29` (arch gate), `src/ops/CMakeLists.txt:53-58`
  (stub-swap template), `include/ninfer/ops/*.h` (cudaStream_t seam),
  `src/core/{device.h, decode_graph.cpp, cuda_vmm.h, copy_batch.cpp, pdl.cuh}`,
  `src/ops/common/{mma.cuh, memory.cuh}` (PTX concentration),
  `src/ops/linear/t2/` (ternary kernels), `src/ops/offloaded_sparse_moe/cpu/`
  (portable AVX2), `tools/convert/` (portable Python converter).
- PrismML reference: https://prismml.com/news/bonsai-2-27b · ROCm: https://github.com/rocm/therock
