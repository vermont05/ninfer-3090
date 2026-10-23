# Remaining work (handoff, 2026-10-21, updated 2026-10-22)

Status: decode advanced past the pause — the Prism mmvq schedule shipped as the T=1 dp4a
GEMV (723f2b7a, cal17 bit-exact battery), short graph-on runs read **~18.7 t/s** vs the
17.3–17.4 pre-schedule figures (short-run observation, not a matched end-to-end A/B; the
30 t/s target stays parked). The next thrust is **prefill** — the §1 correctness gates
closed 2026-10-22 (evidence now in §1). Ordered by priority: correctness → contract →
prefill perf → decode backlog. Every GPU run here needs the standing env recipe
(`ROCR_VISIBLE_DEVICES=1`, Tensile path, `NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off`; see
[[hip-egpu-env-landmines]], [[t2-dispatch-needs-simt-env]]). A/B baselines stay
NInfer-vs-NInfer; the Prism end-to-end comparison is **deferred by user instruction, not
banned** — don't run it unasked, and reading Prism for design stays fine.

## 0. State of the decode route (done — context, not TODO)

- dp4a byte-dot GEMV is the default T=1 decode route (`t2_gemv_dp4a.*`, dispatch hook in
  `t2_dispatch.cpp`). Step 68.5 → ~57 ms/token across the port.
- Pair-path fold shipped (`gdn_input_proj.cpp` width-1 branch): one quantize feeds three
  GEMVs straight to destination. proj 0.229 → 0.191 ms/layer.
- CUDA graphs work and boot clean ("CUDA graphs ready"). **The +0.4 t/s claim is NOT
  established** — the one controlled 64-token pair measured 17.410 vs 17.355 (noise). Treat
  graphs as "free, keep on," not as a throughput lever. Probe runs need `--no-cuda-graph`
  (the fine probe mis-accounts under replay; see [[cuda-graphs-on-probe-distorted]]).
- Arena-backed scratch + policy gate (`allows_a8_int`) + zero-group scale fix are in
  (commits 6784c2df, 65d051b7). `CoreTests.cmake`/`hipshim` build-infra fixes let the test
  bundle configure on HIP — the bundle now builds and the §1/§2 tests listed above pass
  (run 2026-10-22).

## 1. Correctness first (blocks trusting any output)

The dp4a route produces plausible-but-incoherent generations (invented conversation
history, wrong discourse). The **bits route is incoherent the same way** — which points away
from quantization (route-independent signature) toward prompt/template/conversion/prefill.
This is the top open question. Bisect **without Prism**:

- [x] **mma.cuh transpose helper bug — FIXED, test added, PASSING.** `ninfer_mma_ldmatrix_test`
  passes, as do the affected Op tests the fold and routes need: `ninfer_gdn_input_proj_test`,
  `ninfer_linear_t2_a16_test`, `ninfer_linear_t2_a8_test`, `ninfer_gdn_input_proj_conv_snapshot_test`,
  `ninfer_gdn_input_proj_conv_record_test` (2026-10-22 bundle).
  `bcast_pair_t` dereferenced its row base and added `col` to the loaded value instead of
  indexing the row (sibling `pair_at` shows the correct pointer-offset form). Fixed to index
  the element; `tests/test_mma_ldmatrix.cu` (distinct-value 32x8 matrix, all four ldmatrix
  refs, host-oracle fragment compare) registered as `ninfer_mma_ldmatrix_test`. Original note:
  `*reinterpret_cast<const std::uint16_t*>(r0) + col` — dereferences r0 (row base) then adds
  `col` **to the value**, instead of indexing the row at column `col`. Compare the sibling
  `pair_at` (line 58) which correctly offsets the *pointer*. Correct form: index the element
  at column (e.g. `*reinterpret_cast<const std::uint16_t*>(r0 + std::uintptr_t(col)*2)`),
  matching `pair_at`'s byte addressing. Its trans path feeds `ldmatrix_x2_t`/`_x4_t`, used by
  the chunked GDN recurrence (`gated_delta_net/chunked/recurrence.cu`,
  `kimi_delta_attention/chunked/recurrence.cu`) and most attention kernels
  (`softmax_attention/**`, `qsa/attention_*`).
  Blast radius — **answered: it covers decode, not only prefill.** The `.trans` helpers
  are called from `softmax_attention/dense/causal_cache/small_t_i8.cuh:780` (the small-T
  path, which has T=1 instantiations such as `small_t_i8_g16_t1_3.cu`), from
  `context_query.cuh`, and from the chunked GDN/Kimi recurrences — so the T=1 decode step
  routes through it too. And because `mma.cuh` is the **shared SIMT emulation layer**
  (its own header: "GCN has neither ldmatrix nor an mma.sync tensor core"), not a route,
  the bug hits the dp4a and bits decode routes *identically*. That is the one defect so
  far that reproduces the incoherence's route-independent signature, so it is a prime
  suspect for §1's top question, not just a prefill-side cleanup.
  Fix + test added; run the new `ninfer_mma_ldmatrix_test` plus the affected Op tests
  before believing any downstream number — **done 2026-10-22 (all listed above pass)**.
  Still unverified end-to-end: the route-level logit oracle below.
- [x] **Chat-template / prompt assembly — DONE (2026-10-22).** Two parts. Provenance: the
  bundle template is byte-identical to the GGUF-derived reference (8,952 bytes, sha
  c3cf9e34abf4f9e3…). Rendering: ninfer's `JinjaTemplate` vs the jinja2 reference agree
  6/6 cases (system+user, multi-turn, add-generation-prompt, no-thinking, multiline) —
  harness `.qual/qual1/template_check.py` + `cases_render.json`. Template exonerated.
- [x] **Base continuation sanity (no template) — DONE (2026-10-22).** Four raw prompts
  (`/tmp/q1_base_base.txt`): coherent continuations, byte-identical to the
  `NINFER_T2_DP4A=off` arm — no invented history on base text. The invented-history lines
  sit in the model's own reasoning text (visible output stays on-topic), model-side.
- [x] **Logit oracle, in situ — DONE at route level (2026-10-22).** Full-serve chat battery
  (`.qual/qual1/chat_battery.sh`, 8 fixed cases, temp 0) across three arms — shipped,
  `NINFER_T2_DP4A=off`, `NINFER_T2_A8_WMMANATIVE=off`: shipped ≡ WMMA-off **8/8 identical**;
  the dp4a-off arm differs in exactly one case (`facts`: one argmax flip at a near-tie,
  paraphrase-level, other 7 identical). Routes agree end-to-end. A full FP64-dequant
  corpus diff stays parked with §4's numeric qualification. Quality sample: the 40,510-
  token coherent corpus scores **PPL 4.392** (bf16 KV, `/tmp/q1_ppl_bf16.out/report.json`).
- [x] **Run the test suite once the bundle builds** — **done 2026-10-22.**
  `build-hip-test/tests/ninfer_tests` ran green: mma_ldmatrix, gdn_input_proj (covers the
  shipped fold), t2_a16, t2_a8, gdn_input_proj_conv_snapshot, gdn_input_proj_conv_record.
  One test-side gate added: conv_record skips its NVFP4 leg on HIP — the record path's
  NVFP4 A16 schedules request 73 KiB dynamic smem (> the 64 KiB HIP per-block cap), so
  the one-time `cudaFuncSetAttribute` fails and `CUDA_CHECK` aborts the process before
  any case can even report a skip. Pre-existing platform gap, not a Q8/fold regression
  (FP8 A16 record legs, which fit, run and pass; FP8 A8/NVFP4 A4 skip via the stub
  refusals).
- [x] **`ninfer-perplexity` aborted on its default Int8Group64 KV — FIXED 2026-10-22.** The
  i8 FA2 prompt kernel (`CausalPromptFa2Shape<8,false,false>` ≈ 98 KiB dynamic smem)
  exceeded gfx1102's 64 KiB LDS ceiling: the one-time `cudaFuncSetAttribute` threw
  `hipErrorInvalidValue` and the site's `CUDA_CHECK` aborted before any token scored —
  the same trap commit 24306ae4 fitted the bf16 causal tiers for, which this launcher
  never got. Fixed with the same shape-descent idiom: `CausalPromptFa2Shape`/the kernel
  take a `Stages` depth, and a HIP-gated `CausalPromptFa2LdsFit` rebuilds an over-ceiling
  2-stage shape as a 1-stage (and, for 8-warp, 4-warp) variant — the only INT8 geometry
  that fits (51,200 B). The 1-stage path copies and computes each 64-key tile sequentially
  (no double-buffer overlap) and keeps its Q group scales in a dedicated LDS region (the
  2-stage path borrows the unused stage); the QK/PV math is unchanged and shared. Verified
  on gfx1102: the default `int8` path now scores (ppl 4.3917 over the §1 corpus) and
  matches the `bf16` reference (ppl 4.3920) to four decimals. The `--kv-dtype bf16`
  workaround in `.qual/qual1/ppl.sh` is no longer required.

## 2. Contract fixes (from Codex review — both decided 2026-10-21)

- [x] **T2 gate policy — DECIDED: keep `allows_a8_int`, documented.** `t2_dispatch.cpp` keys the
  T=1 dp4a route on `allows_a8_int`, and `include/ninfer/ops/linear.h` documents `AllowA8Int`
  itself as "Admit either A16 or integer-A8 compute profiles" — while `AllowA8IntDecode` is
  documented as the *additional* q4 small-T route for the 27B gate_up, not a decode-wide
  integer gate. The existing gate comment states the policy rationale; tightening the T2
  gate to `AllowA8IntDecode` would contradict the enum docs. No code change.
- [x] **Q8 GDN route now honours caller policy.** The byte-dot decode launches
  (`q8_gdn_input_decode.cu`) took `workspace != nullptr` as their whole integer-route
  admission — an explicit `A16Only` call could still get s8 quantisation. They now take an
  `allow_integer_route` flag from the wrapper (`allows_a8_int` of the *original* policy,
  captured before `without_integer` normalisation), threaded through
  `q8_gdn_input_dispatch`; the snapshot call site passes it too. Matches how the T2
  two-parent fold already gates on policy.
- [x] **Q8 capacity queries follow the exec gate — landed + verified 2026-10-22.**
  Interval accounting is in (`min_tokens > 1` → 0; the snapshot scratch leg only counts
  when `min_width <= 1`), and the previously-expected-fail high-water checks pass:
  `ninfer_gdn_input_proj_conv_snapshot_test` and the now-unaborted conv_record are green.
  Original note: the Q8 capacity
  functions bake in the dp4a scratch *without* looking at policy (or at `NINFER_Q8_DP4A`),
  so once exec is policy-aware the policy-bearing and shape-only queries disagree with
  execution for declined policies (the conv_snapshot/conv_record high-water failures).
  The agreed shape — env-aware route predicate (`q8_gdn_input_dp4a_route_live`), policy
  gate on both capacity entry points, shape-only snapshot overload returns 0 for the
  scratch leg, and the Q8 special case folded into the policy-bearing call in
  `src/models/qwen3_5/execution/gdn.cpp` — is what shipped; the high-water failures are
  gone.

## 3. Prefill (perf thrust — §1 gates now closed)

No fresh end-to-end prefill profile exists; old logs are indicative only (short prompt
~10–11 tok/s, 948 tok → 45 s, 3600 tok → 175 s; attention cost grows hard with context).
Leading hypothesis (Codex): the CUDA-shaped T2 matrix emulation is the bottleneck.

- [x] **Profile one real Bonsai prefill — DONE (2026-10-22, in-process probe).** The
  instrument is `PrefillProbe` in `text.cpp` (env `NINFER_PREFILL_PROFILE=1`; runs via
  `.qual/prefill_probe_run.sh`): per prefill chunk it folds cudaEvent intervals into
  embed | layers | tail, splits layers by stage kind (gdn/attn × mixer/mlp), and samples
  every 4th GDN mixer into nc|proj|rec|norm|out. Route + policy are echoed at first use —
  no external profiler exists on this box.

  Attribution of the real chunk (prompt 1,065 → one 1,024-token chunk, request ~24 s
  wmma / ~41 s anchor), env echo `NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off`:

  | route | layers | gdn_mlp | gdn_mixer | attn_mlp | attn_mixer |
  |---|---|---|---|---|---|
  | anchor (WMMANATIVE unset) | 34,759 | 17,342 | 8,696 | 5,776 | 2,944 |
  | wmma (`NINFER_T2_A8_WMMANATIVE=on`) | 17,698 | 8,336 | 4,821 | 2,774 | 1,768 |

  (bucket ms per chunk; buckets sum exactly to the layers interval — no unattributed time.)
  GDN-mixer fine split (16 samples/chunk, wmma, ms per sampled layer): proj 56.6, rec 22.8,
  out 19.9, nc 2.5, norm ~0.2 — the mixer is GEMM-heavy too, and the recurrence is ~6% of
  the chunk (≈23 ms × the sampled GDN layers ≈ 1.1 s of 17.7 s).

  Conclusion: the chunk is ~90% GEMM-class work on both routes (MLP stages 66%, mixer
  projections the bulk of the rest); attention mixtures are the smallest buckets. The
  earlier "GEMMs plausibly carry half" note below was an understatement.

### The sub-64-token dead band (2026-10-22)

The ~5 s warmup chunks above were **not** first-launch overhead: with the band off (the
recipe env) a chunk of T ∈ [17, 63] tokens hits `Route::None` and runs the A16 dispatch
schedules — the probe showed the boot chunk (51 tokens) at 5.1 s and a request-tail chunk
(34 tokens) at 5.0 s, versus 5.9 s for a 1,024-token wmma chunk. Re-routing the same chunks:

| chunk | band off (A16 route) | small-T band [17,192] | prefill tile min=17 |
|---|---|---|---|
| boot 51 tok | 5,122 ms | 4,917 ms | **464 ms** |
| tail 34 tok | 5,012 ms | 2,773 ms | **369 ms** |
| 1,024 tok | 5,947 ms | (untouched) | 5,928 ms |
| request TTFT (1,065 tok) | 12.0 s | 9.7 s | **7.3 s** |

The small-T i8 band is no rescue on this card either: its mma comes from the same
`mma.cuh` SIMT emulation and it loses to the hardware
tile at every tested width. Shipped change (2026-10-22): the HIP build defaults the band
to off (env still overrides; CUDA keeps the band), and `kT2A8MinTokens` dropped 64 → 17,
so every T ≥ 17 chunk takes the wmma tile. The recipe env (`NINFER_T2_I8_SMALL=off`)
became redundant-but-harmless on HIP builds. The boot chunk is one-time; the request
tail recurs on every prompt whose length mod 1024 < 64 — and one 5 s tail was ~41% of the
W=1024 request's old 12 s TTFT.

### Measured baseline (2026-10-21, cold-boot battery `.qual/prefill_ab.sh`)

Each leg = fresh server boot, one chat request, prefix cache empty. Medians of 3, all
repeats within ±0.1 s:

| W (target) | prompt | TTFT | rate |
|---|---|---|---|
| 128 | 174 | 7.9 s | 22.1 tok/s |
| 512 | 553 | 21.0 s | 26.3 tok/s |
| 1024 | 1,065 | 41.4 s | 25.7 tok/s |
| 4096 | 4,100 | 148.9 s | 27.5 tok/s |

Route costs at the three registered shapes (`.qual/t2tile_probe.cu`, L2 flushed, median of
9, env recipe): the shipped int8 route (`t2_a8` → `t2_prefill_i8` tile, `AllowA8Int`) runs
**~1.5 TOP/s** (gate_up T=1024: 239 ms; mlp/down: 124 ms; gdn/value_z: 86 ms) and the A16 route
**~0.46 TOP/s**. WMMA's measured ceiling on this card is ~58 TOP/s ([[gfx1102-wmma-i8-layout]]),
so the emulation in `mma.cuh` is costing ~40x against the achievable bar on the biggest GEMMs —
that is §3's target. At T=1024 those three GEMMs alone are ~450 ms per layer; a 1,065-token
prompt spends ~36 s per chunk, so the GEMMs plausibly carry half of prefill.
- [x] **T2 prefill matrix emulation — DONE (hardware int8 WMMA), now the default route.**
  `src/ops/linear/t2/t2_prefill_wmma.cuh` (`t2_prefill_wmma1_kernel<MF,NF>`, (MF,NF)=(1,2)):
  one hardware `v_wmma_i32_16x16x16_iu8` per 16x16 tile replaces the `mma.cuh`
  shuffle-gather emulation; the weight side decodes on the fly from the T2 payload
  (per-group 32-byte lines, same bytes/order the shipped tile cp_async's), so the
  quantiser and every epilogue are shared unchanged and outputs are **bit-identical**
  to `t2_prefill_i8` (cal23 battery: 16 variant/shape pairs, 0 bit diffs). `NINFER_
  T2_A8_WMMANATIVE=off` puts the emulation back for A/B.
  - **Spill saga (cal20→23):** cal20 proved the first shipped tile (4-warp block,
    cal18 loader) ran ~2.4x slower than cal18's standalone probe — resource remarks
    showed 256 VGPRs / 470 spilled / 1376 B/lane scratch at 4 waves/SIMD. cal21's
    structural variants only helped when one-warp; cal22 found the root cause: the
    fold's duplicated xs loads must be HOISTED once per group (before the m-loop) —
    with the hoist every used tile is spill-free (≤223 VGPRs, 0 scratch) and fast.
    cal23 swept (MF,NF) and pickled (1,2) one-warp: gate_up 28.7 ms / mlp_down 15.1 /
    valuez 10.4 (shipped 4-warp: 112.7 / 57.1 / 40.3; anchor 264 / 135 / 95).
  - cal19/cal23 batteries (`.qual/wmma/`): bit-exact vs anchor; `ninfer_linear_t2_a8_test`
    passes on both routes (Full comparison at 1024 rows), default-on and `=off`.
  - Cold-boot A/B (2026-10-22, 3 reps each, spread ≤0.3 tok/s): vs the base-table
    anchor — 22.1→**81.9** (W=128), 26.3→**126.0** (W=512), 25.7→**87.2** (W=1024),
    27.5→**126.8** tok/s (W=4096); TTFT 1024 41.4→**12.2 s**, 4096 148.9→**32.3 s**
    (old 4-warp wmma route for scale: 39.1 / 49.3 / 44.3 / 51.1, TTFT 1024 24.1 s;
    cal23 standalone for the old 4-warp: gate_up 112.7 ms). Standalone cal18 loader
    still sits at ~47.5 ms on gate_up — the remaining ~1.7x gap to cal23's (1,2) is
    loader-shape, not spill.
  - Follow-ups: attention/GDN co-bottlenecks (§3 next bullet) gate any further
    prefill attribution; the (1,2) tile's win is pinned on the three engine
    shapes only — new widths want a quick probe before shipping.
- [ ] **Attention / GDN recurrence** are plausible co-bottlenecks at long context once §1's
  helper is fixed. Any new tile: bounded per-tile workspace and measured peak VRAM (the KV
  config already leaves ~100 MiB free; a big expanded-weight buffer won't fit).
  - Probe line (2026-10-22): the f16/bf16 builtins
    `__builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32` (3-arg vector form) are correct on
    this card — cal25l/cal26 batteries 0/256 vs the dot oracle (chained K=128 included;
    maxdiff 1.8e-5 is the macro's fixed ULP noise, not corruption). cal30: the same
    chained loop is ~33x the `mma.cuh` emulation (SIMT 213 ns vs WMMA 6.4 ns/launch,
    ABBA, 500 iters) — the throughput premise for moving the GDN recurrences off SIMT.
    cal32 pinned the loader slot map (A-side slots {2c | 17+2(c−8)}, B-side 32 rows with
    row r duplicated at r+16): cal29 map 0/256, clean doubling 128/256 wrong (second-half
    windows), dense 236/256 wrong — the loader stages verbatim. Probes:
    `/tmp/cal25/cal2{6,9}.cu`, `cal30`, `cal31`, `cal32`, `cal34`, `cal35` (narrow 16x8 tile,
    column-gather S-side → transposed left matrix; site-3/matrix_delta shapes).
  - GDN recurrence WMMA variant **shipped** (2026-10-22). `chunk_recurrence_wmma_kernel`
    in `gated_delta_net/chunked/recurrence.cu` (+ new `recurrence_wmma.cuh` helper: layouts,
    narrow/full S/F loaders, phases site2/dot/state) for the DV=64 value tile, behind a
    default-on env gate `NINFER_GDN_REC_WMMANATIVE` (`=off` → SIMT). Site-2 keeps the
    both-mma-before-accum ordering (avoids LDS spill on w32).
  - Correctness: `ninfer_gated_delta_net_test` + `ninfer_gdn_state_fp16_chunked_test` PASS
    with the gate on (all three profiles within rel_l2 limits; 27b H=16/48 T=2049 cases
    exercise the variant).
  - A/B (2026-10-22, private-launcher `ninfer_gated_delta_net_bench --force-chunked
    --breakdown`, cold-L2 graph replay, 20+100/case, 3 reps ABBA per arm, T=1024/2049/4096,
    H=16/48): **chunked.recurrence 11.4–11.5× faster** (1649.9 vs 18860.6 µs @T=1024;
    3351.1 vs 38431.4 @2049; 6668.5 vs 76883.0 @4096); **chunked pipeline (prepare+recurrence)
    ~5.1–5.3× end-to-end** (4.11 ms vs 20.9 ms @1024); `chunked.prepare` unchanged to noise
    (gate touches only the recurrence). cal30's micro-bench premise (6.4 vs 213 ns) holds
    in-kernel at 11.5×; the remaining gap is bandwidth + the prepare stage.
- [ ] No promised prefill tok/s target yet. First deliverable is a corrected, measured
  baseline + breakdown, not a number.

### Battery re-run on the shipped route (2026-10-22)

Same cold-boot battery as the 2026-10-21 anchor table, same env recipe, one boot per
leg, 3 reps/width with spread ≤0.2 s — this is the table an A/B must beat now:

| W (target) | prompt | TTFT | rate |
|---|---|---|---|
| 128 | 174 | 1.9 s | 90.4 tok/s |
| 512 | 553 | 3.9 s | 140.5 tok/s |
| 1024 | 1,065 | 6.6 s | 162.6 tok/s |
| 4096 | 4,100 | 28.8 s | 142.2 tok/s |

(Per-leg logs `/tmp/pab_wmma_*.log`.) The W=1024 point reads ~1.8x the ship-time A/B
note's 87.2 tok/s (12.2 s) and the other widths 5–10% above it — until that note is
re-checked under matched conditions, treat this table as the current shipped-route
reference.

### Baseline re-checked + GEMM tile sweep (2026-10-22)

- **Baseline re-checked under matched conditions.** A fresh base cold-boot battery
  (`.qual/prefill_ab.sh base 1024`, 3 reps, no raster override, recipe env) reads
  **166.0 / 165.2 / 165.1 tok/s** → median **~165 tok/s** at W=1024 for the **1,065-token
  prompt specifically** (other prompt sizes on the same route: ~90 tok/s @ 174 tokens,
  ~141 @ 553, ~142 @ 4100). This is the baseline for *this* prompt, not a general rate. The
  gap to the 250 tok/s target on this prompt is **~1.52x** (~6.45 s TTFT → ~4.26 s, about
  2.2 s of headroom to find).
- **87.2 tok/s was a real earlier-configuration number, not a lost measurement.** The
  ship-time A/B note's 87.2 tok/s (12.2 s) predates two later fixes — the short-chunk
  routing fix and the new GDN recurrence kernel — and together those drop the 34-token
  decode tail from ~5 s to < 0.5 s. The 87.2 → 165 improvement is attributable to those
  prior fixes, not a measurement artifact; it is not "not reproducible," it was a different
  (older) build.
- **GEMM tile/warp-shape sweep: the tested loader variants lose to the shipped tile**
  (`.qual/wmma/cal36.cu`, `cal37.cu`; full battery bit-exact vs the i8 anchor, cold-L2
  medians on the three engine shapes). cal36 swept the one-warp family — `(1,2)@8`
  (shipped), `(1,2)@16`, `(1,4)@8/16`, `(2,2)@8`, `(2,4)@8` — plus the batched-loader `wp`
  family: the shipped `w1(1,2)@8` is the fastest on all three shapes; fatter-N tiles are
  ~1.9–3.3x slower and `wp` is ~2–2.4x slower (hoisting all `av[8][NF]` fragments blows the
  register budget). cal37 swept multi-warp *block* loaders — 2/4/8 warps per block, each a
  `(1,2)` warp tile, block tile `(16·WM)x(32·WN)`, **no LDS operand sharing** (each warp
  loads its own weight lines) — and every multi-warp variant loses to the one-warp route by
  1.4–2.3x (best: 8-warp `wb(2,4)`).
  **Scope of that conclusion:** because cal37's multi-warp blocks do *not* share operands
  through LDS, the sweep does **not** establish whether a *cooperative* LDS-staged loader
  (one block loads a tile into LDS once, all warps read it) would win — that is still open.
  What is settled is only that the tested variants of the *existing per-warp loader* lose.
- **Lever 1 (GEMM) is characterized and is a NO-GO: the kernel is DRAM-bandwidth-bound at
  ~80% of the physical ceiling, so there is no meaningful TTFT headroom here**
  (2026-10-22, `.qual/wmma/cal38.cu`; battery bit-exact vs the i8 anchor, cold-L2 medians).
  The device is an **RX 7600M XT: 16 CUs, 2 MB L2, 64 KiB LDS/block, 8 GB** — not the 28 CUs
  the handoff assumed. A raw triad over 256 MB (≫ L2) measures the achievable **DRAM BW
  ceiling at 237.6 GB/s** (the naive `memoryClockRate` estimate of ~36 GB/s is far below this
  and is unreliable for GDDR6 — do not use it). The shipped `w1(1,2)@8` tile streams its
  dominant traffic — weight re-reads (W is N×K int8, re-read by all T/32=32 token-blocks since
  W ≫ 2 MB L2) — at:
    | shape | weight-code traffic | shipped time | eff weight BW | % of 237.6 GB/s |
    | gate_up 34816×5120×1024 | 5.70 GB | 30.6 ms | 186 GB/s | 78% |
    | mlp_down 5120×17408×1024 | 2.85 GB | 15.1 ms | 189 GB/s | 80% |
    | gdn_valuez 12288×5120×1024 | 2.01 GB | 10.4 ms | 194 GB/s | 82% |
  Two targeted ablations settle the two remaining questions: (a) the **no-epilogue** variant
  (identical loads + WMMA burst, per-group fold/store removed) runs *slower* (97–139 GB/s),
  so the per-group fold ALU and store epilogue are **not** the bottleneck — they are off the
  critical path; (b) the fatter-N tiles are ~1.9–3.3x slower (cal36), so the tile is not the
  lever either. The kernel is bound by weight-traffic throughput and already sits at
  ~80% of the DRAM ceiling. **Max achievable savings** if it hit the full 237.6 GB/s ceiling:
  gate_up 30.6→24.0 ms, mlp_down 15.1→12.0 ms, gdn_valuez 10.4→8.5 ms ≈ **~11.6 ms per
  1024-token chunk**, i.e. ~0.5% of the 2.2 s TTFT budget — far below the bar for a major
  rewrite. Weight staging/persistence cannot help: W (178 MB gate_up) exceeds LDS (64 KiB) and
  L2 (2 MB), so it cannot be kept resident, and a weight-stationary layout would still be
  capped at the same 237.6 GB/s DRAM ceiling. **Do not pursue the activation/weight-staging
  redesign.**
  **Whole-prefill consequence:** the chunk is ~90% GEMM-class work (MLP stages 66%, mixer
  projections the bulk of the rest — those are GEMM-heavy too; see the wmma bucket split above),
  with attention the *smallest* bucket and the GDN recurrence ~6%. So the non-GEMM remainder is
  both small and largely GEMM-dominated, and it cannot close a 2.2 s gap. **On the RX 7600M XT
  the prefill is DRAM-bandwidth-bound on weight streaming at ~80% of the card's 237.6 GB/s
  ceiling, and the 250 t/s target (~33% more prefill throughput) is not reachable by kernel-level
  optimization on this hardware.** Reaching it would require a wider/faster DRAM bus, a more
  compressed weight stream, or a lower target for this card.
- **Raster A/B is inconclusive** (2026-10-22, `NINFER_T2_A8_RASTER`, W=1024 cold-boot, 3
  reps): default `cols`-fast (x=columns) median **165.2 tok/s** vs `rows`-fast (x=rows)
  **163.7 tok/s** — a ~1% delta. Two 3-run sets do not by themselves establish a noise
  distribution, so this is treated as "no dispatch change warranted," not as proof the block
  raster is inert; the block-raster is not a lever worth further A/B on this card.

## 4. Decode backlog (only if decode is resumed — parked for now)

- §1's mma.cuh helper bug means every Phase-4 decode bucket is pre-fix evidence. Fix it,
  then re-trace before choosing a new lever.
- If decode resumes, start at `docs/decode-30tps-plan.md` **Phase 5** (external review
  2026-10-21): a rocprofv3 kernel trace first, then the cal16 mmvq-schedule candidate
  (Prism's same-codec matvec hits ~93% of the byte-DOT ceiling — the lever is the
  schedule, not the codec), then the unbroken-down attn_mixer bucket.
- gdn_mlp is the biggest bucket (~21–23 ms/step), then gdn_mixer (~16). cal14/cal15 NO-GOs
  exhausted the Q8 + de-staging ideas. cal2-6 family is exhausted — don't re-run.
- Broader numeric qualification (Codex item #4): FP64/logit-diff on a coherent corpus before
  the dp4a route ships beyond this machine.

## Notes on measurement discipline

- Idle GPU before any number: `pgrep -x ninfer-serve` empty AND lock free.
- Probe buckets: `--no-cuda-graph`. Wall/throughput: graphs on.
- Report medians over ≥3 repeats for anything under ~1 t/s; single-pair deltas in noise
  (the 0.4 t/s graph claim is the cautionary tale).

## RDNA3-native roadmap

Int8 WMMA (`v_wmma_i32_16x16x16_iu8`) works on gfx1102; the decode plan's "rejected" entry was
wrong. Ranked list of CUDA-shaped designs to replace: `decode-30tps-plan.md`, "Correction +
RDNA3-native roadmap".

2026-10-21 verification: the builtin `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` computes correct
16x16 products on this card — the probe battery (`.qual/wmma/cal16{c,e,f}.cu`) passes 10/10
random/identity cases, and the fragment loader is pinned byte-exactly
(`C(l,w) = dot(Awin[(l<16?0:17)+2w], Bwin[l])`; Bwin slot l = A row l, Awin window = B column,
byte k = B[k][col]; results land at `(A·B)(l&15, (l>=16?8:0)+w)`). Instruction-bound ceiling
measured ~58 TOPS (cal16 peak loop). The hand-filled-WMMA GEMM path is now unblocked for
roadmap items 1/2.
