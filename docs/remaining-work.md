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
  **Scope of that conclusion:** cal37's multi-warp blocks do *not* share operands through LDS, so
  that sweep alone does not cover a *cooperative* LDS-staged loader. That design was since built
  and measured as `wcoop` in cal38 (see the Lever-1 root-cause note): bit-exact, register-clean,
  and it **loses modestly (+3.5–8.2%)** to the one-warp `w1(1,2)@8`. So the cooperative LDS
  loader is now settled too — it does not win on this card.
- **Lever 1 (GEMM): the T2 prefill GEMM runs at ~19% of the DRAM ceiling in logical
  packed-code terms — there is large headroom; the earlier "tapped out / ~80% of ceiling"
  conclusion is WITHDRAWN (factor-of-4 bandwidth error, corrected 2026-10-22)**
  (`.qual/wmma/cal38.cu`; battery bit-exact vs the i8 anchor, cold-L2 medians).
  The device is an **RX 7600M XT: 16 CUs, 2 MB L2, 64 KiB LDS/block, 8 GB**. A raw triad over
  256 MB (≫ L2) measures the achievable **DRAM BW ceiling at ~240 GB/s** (the naive
  `memoryClockRate` estimate of ~36 GB/s is far below this and is unreliable for GDDR6 — do not
  use it).
  **The error:** cal38 computed "weight traffic" as `N*K*(T/32)` — **one byte per weight**. T2
  packs **four two-bit codes per byte**, so the packed buffer is `N*K/4` bytes, and the logical
  (deduplicated) packed-code traffic is `N*K/4*(T/32)`, i.e. **4× less** than the value the
  benchmark printed. The "186 GB/s / 78% of ceiling" figure was therefore 4× too high. Corrected:
    | shape | logical packed-code traffic (N·K/4·T/32) | shipped time | eff logical BW | % of 240 GB/s |
    | gate_up 34816×5120×1024 | 1.43 GB | 30.6 ms | ~46.6 GB/s | ~19% |
    | mlp_down 5120×17408×1024 | 0.71 GB | 15.1 ms | ~47.3 GB/s | ~20% |
    | gdn_valuez 12288×5120×1024 | 0.50 GB | 10.4 ms | ~48 GB/s | ~20% |
  So the GEMM streams its logical weight codes at **~19% of the DRAM ceiling** — it is **not**
  bandwidth-saturated. A single ceiling-rate pass of the gate_up logical data would be
  1.43/240 ≈ **5.9 ms**, vs the measured 30.6 ms → up to **~5× headroom** on the GEMM, from the
  32× token-block re-reads that L2 (2 MB) cannot absorb against the 44.5 MB gate_up working set,
  plus any latency/occupancy limits. **The GEMM efficiency/traffic lever is OPEN, not exhausted.**
  (The no-epilogue and fatter-tile ablations still show *those specific variants* lose — that is
  evidence about those variants, not a proof the GEMM is at its limit.) Weight staging cannot
  keep W resident (178 MB ≫ 64 KiB LDS, 2 MB L2), so the lever is **reducing the re-read
  traffic / lifting DRAM utilization** (L2-aware panelization, a weight-stationary schedule, a
  higher-occupancy tile that hides the re-fetch latency), not pinning weights in on-chip memory.
  **What this does and does not say — and a second correction.** The *weight-bit* lever I
  proposed earlier ("compress int8 → 2-bit") is a **non-goal: NInfer's T2 weights are already
  2-bit-class.** T2 = ternary codes {-1,0,+1}, two bits each, four per byte, one fp16 scale per
  128 → **2.125 bpw** (`src/core/weight.h:36-38`). The `a8`/`int8` in the kernel names is the
  **activation** dtype, not the weights. So NInfer (T2, 2.125 bpw) and Prism (PQ2_0, 2.13 bpw)
  stream the **same weight bit-rate** — the earlier "int8 streams ~4x the bytes" line was wrong.
  The **same RX 7600M XT** runs the Prism `llama-server` service (`bonsai-egpu`, port 8090) at
  **296 t/s prefill** (4069-tok prompt → 296.2 t/s; decode **22.2 t/s** with ngram spec off,
  21.0 with it — ngram drafts barely accepted at 18.75%, a mild net loss) on that 2.13-bpw 27B.
  **At equal weight bit-rate, the ~1.8x prefill gap (NInfer ~165 t/s vs Prism ~296 t/s) is a
  kernel/access-efficiency gap, not a bit-rate gap:** NInfer's T2 prefill GEMM runs at only ~19%
  of the DRAM ceiling in logical packed-code terms (cal38, corrected) with a 32× weight re-read
  access pattern, whereas Prism's tuned PQ2_0 mmq kernels reach a higher effective throughput on
  the same bytes. **The lever for 300+ t/s is closing that efficiency/traffic gap** — DRAM
  utilization (~19% → near-peak) plus reducing *effective* weight re-read traffic via L2-aware
  tiling / a weight-stationary panel layout that matches Prism's access pattern — and cutting
  non-GEMM (attention/GDN) overhead.
  It is **not** weight-bit reduction (already at the 2-bit floor) and **not** more DRAM.
  **Root cause measured (2026-10-22, `rocprofv3` GL2C DRAM counters on the shipped `w1(1,2)@8`
  via `.qual/wmma/cal38.cu`):** the T2 prefill GEMM over-fetches DRAM by **15–63×** its logical
  weight data. Per launch (warm L2): gate_up reads **22.1 GB** from DRAM to compute on **1.42 GB**
  of logical T2 weight codes (44.5 MB × 32 re-reads) — **15.5×**; mlp_down 11.1 GB vs 0.176 GB =
  **63×**; gdn_valuez 7.6 GB vs 0.51 GB = 15×. The kernel reads weight codes as 8-byte `int8v`
  fragments at 32-byte strides (`line[m] = *(int8v*)(pA + gBase/4)`), so each 8-byte read pulls a
  full **128-byte** DRAM line (`GL2C_EA_RDREQ_128B`); with 2 MB L2 against a 44.5 MB working set
  the lines evict between the 4 sub-reads of one line and across the 32 token-block re-reads, so
  each 128B line is fetched from DRAM multiple times. (gfx1102 exposes GL2C PMC counters in the
  TheRock rocprofv3 1.4.1; no system rocprof/nsys.) **Counter caveat (RECONCILED 2026-10-23 by
  `cal55`/`cal56` — see the reconciliation block below):** the per-dispatch DRAM counters are
  exact, the 22.1 GB warm figure is real traffic, and the "implies ~720 GB/s, impossible" verdict
  was based on the wrong ceiling (the ~240 GB/s number is the mixed read+write triad rate; pure
  read streams sustain ~460 GB/s on this card). The traffic magnitude stands; its interpretation
  changed — the kernel is not DRAM-bound. **The proposed fix — read each 128B line once and derive all four
  32-byte fragments from it (a coalesced `int32v` 128-byte load) — was implemented and measured
  (`cal38` `w1c` vs shipped `w1`, bit-exact bitdiff=0), and that *variant* loses:** it is ~70%
  slower (gate_up 52.5 ms vs 30.6 ms), because the persistent 128-byte register line + the
  quarter-extraction cost VGPRs and occupancy and drag the tuned `(1,2)@8` tile down to the level
  the other variants sit at. **A second, independent design was also built and measured — the
  weight-stationary, LDS-staged tile `wlds`** (`cal38`, bit-exact bitdiff=0): one block owns a
  16-row weight panel *and* a chunk of `R` token blocks, stages a 1024-K weight slice into LDS
  once, and runs all `R` token blocks off LDS so a weight byte hits DRAM once per slice instead
  of once per token block. It **also loses, badly**: gate_up R4/R8 114.8/115.2 ms and R4@16occ
  163 ms vs `w1`'s 30.0 ms (12.4 / 12.4 / 8.7 GB/s logical vs `w1`'s 47.5). The cause is the
  opposite of a traffic problem: folding `R` token blocks into one warp cuts the in-flight warp
  count `R`-fold (w1 launches ~69,632 one-warp blocks for gate_up; wlds R4 launches ~17,408),
  starving the 16 CUs of the occupancy that hides memory/issue latency. **A third design — the
  cooperative multi-warp LDS loader `wcoop` — was then built and measured** (`cal38`, now
  bit-exact bitdiff=0 after fixing a per-slice `__syncthreads` read/write race on the shared LDS
  panel): a block of `NWARP` warps shares one 1024-K weight panel in LDS (loaded cooperatively
  once), each warp runs one token block — so it keeps `w1`'s full warp count *and* cuts weight
  DRAM traffic `NWARP`-fold. **It loses modestly to the shipped `w1(1,2)@8`:** W4 +3.5% gate_up
  / +8.2% mlp_down / +3.6% gdn_valuez, W8 +6.9/+8.9/+6.2% (46.2/44.3/45.5 GB/s vs `w1`'s
  47.8/48.0/47.2). Compiler resources (device-asm `.amdhsa_*`): `wcoop` is **register-clean —
  zero scratch spill, 8–12 KB LDS, 14 SGPR** — unlike `w1c` (320 B spill) and `wlds` (288 B
  spill). So `wcoop`'s small loss is **not** a register/occupancy collapse (that is why `w1c`/
  `wlds` lost 1.7–4×); it is the per-slice `__syncthreads` barriers (5 slices for gate_up)
  serializing the block's warps plus the LDS round-trip. **Conclusion, scoped narrowly:** the
  *cooperative LDS-staged loader* is ruled out as a win on this card. What the three results
  *together* support — and what they do not *prove* from timing alone — is that the GEMM sits at
  ~20% of the DRAM ceiling because it cannot keep the CUs fully fed; reducing bytes (coalesced,
  LDS-stationary) and staging through LDS each remove or serialize the warp parallelism that was
  feeding them. Register pressure vs occupancy vs LDS-staging vs barrier cost are not separated by
  timing; the spill counts above show the two big losses (`w1c`,`wlds`) were spill-driven, while
  `wcoop`'s small loss is barrier/staging-driven. GEMM optimization as a whole is **not** closed
  by this — only this loader family is.

- **DRAM-traffic reconciliation and the real bound (2026-10-23, `cal55`/`cal56`, rocprofv3 GL2C
  counters, shipped `w1(1,2)@8`, gate_up 34816×5120×1024, L2 flushed per timed launch).**
  **The counters are exact.** Calibration kernels with known traffic reproduce to ±5 counts:
  a 44,564,480 B array read exactly once → 348,161–348,165 × `GL2C_EA_RDREQ_128B` (expected
  348,160); a 512 MB pure read → 4,194,305 (expected 4,194,304). The counter unit is the 128-byte
  DRAM read request (gfx11 exposes no `GL2C_EA_WRREQ_128B`; writes count in 64 B units).
  **Cold-L2 w1 = 13.7–13.8 GB reads per launch in 30.4 ms ≈ 450–460 GB/s** — at the card's
  **pure-read** stream ceiling, ≈ the worst-case logical re-read bound (x panels re-read by all
  2176 weight blocks = 11.4 GB + weights re-read by all 32 token blocks = 1.43 GB = 12.83 GB)
  ⇒ **L2 reuse between blocks ≈ 0**: the shipped schedule varies the token dim fastest, so
  co-resident blocks (512 = 16 CU × 8 blocks/SIMD × 4 SIMD) thrash up to 5.2 MB of 163 KB x
  panels through the 2 MB L2. The 22.1 GB warm figure scales with T (the engine prompt runs
  T = 4210, where the x re-read bound is ~52.8 GB) and is real; it looked impossible only
  against the **mixed R+W triad rate (~225 GB/s measured here)** — the ~240 GB/s ceiling the
  caveat used is not the read ceiling (256-bit GDDR6 @ 18 Gbps = 576 GB/s theoretical).
  **Schedule A/B (`cal56`): halving DRAM traffic buys ~4%.** B = `rows_fast=true`, grid
  (2176, 32): consecutive block ids share ONE x panel (fits L2) and stream weights. DRAM reads
  13.7 → 7.2–7.5 GB (−47%; MISS 108 M → 63 M, HIT 85–102 M → 110–119 M) but time moves
  30.4 → 29.7 ms cold (31.3 → 30.1 ms warm). **So the GEMM is not DRAM-bound.** The binding
  resource is the **GL2C request/sector-return path (~850 GB/s of combined return measured here;
  ~6.6 Greq/s)**: total requests (HIT+MISS) 202 M/launch at A, 176 M at B. Per-block sector
  model: x-panel loads = 16 B/lane from 16 rows (`colWin = lane>>1` ⇒ lane pairs broadcast, each
  32 B sector half-used — 50 % waste) → 2560 reqs + weights 320 + out ~32 = **2896 predicted vs
  2898 measured** requests/block; **x loads are 88 % of all requests** (writes show ~1.5×
  amplification: 213.8 MB measured vs the 142.6 MB output array).
  **The lever is therefore requests/block, not DRAM bytes:** requests ∝ 1/MF (x-panel traffic per
  block is fixed, block count ∝ 1/MF) and the 50 % sector waste is removable by LDS staging;
  register pressure (`accf[MF][NF][8]` + int8v accs = 64 i32 at MF1/NF2) trades against
  occupancy (launch_bounds(32, MINB); 8 blocks/SIMD ⇒ ≤64 VGPR). A MF/NF/MINB shape battery
  (`cal57`) is the next probe. Separately, **B is a bit-exact, zero-kernel-change knob** (the
  per-block computation is identical; only the blockIdx mapping changes) worth an engine-level
  A/B — its DRAM halving also relieves cross-kernel L2 pressure in the warm multi-GEMM context.
- **Shape battery: the request model is exact; register spill — not shape — kills big tiles
  (2026-10-23, `cal57`, MF/NF/MINB battery on schedule B, gate_up shape, cold-L2 timed launches,
  timings + GL2C counters + `.amdhsa` metadata).** Cold best-of-5 (ms) with measured requests,
  DRAM reads, scratch, and the time model `t = max(reqs/6.6 Greq/s, DRAM/510 GB/s, wr/225 GB/s)`:

  | shape (MF,NF,MINB) | ms | reqs M | DRAM GB | wr GB | spill B | model ms | binding |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | (1,2,8) control | 29.9 | 176 | 7.85 | 0.214 | 36 | 26.7 | requests |
  | **(2,2,8)** | **29.4** | 197 | 15.0 | 0.187 | 44 | 29.8 | requests≈DRAM |
  | (1,2,16) | 67.5 | 741 | 38.5 | 22.0 | 304 | 97.8 | spill traffic |
  | (2,2,16) | 91.8 | 1080 | 50.6 | 34.3 | 440 | 152.6 | spill traffic |
  | (4,2,8) | 49.3 | 434 | 26.9 | 9.57 | 460 | 65.7 | spill traffic |
  | (2,4,8) | 44.6 | 454 | 19.6 | 10.1 | 492 | 45.0 | spill traffic |
  | (4,4,8) | 104.7 | 1095 | 48.6 | 43.0 | 1272 | 191 | spill traffic |
  | (1,4,8) | 49.7 | 526 | 25.4 | 17.0 | 432 | 79.7 | spill traffic |

  **The request model validates exactly.** x-panel requests ∝ 1/(MF·NF) as predicted: (1,2) 155 M
  → (2,2) 88 M → (4,2)/(2,4) 44 M of x traffic; weight requests ∝ 32/NF per launch
  (32 token-block rows × 4.28 M weight reqs: 143 M at NF=2, 71 M at NF=4), which is exactly the
  DRAM growth (weights re-read 32/NF times: 1.43 → 2.9 → 7.2 GB). The 2896-reqs/block sector
  model carries through every spill-free shape.
  **The return-path ceilings are confirmed:** (2,2,8) at 197 M reqs runs 29.4 ms = 6.70 Greq/s
  **and** 15.0 GB = 512 GB/s simultaneously — both ceilings reached, matching cal55/56's
  independent numbers (≈6.6 Greq/s, ≈510 GB/s pure read). The three-way model predicts **all 8
  shapes within ±10 %**; every loser is bound by **spill write traffic**, not by shape:
  MINB=16 forces 64 VGPR (the code's natural allocation is 128) → 304–440 B scratch → 22–34 GB
  of write traffic per launch (150–240× the output array); the big MINB=8 tiles spill 432–1272 B
  for the same reason (`accf[MF][NF][8]` = 12·MF·NF of the 64 VGPR budget, leaving too few for
  loop state — spill bytes ≈ 4·(VGPRs needed above 64) − 12, exact across all shapes).
  **Consequences:** (i) **(2,2)@8 is a real, bit-exact −4 %** on schedule B (29.4 vs 29.9 ms;
  nearly spill-free at 44 B); (ii) the request-floor arithmetic says the **target shape is
  (1,4)@8** — 79 M reqs (44 M x + 32/NF·4.28 M weights) + 7.15 GB ⇒ model **13.2 ms, ~3× the
  control** — and it is *register-bound*, not algorithmically dead: 48 accf VGPRs leave 16 for
  loop state where ~48 are needed. The path is a register diet (hoisted single-pointer walks,
  LDS-staged x panel to kill address registers, `xs`-style scale hoisting) to fit (1,4) in the
  64-VGPR/8-blocks budget spill-free; (iii) weight traffic ∝ 32/NF means panel-style L2 blocking
  (the engine's `Panel` mode) is the complement: at NF=4 with the in-flight set kept ≈1.4 MB
  (≤2 MB L2), DRAM falls to x-once + weights-once ≈ 6.7 GB ⇒ 13 ms stays DRAM-legal, and
  92 M reqs ⇒ **~14 ms/launch ≈ 2.1× — the full 300 t/s GEMM budget on this card**. (iv) All
  shapes are bit-identical by construction (same per-block computation as the shipped tile;
  cal57 kernel bodies are verbatim copies), so any shipped change needs only engine-level
  end-to-end verification.
- **Panel battery: L2 blocking works, requests are unchanged — panel is a DRAM lever, not a
  request lever (2026-10-23, `cal58`, panel-mode battery, gate_up shape, cold-L2 timed launches,
  same counters/asm protocol as cal57).** Panel geometry = shipped (`col0 = y·16NF`,
  `row0 = (z·P + x)·16MF`, grid `(P, column blocks, row blocks/P)`; x-fastest scheduling),
  cold best-of-5, with the `.amdhsa` spill identical to cal57 for every shape (panel geometry
  costs zero registers — confirmed a second time):

  | shape (MF,NF,P) | ms | reqs M | DRAM GB | spill B | binding |
  | --- | --- | --- | --- | --- | --- |
  | (1,2,32) engine default | 30.2 | 156 | **1.46** | 36 | requests |
  | (1,2,8) | 28.2 | 182 | 4.2 | 36 | requests (amplified) |
  | (2,2,16) | 26.9 | 253 | 1.9 | 44 | requests (2.9× amplified) |
  | (2,2,8) | 29.7 | 313 | 4.7 | 44 | requests (3.5× amplified) |
  | (1,4,8) | 51.6 | 536 | 26.3 | 432 | spill traffic (17.0 GB wr) |
  | (1,4,16) | 50.6 | 536 | 25.6 | 432 | spill traffic |
  | (2,4,8) | 45.5 | 526 | 21.5 | 492 | spill traffic (10.1 GB wr) |
  | (4,2,8) | 44.1 | 462 | 19.0 | 460 | spill traffic (9.6 GB wr) |

  **Panel mode drives DRAM to the theoretical floor and time does not move.** (1,2,32) reads
  1.46 GB — the weights-once + x-once ideal (5.4× below schedule B's 7.85 GB, 12.2 M L2 misses
  ≈ 1.56 GB at 92 % L2 hit) — and takes 30.2 ms, identical to schedule B's 29.9 ms: **L2 hits do
  not reduce the GL2C request stream** (156 M, the (1,2) geometry floor), which is the binding
  term at 6.7 Greq/s. The cal57 consequence (iii) prediction ("panel NF=4 ⇒ 92 M reqs ⇒ 14 ms")
  is **refuted**: (1,4) panel shapes carry the same 432–492 B spill as their schedule-B twins,
  so the 10–17 GB scratch write stream dominates; the (2,2) in-flight panel overshoots L2 and
  amplifies requests 2.9–3.5×. The warm-context panel gain stays explained by cross-kernel L2
  persistence (cal41 +6–8 %, engine +2.3 %), not by within-launch request cuts.
  **The (2,2) panel failure is an ordering artifact, and it points at the remaining lever:**
  x-fastest raster puts ≈512 concurrent blocks spanning many z-slices in flight, so the
  in-flight x set (P·16MF columns × 16NF·k_cols) exceeds 2 MB L2 for MF=2 (5.2 MB at P=8) —
  requests amplify 2.9–3.5×. Swapping grid.y↔grid.z (**y-fastest panel**: concurrent blocks
  share the *same* 16NF-column x window, in-flight 32 windows × 8.2 KB·16MF ≈ 1.31 MB ≤ L2)
  makes per-launch traffic ideal for (2,2): x-once 5.24 GB + weights-once 1.43 GB + writes 0.2
  ≈ 6.9 GB ⇒ 13.5 ms DRAM-term, and the 88 M request floor ⇒ 13.1 ms — **both terms land at
  ~13–14 ms = 2.3×** if (2,2)@8 is made spill-free (its 44 B × 430 GB/launch spill stream is
  exactly what keeps cal57's (2,2) at 29.4 ms). The critical path is therefore a **register diet
  fitting (2,2)@8 in 64 VGPR** (natural 128: 48 accf + 16 acc leaves ~64 loop-state regs where
  16 are available) — cal59 probes y-fastest geometry + diet variants.
- **Raster A/B is inconclusive** (2026-10-22, `NINFER_T2_A8_RASTER`, W=1024 cold-boot, 3
  reps): default `cols`-fast (x=columns) median **165.2 tok/s** vs `rows`-fast (x=rows)
  **163.7 tok/s** — a ~1% delta. Two 3-run sets do not by themselves establish a noise
  distribution, so this is treated as "no dispatch change warranted," not as proof the block
  raster is inert; the block-raster is not a lever worth further A/B on this card.
- **WMMA compute roofline measured** (2026-10-23, `.qual/wmma/cal42.cu` — pure compute loops,
  independent accumulator sets, median of 7, RX 7600M XT @ 2.33 GHz, 16 CUs):

  | instruction | peak warp-ops/s | peak MAC rate | cycles/op/wave |
  | --- | --- | --- | --- |
  | `wmma_i32_16x16x16_iu8_w32` (shipped) | **4.787 G** | 19.6 T MAC/s (39.2 "TMAC" 2-op) | 16 |
  | `wmma_i32_16x16x16_iu4_w32` | **9.419 G** | 38.6 T MAC/s (77.2) — **exactly 2× iu8** | 8 |
  | `wmma_f32_16x16x16_f16_w32` | 4.796 G | 19.6 T MAC/s — same op rate as iu8 | 16 |

  Control: a pure fma loop issues **75.8 G warp-instr/s ≈ 101%** of the 74.7 G instr/s issue
  capacity (16 CUs × 2 × 2.33 GHz, one wavefront per cycle per scheduler), validating the
  counting model (the earlier `lds` control was invalidated by dead-store elimination and is
  not used). **Consequence:** the shipped G32-scheduled gate_up (25.7 ms, 182.5 G useful MAC,
  ~2× fragment waste ⇒ 14.2 TMAC/s issued) sits at **~37% of the iu8 compute peak** — the GEMM
  is compute-issue-schedule-limited with **~2.7× scheduling headroom**, not DRAM-limited (it
  streams at ~55 GB/s of the 237 GB/s ceiling). The roofline answers the handoff's open
  "what is the WMMA peak" question.
  **iu4 caveat:** the 2×-rate `iu4` builtin takes an **i4 activation fragment**, so it needs
  genuine 4-bit activations. A bit-exact decomposition `x = 16·hi4 + lo4` (two wmmas per tile)
  exactly cancels the 2× rate — net zero. Reaching iu4 speed is a **numerics change**
  (4-bit-activation quantization, perplexity/coherence-gated), not a bit-exact rewire.
- **G32 panel raster integrated into the shipped tile; default ON** (2026-10-23):
  `t2_prefill_wmma1_kernel` gains a `Panel` template parameter (Panel>0 ⇒ grid
  `(P, column blocks, row blocks/P)`, `row0 = (z*P + x)·16`); `t2_a8.cu` launches it when
  `wrow_blocks % 32 == 0` (all engine shapes qualify: gate_up 2176, mlp_down 320, gdn 768
  tiles), **default panel=32**, disabled by `NINFER_T2_A8_WMMAPANEL=0`. GEMM-level (cal41,
  cold-L2, bit-exact): gate_up **25.7 ms vs 27.9**, mlp_down **13.3 vs 15.9**, gdn_valuez
  **9.1 vs 9.7** (+6–8% every shape). Device-asm resources for all six Panel=0/32
  instantiations are **identical to HEAD** — 128 VGPR, 38–43 SGPR, 36 B private segment (the
  pre-existing 9-dword spill), so panel mode costs zero registers. Engine ABBA (cold-boot,
  W=4210, 3 alternating legs in one lock, panel off vs G32): medians **142.6 vs 139.4
  tok/s** (TTFT 29.7 vs 30.4 s), every G32 leg strictly above every off leg — **+2.3%
  prefill** at 4,210 tokens, and the bit-exactness gate (`wmma_bitexact.sh`, 128-token
  greedy lighthouse prompt) gives **TEXT MATCH** on the panel path. Baseline reference at
  this width was ~144 t/s (TTFT 29.2 s); the panel raster recovers ~3 t/s of cold-boot
  prefill and moves the 300 t/s target's remaining gap onto the other levers (attention
  39.5% of prefill at 4096-ctx; iu4 numerics).

## Goal (set 2026-10-22)

Targets for the NInfer 27B stack (T2 ternary, **2.125 bpw** weights) on the **RX 7600M XT**
(gfx1102, 16 CUs, 237.6 GB/s DRAM):

- **Prefill (PP): 300+ t/s — the target stands; the GEMM traffic lever is OPEN (earlier
  "190–200 cap / GEMM tapped out" conclusion WITHDRAWN after a factor-of-4 bandwidth correction).**
  The card demonstrably does 296 t/s here (Prism's 2.13-bpw PQ2_0 service, same ~2.1 bpw as
  NInfer's T2 2.125 bpw). Corrected `cal38` measurement: the NInfer T2 prefill GEMM streams its
  **logical** packed codes at **~20% of the ~240 GB/s DRAM ceiling** (~47 GB/s). Two traffic-
  reduction designs were built and both **lost** (coalesced `int32v` full-line read ~70% slower
  with 320 B scratch spill; weight-stationary LDS-staged `wlds` ~4× slower with 288 B spill),
  and the **cooperative multi-warp LDS loader `wcoop` was also built and loses modestly
  (+3.5–8.2%)** while staying register-clean (zero spill, 8–12 KB LDS). The two big losses were
  spill-driven; `wcoop`'s small loss is barrier/LDS-staging-driven. Weights are already at the
  2-bit floor (sub-2-bit GGML formats have no in-tree kernels, no PQ/VQ path), so bit-rate is off
  the table. **The GEMM traffic/parallelism lever is still open** — these three designs rule out
  *those* implementations (traffic reduction and the LDS-staged loaders), not GEMM optimization
  as a whole; the ~80% headroom to the DRAM ceiling on logical packed-code traffic is real. The
  other major lever is the **mixer/attention time** — with the correction that this bucket
  **includes the mixer projection GEMMs**, so a faster GEMM *can* help it (the earlier "no GEMM
  change touches the 45%" framing was wrong). **Long-prompt bucket profile (measured 2026-10-22,
  `NINFER_PREFILL_PROFILE=1`, 4,210-tok prompt → four 1024-tok prefill chunks, TTFT 29.2 s =
  144 t/s):** every GEMM bucket is *flat* across chunks while the attention path *grows linearly*:
    | chunk (ctx) | gdn_mixer | gdn_mlp | attn_mixer | attn_mlp | total |
    | ~1024 | 1274 | 2140 | 982 | 677 | 5074 |
    | ~2048 | 1275 | 2135 | 2114 | 678 | 6203 |
    | ~3072 | 1282 | 2147 | 3232 | 680 | 7341 |
    | ~4096 | 1284 | 2142 | 4379 | 683 | 8488 |
    | **Σ** | 5115 | 8564 | **10707** | 2718 | **27104** |
  (ms, per chunk). **At 4,096 context the attention path (`attn_mixer`) is 4,379 ms = 52% of that
  chunk and 10,707 ms = 39.5% of the whole 27.1 s prefill** — it is the single largest and only
  growing cost. The GEMM-dominated buckets (gdn_mlp 8564 + attn_mlp 2718 + gdn_mixer's projection
  share) are the flat ~60%. **So 300+ t/s at long context needs BOTH levers, each roughly 2×:**
  the flat GEMM work (~2× → −8 s) *and* the growing attention path (~2× → −5 s) together take
  27.1 s → ~14 s ≈ 300 t/s. Neither lever alone gets there. This is the "real longer-prompt
  attention path" the plan called for profiling — it is now characterized: attention is ~40% of
  long-context prefill and scales linearly with context, so it is a first-class lever alongside
  GEMM, not a residual.
- **Decode (TS): 25–30 t/s.** No-spec Prism baseline on this card is 22.2 t/s; closing the gap
  to 25–30 is the decode-side lever (separate from prefill).

Baseline to beat (NInfer T2, 1,065-tok prompt): **~165 t/s prefill** (the closest same-scale
NInfer record is 142.2 t/s at 4,100 tokens). Prism's 296 t/s (4,069-tok prompt, no cached
tokens) is a real reference on the same card, but the two engines use different timing
definitions, token counts, formats, and paths, so it is a **reference, not a controlled
attribution** of the gap. The gap is closed by lifting the GEMM's ~19% DRAM utilization toward
the ceiling (cutting the 32× weight re-read traffic) plus the mixer/attention time — **not** by
any measured ceiling on the GEMM.

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

## 2026-10-23 iu4 GEMM prototype (cal43–cal54): numerics proven, 2.25–2.76× on the dominant GEMMs — **quality-gated, do NOT ship without the user**

**What.** Prototype GEMM `u4g` (`.qual/wmma/cal43.cu`) runs **4-bit signed activations** through
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` (wave32) with the shipped T2 weight layout (4 codes/byte,
fp16 scale per 128-col group), the shipped G32 panel grid (`(32, T/32, (N/16)/32)`), and the shipped
magic-bits i32 epilogue (seed `0x4B400000` = float `1.5·2^23`; `dot = __int_as_float(v) − kMagic`).
The hardware reads BOTH operands as signed i4, so weights bias-pack code+8 (signed read = u−8; the
semantic (u−1) is restored by a per-group rowsum term `rs` in the epilogue: `dot = dot_hw − rs`).
Fragments are 8 B per lane per side vs 16 B for iu8 — half the fragment bytes and half the load
instructions per WMMA.

**Layout decode chain** (the iu4 fragment layout is undocumented on this ISA — measured, not assumed):
- cal53: single-1×single-1 pairing probes decode it: k-pairing is identity, nibbles PLAIN both sides,
  only rows permute — `rowA(i) = ((i&3)<<2)|(i>>2)` (A side), `rowB(j) = ((j&7)<<1)|(j>>3)` (B side),
  both self-inverse; lane→row `widx(s) = ((s&1)<<3)|(s>>1)` (its inverse is `rowB`).
- cal51: dense 16×16 end-to-end model PASS on that decode.
- cal52: explains cal51's earlier dense failure and the cal50/48 "nibble-perm" artifacts (spread vs
  interleaved 8-byte pack orders — shifting B nibbles within a word is invisible).
- cal54: kernel-shaped exhaustive routing sweep (real kernel shape MF=1/NF=2/P=32, every candidate
  mapping a runtime knob; 416 combos, dataset engineered so ONLY routing matters): the canonical
  PASS `role=0 am=widx bm=id ship sw=0 st=0` is exactly cal43's loader mapping. Layout proven twice
  over; cal43's initial battery FAILs were numerics in the reference, not routing (see below).

**Battery (numerics).** cal43 vs a double-precision CPU reference **on the u4 grid**: maxRel
**6.5e-4 / 6.6e-4** (gate 1e-3) on shapes 512×5120×1024 and 512×17408×1024. The kernel was correct
on first try with the cal54-validated mapping; all three battery failures were harness artifacts:
(1) reference used the i8 engine grid (`xc = 16·c`) instead of decoding signed c from x4 (16× scale
mismatch, maxRel 6210 → 1); (2) reference compared tokens the launch deliberately masks (`t ≥ T−3`,
kernel skips storing them); (3) battery shape N=768 violates the G32 guard `N % 512 == 0` and
silently drops rows ≥512 (the same guard proven on the engine path). The gate denominator is
`max(1e-3, |ref|, 1e-4·Σ|group terms|)` to absorb fp32 fmaf-chain cancellation noise at near-zero
cells (136-group sums of O(100) terms); the worst |ref|>10 cell shows rel ~1e-5, ratio 1.0.

**Timing** (RX 7600M XT, lock verified, ABBA, median of 11, 96 MB L2 flush between launches):

| shape (N×K×T) | shipped i8 (µs) | u4g (µs) | speedup |
|---|---|---|---|
| gate_up 34816×1024×5120 | 29675.8 | 10766.2 | **2.76×** |
| mlp_down 5120×1024×17408 | 15455.6 | 6855.7 | **2.25×** |
| gdn_valuez 12288×1024×5120 | 10427.2 | 4075.3 | **2.56×** |

cal42 roofline: iu4 peak = 9.419 G warp-ops/s = **exactly 2.00×** the iu8 peak (4.787 G) at 2.33 GHz —
hardware gives the 2× for free at 4-bit. u4g hits 4.13 G warp-ops/s on gate_up = **44% of iu4 peak**
(shipped i8: 37% of iu8 peak): the ceiling converts into real GEMM time, not just paper headroom.

**Resources** (same TU, `-O3`, gfx1102, `llvm-readelf -n` on the device object; full notes in
`.qual/wmma/cal43_resources.txt`): u4g **106 VGPR / 41 SGPR, 0 spills, 0 private, 0 LDS**, 753 asm
lines, 14 b128 loads — vs shipped i8 128 VGPR / 41 SGPR / **8 spills** / 36 B private / 1 KB LDS,
1244 lines, 19 b128 loads. `__launch_bounds__(32,8)` fully satisfied with headroom (granularity 112).

**Gate.** The kernel is exact *for its grid* (battery proves it), but it quantizes activations to 16
levels (i4) where the i8 engine grid has 256: this is a **model-quality decision, not a bit-exact
optimization**. Before any adoption: end-to-end quality A/B (perplexity + coherent text on production
prompts) with the real model, and the user's explicit decision. If accepted, the GEMM-dominated
prefill buckets gain 2.2–2.8×; combined with the attention lever (39.5% of 4096-ctx prefill, §3
split) the 142 t/s baseline moves into 300+ t/s range at long context — GEMM alone is not enough.

**Integration path** (only after the user decides): i4 quantize + per-group rowsum kernels producing
`x4`/`xs`/`rs`, wire `u4g` in behind an env gate, engine shapes already satisfy the G32 guard
(34816/5120/12288 all N%512==0), then the full quality suite.

## 2026-10-23 qwen4exp (Qwen3.8-Flash-Next) test pass on gfx1102 — full runnable surface green

The handoff's lazy "pre-existing failures" claim does not cover the qwen4exp suite here: everything
that can run on this machine (no Flash-Next artifact on disk) **passes, on `hip-port`, first try**.

- **Build:** all 13 `ninfer_qwen4_exp_*` targets compile clean in `build-hip-test` (including the
  engine/vision/MTP real tests — the `qwen4_exp_instance.cpp` engine path is in `build-hip` too).
- **Host (cpu-labelled) 6/6 PASS** (`ctest -R 'ninfer_qwen4_exp_(ngram_hash|ngram_volume|
  rope_positions|expert_cache|memory_plan|context_cache)_test'`, 2.3 s total): ngram hash fixture,
  ngram volume reader, three-axis RoPE positions, expert cache LFRU/frames/controller, memory plans,
  context-cache host tier + checkpoint images.
- **GPU (lock held, single test): `ninfer_qwen4_exp_nvme_tier_test` PASS** (0.5 s) — NVMe tier
  frames, write-through/eviction, direct captures, host I/O, failing disk, no frames.
- **Python tool tests 18/18 PASS** (`tests/flash_next/test_tools.py`; no system pytest, so
  `uv run --with pytest --with numpy --with torch==2.6.0+cpu` against the CPU wheel index):
  `strata_compare.engine_rows` header validation, `compare` input refusal cases,
  `reference.build_parser` boundary flags.
- **Real-artifact tests (7) skip by design (rc 77):** forward/mtp/vision/vision_engine/engine/
  lanes require `NINFER_TEST_QWEN4EXP_ARTIFACT`. This machine has no artifact and no source GGUF:
  the model card manifest is `qwen4exp-iq2xs-mtp-vision.ninfer` 32.0 GB + 29.1 GB IQ4_NL n-gram
  volume, from the 68.0 GB two-part `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` GGUF. Producing a local
  artifact is a ~70 GB download + a ~15 min conversion — a user-side decision (disk, network),
  not a code gap. Free disk here is 290 GB.

## 2026-10-23 cal59 (y-fastest panel + register-diet battery): the diet works, the request floor is dead

cal58 ended with a critical path: y-fastest panel geometry (swap grid.y↔grid.z so the in-flight
working set is `16NF`-column windows ≈ 1.31 MB ≤ 2 MB L2, killing the (2,2) in-flight overshoot
that amplified requests 2.9–3.5×) **plus** a spill-free (2,2)@8 (44 B scratch removed). Predicted
per-launch ideal: x-once 5.24 GB + weights-once 1.43 GB + writes 0.2 GB ⇒ 13.5 ms DRAM-term;
88 M request floor ⇒ 13.1 ms ⇒ ~13–14 ms = 2.3×. Three kernel families, shipped body verbatim
(bit-exact by construction), all `__launch_bounds__(32,8)`, gate_up 34816×5120×1024, cold-L2
best-of-5 with 96 MB flush, rocprofv3 two-pass GL2C (protocol identical to cal57/58):
`Y` = y-fastest geometry, `D` = y-fastest + walked pointers (walk `pA/pS/pB` per group instead of
recomputing `gBase`-indexed addresses), `L` = y-fastest + LDS-staged x panel (4 KB NF=2 / 8 KB
NF=4, static shared, copy 128 B/row per group, fragment reads from LDS).

| shape | VGPR/spill | best ms | DRAM GB | reqs M | wr MB | binding term |
|---|---|---|---|---|---|---|
| Y22_p8 (2,2)P8 y-fastest, original loads | 128/44 | 30.43 | 12.0 | 210 | 187 | reqs (model 31.8) |
| Y22_p4 (P=4) | 128/44 | 27.95 | 13.7 | 202 | 187 | reqs (model 30.6) |
| Y14_p8 (1,4) | 128/432 | 52.24 | 25.4 | 524 | **16 969** | spill-writes (5.4× DRAM) |
| Y12_p8 (1,2) | 128/36 | 28.45 | 5.5 | 180 | 214 | reqs (model 27.3) |
| **D22_p8 (2,2)P8 spill-free diet** | **127/0** | **27.99** | 9.6 | 190 | 142.5 | reqs (model 28.9) |
| L22_p8 LDS-staged x | 128/80 | 158.04 | 8.3 | 404 | 258 | LDS reads (measured ≫ model) |
| L24_p8 (2,4) LDS | 241/16 | 86.49 | 1.38 | 215 | 151 | LDS-bound (measured 30× model) |
| L22_p32 (P=32) | 128/80 | 157.99 | 6.1 | 393 | 249 | reqs+LDS |

- **The register diet works exactly as designed.** Walking the pointers (`D22_p8`) collapses the
  (2,2)@8 footprint to **127 VGPR, 0 spill** (Y22_p8: 128 VGPR, 44 B), cuts scratch writes
  187→142 MB, DRAM 12.0→9.6 GB, requests 210→190 M — and time 30.4→28.0 ms. The diet is
  bit-exact and needs no engine change (same body, reordered address arithmetic).
- **The 88 M request floor for (2,2) is dead.** Best achieved on this geometry: 190 M, and
  190/6.6 Greq/s = 28.9 ms ≈ measured 28.0. The request stream — not DRAM (9.6 GB ⇒ 18.8 ms),
  not writes (142 MB ⇒ 0.6 ms) — binds every shape in this family. y-fastest geometry + zero
  spill together buy −8% vs the 29.9 ms control, nowhere near 2.3×.
- **(1,4) is confirmed dead at this occupancy:** 432 B scratch = 17 GB of writes per launch
  (120× the output bytes). NF=4 requires a register budget ≫ 128 VGPR (L24_p8 at 241 VGPR
  spills only 16 B and lands at 4 blocks/SIMD — 86 ms).
- **LDS staging is net-negative on this chip and explains cal58's "hits don't reduce
  requests".** The global→LDS copies are perfectly coalesced (L22_p32: 6.12 GB DRAM — the lowest
  ever measured on this shape, 1.13 GB weights + x-once), **but LDS reads enter the GL2C request
  stream**: hits jump 115→337 M and total requests to 393–404 M ⇒ 158 ms, 5.6× slower than the
  global-load twin. wmma B-fragments are 16 B/lane; every fragment is a new request. This
  confirms requests are counted at issue, per-access, with no coalescing credit for LDS.
- **Consequence.** Time on gfx1102 = issued requests / ~6.6–6.8 Greq/s, and the shipped wmma
  fragment pattern (16 B/lane x-loads from rows `k_cols` apart) issues ~180–210 M requests on
  gate_up no matter the geometry (panel, y-fastest, spill-free all land 27.5–30.5 ms). 13–14 ms
  needs ≤ ~90 M requests ⇒ x loaded at full-sector (≥32 B/lane) granularity with fragments
  fed from registers/LDS **without new per-fragment requests** — i.e. a fundamentally different
  fragment supply (e.g. one wide load amortized across 8 `s`-steps, 32 B/lane `int8v` loads),
  not more L2 blocking. Next probe must attack request granularity, not locality.

Diet note for any future integration: `D22`-style walked pointers alone are a **bit-exact
−8% (27.99 vs 29.90 control; vs the 30.4 ms y-fastest twin)** on the (2,2)@8 shape — strictly
below the shipped G32 panel's 30.2 ms cold, so not a ship candidate by itself; it is the
proof that the spill law is beatable at 128 VGPR. Files: `.qual/wmma/cal59.cu`,
`roc59/` counter CSVs, `parse59.py`.

## 2026-10-23 cal60 (wide-load granularity battery): requests are SECTOR-bound — granularity is not the lever

cal59's consequence said the 13–14 ms target needs ≤ ~90 M requests and suggested loading x at
full-sector (32 B/lane `int8v`) granularity, amortized across the eight `s`-steps. cal60 built
exactly that: same shipped body semantics, one 32 B/lane `int8v` load feeding two `s`-steps
(`av[n][0/1]` = the load's two halves — byte-identical per-lane to the engine's two 16 B loads,
so pairing is preserved), sweeping MF/NF at panel 8. Result: **wide loads halve the instruction
count, not the request count.** Requests ≈ 190 → 185 M and time did not move:

| shape | best/avg ms | note |
|---|---|---|
| X42_p8 (4,2) | 33.92 / 34.48 | vs D22_p8 control 27.99 — **worse** |
| X41_p8 (4,1) | 46.20 / 46.89 | |
| X81_p8 (8,1) | 61.82 / 62.31 | |
| X82_p8 (8,2) | 61.05 / 61.53 | |
| X44_p8 (4,4) | 49.30 / 49.46 | |

Sector arithmetic fit: R ≈ 356.5/MF + 89.1/NF M reqs, measured ≈ 0.83 × the sum. The request
stream charges per **sector touched**, not per instruction: widening the load changes instruction
mix, not sectors. Granularity is not the lever — **layout** (who touches which sectors, and how
often) is. Files: `.qual/wmma/cal60.cu`, `roc62-stdout.txt`.

## 2026-10-23 cal61 (launch_bounds VGPR cap): at occupancy 8, MF>2 payloads spill — and spill is geometric

`__launch_bounds__(32,8)` caps the allocator at 128 VGPR. Everything above the (2,·) payload
lands in scratch, and the cal60 payload sizes turn time into a spill tax table:

| shape | scratch B | time ms |
|---|---|---|
| (4,2) | 552 | 46 |
| (8,1) | 660 | 78 |
| (8,2) | 1760 | 113 |

(The cal60 X-shapes above are exactly these spills: X42 = (4,2)+wide loads 34 ms beats the
naked-spill 46; X81/X82 ≈ 61 beat 78/113 — wide loads soften spill traffic, they don't beat the
(2,·) class.) Conclusion: at occupancy 8 the register wall is (MF≤2, NF≤2). Files:
`.qual/wmma/cal61.cu` (device-asm `.amdhsa_private_segment_fixed_size` records).

## 2026-10-23 cal62 (occupancy 4, spill-free high MF): the issue rate halves — occupancy is not negotiable

The escape hatch from cal61: drop `__launch_bounds__` to (32,4), let MF=4/8 live spill-free
(233–251 VGPR, 0 scratch), keep the wide loads. It removed every scratch byte and **lost**:

| shape | best/avg ms (per pass) |
|---|---|
| Z22_p8 (2,2) | 28.47 / 29.47, 28.42 / 29.08 |
| Z22_p4 (P=4) | 27.22 / 28.80, 26.55 / 28.23 |
| Z42_p8 (4,2) | 47.52 / 47.96 |
| Z24_p8 (2,4) | 41.91 / 42.08 |
| Z44_p4 (4,4) | 94.30 / 94.89 |

Z42_p8 is spill-free at 47 ms vs the spilled (4,2) at 46 — spill-free buys nothing when the
in-flight wave count drops 8→4 per SIMD: the measured request issue rate collapses ~6.6 →
~3.2 Greq/s. **Occupancy is not negotiable**; the request stream needs 8 waves/SIMD to saturate.
Files: `.qual/wmma/cal62.cu`, `roc63-stdout.txt`, `roc63/` CSVs.

## 2026-10-23 cal63 (two-warp row-split blocks): null — the warps read the identical x

Premise: if each warp of a 64-thread block owns a **row** tile (row-split), the x panel is read
once per block and R ≈ 356.5/(2·MFw) + 89.1/NFw. Measured: **requests do not fall** — the two
warps read the identical x window (same tokens, same rows), so the sectors touched are the
same; splitting rows only halves weight-side traffic, which was never the binding term:

| shape | best/avg ms (per pass) |
|---|---|
| W22_p8 | 29.38 / 29.50, 29.19 / 29.41 |
| W22_p4 | 28.36 / 28.55, 28.05 / 28.31 |
| W12_p8 | 28.76 / 29.12, 28.27 / 29.01 |
| W14_p8 | 54.67 / 54.82, 54.67 / 54.76 |

Row-split lands within noise of the D22 control (27.99). Files: `.qual/wmma/cal63.cu`,
`roc60-stdout.txt` (W-family lines).

## 2026-10-23 cal64 (column-split blocks): apparent 23 ms win — VOID, coverage audit

The symmetric idea: 64-thread blocks whose warps split **columns** (`col0 = blockIdx.y·16NF·2 +
warp·16NF`), sharing one row panel and the epilogue x-scale reads. The recorded sweep looked like
the breakthrough cal59 asked for — U21_p16 22.82, U41_p8 23.61, U21_p8 25.75, U21_p4 24.64.
It was **measuring less than half the outputs**. The launches used tile stride `16·MF·2` across
panels, but a column-split block covers `16·MF` rows, so rows past panel-max were never
computed: coverage p8 22.8%, p4 27.2%, U41_p8 33.1%, **p16 50.4%**. All cal64 U timings are
VOID (best case 50% of the true row work). Files: `.qual/wmma/cal64.cu`, `roc64/` CSVs.
(The direction survived: cal64b below is the corrected, certified measurement.)

## 2026-10-23 cal64b (verify gate = engine kernel on identical buffers; column-split CERTIFIED, −25%)

Two things had to be learned before any column-split shape could be trusted.

**1. The verification gate is the engine kernel, not a scalar reference.** The first verify
attempt (scalar identity-pairing reference, unsigned `(by>>2j)&3` codes) mismatched
35,651,583/35,651,584 — and the investigation showed why: T2 activations are stored in
**quantiser-permuted order** (`t2_prefill_i8.cuh` contract, the permutation compensating the wmma
fragment byte order). Random synthetic x is unpermuted, so an identity-pairing scalar reference
diverges from **both** the engine kernel and the probe by design. The ship gate is
**probe kernel vs shipped `t2_prefill_wmma1_kernel` on identical buffers** — the pairing cancels,
isolating the geometry change. (Scalar reference kept as a diagnostic only.)

**2. Row-coverage law (pinned by two verify runs):** a column-split block covers **16·MF rows,
not 16·MF·2**. Correct tiling: `row0 = (blockIdx.z·gridDim.x + blockIdx.x)·16·MF`, grid.x =
`N/(16·MF·P)`. The intermediate form (stride gridDim.x, tile 16·MF·2) computed every tile's
first 32 rows only — 50% of outputs never written (verify: 17,825,792/35,651,584 mismatch, first
at row 32/64, sentinel 0x7F).

Certified result (gate_up 34816×5120×1024, probe `cal64b`):

- **ALL FIVE column-split shapes are BIT-EXACT vs the shipped engine kernel — 35,651,584 /
  35,651,584, both passes.**
- Engine ground truth in the same probe: **30.7–30.9 ms** (1,2,G32 panel, shipped routing).

| shape | best/avg ms (per pass) | VGPR/spill | 128B RD | MISS | HIT |
|---|---|---|---|---|---|
| **V21_p32 (2,1) col-split P=32** | **22.99 / 23.18, 23.01 / 23.14** | 86 / 0 | 17.4M | 24.0M | 974.7M |
| V21_p16 | 23.85 / 24.13, 23.84 / 24.05 | 86 / 0 | 41.2M | 49.4M | 811.0M |
| V21_p8 | 24.91 / 25.15, 24.64 / 24.96 | 86 / 0 | 99.9M | 107.8M | 674.3M |
| V41_p16 | 20.99 / 23.81, 21.92 / 23.81 | 128 / 112 B | 225.5M | 241.2M | 673.5M (59.6M wr) |
| V41_p8 | 23.14 / 24.86, 20.95 / 24.23 | 128 / 112 B | 353.1M | 359.7M | 503.9M (59.5M wr) |

- **Winner: column-split (2,1) at Panel=32 — 23.14 ms avg full coverage: −25% vs the engine
  ground truth, −17% vs the best control ever measured (D22_p8 27.99).** First probe shape
  below the 27.5–30.5 ms band that every row-split geometry landed in through cal59.
- The panel law is monotone for the spill-free V21 family: P=8 → 16 → 32 = 24.96 → 24.05 →
  23.14 ms; misses (≈ DRAM) fall 107.8 → 49.4 → 24.0M (P=32 ≈ 3.1 GB ≈ 134 GB/s at 23 ms —
  headroom under the card's ~288 GB/s, so requests, not DRAM, still bind).
- Column-split beats row-split (cal63) because the two warps read **disjoint** token windows
  (columns = tokens), halving per-block x sectors, and the y-fastest panel (P=32) keeps that
  window hot — 975M hits against 24M misses.
- V41 (MF=4) shapes are noisy and carry a 112 B spill (59.6M wr64 + 63M MC_WRREQ of scratch
  traffic) — same cal61 wall, do not pursue.
- Device asm (V21): 86 VGPR, SGPR 32, LDS 4096 B, private segment **0** — occupancy 8, the cal61
  and cal62 laws both satisfied simultaneously. (Dumped pre-geometry-fix; the fix is index-math
  only — re-dump to confirm is open.)

**Integration path (next step, bit-exact gate PASSED in probe):** column-split is a new kernel
variant — the engine's `t2_prefill_wmma.cuh` routing has only row-split panel grids; it needs a
column-split grid (`grid = (N/(16·MF·P), T/(16·NF·2), P)`, tile stride 16·MF) behind an env
gate, then the engine-level end-to-end A/B at the 4,210-token prompt (baseline 144–148 t/s,
TTFT 29.2 s) with coherent-text check under the GPU lock. Probe files: `.qual/wmma/cal64b.cu`,
`run65.sh`, `roc65/` CSVs, `roc65-stdout.txt`.

## 2026-10-23 cal65 + cal66 (column-split integrated and CLOSED: the probe win was a coverage artifact)

The cal64b integration path, executed end to end. Kernel `t2_prefill_wmma2_kernel<MF,NF,Panel,
Epilogue>` (64 threads, `__launch_bounds__(64,8)`, two warps split token columns, 32 B/lane wide
x loads byte-identical per lane to wmma1's two 16 B loads) ported into `t2_prefill_wmma.cuh`;
engine grid `= (N/(32·P), T/32, P)`, `row0 = (z·gridDim.x + x)·16·MF`, launched only when
`N % (32·P) == 0` and `T % 32 == 0`, behind `NINFER_T2_A8_COLSPLIT` (default **off** — shipped
G32 panel raster unchanged). Device asm post-port: all P∈{8,16,32} instantiations 86 VGPR,
private 0, LDS 4096 — identical to probe V21, cal61/cal62 laws satisfied.

Engine ABBA at the 4,210-token prompt (`run66.sh`, cold boot per leg, one lock hold):

| leg | prefill tok/s | TTFT |
|---|---|---|
| off (A) | 144.9, 129.6, 142.4 (median 142.4) | 29.2, 32.7, 29.7 s |
| cs32 (B) | 104.9, 104.2, 100.2 (median 104.2) | 40.4, 40.6, 42.2 s |

Every B leg below every A leg: **−27% median**. Greedy 128-token text gate: **TEXT MATCH** —
the port is bit-exact end to end; the loss is pure geometry.

**Operating point ruled out (run67).** `NINFER_PREFILL_PROFILE` legs (one per arm, cold boot,
W=4210) show the engine **chunks the prompt at 1024 tokens** (four `tokens=1024` chunks, then
129/5/2) — the prefill GEMM runs at **T=1024, exactly the probe's certified T**. The T=4256
whole-prompt hypothesis is dead. The regression is entirely the T2 prefill tile GEMMs, per
1024-token chunk (off → cs32): gdn_mlp 2075 → 3470 ms (+67%), attn_mlp 656 → 1129 (+72%),
fine proj+conv 13.2 → 22.8 ms/layer (+72%), fine out 4.91 → 8.36 (+70%); non-GEMM GDN segments
(nc/rec/norm) untouched; panel legs flat (cs8 ≈ cs16 ≈ cs32 ≈ 104 tok/s).

**cal66 closes the case: cal64b's timing grid computed HALF the rows.** cal64b's timing pass
carried over cal63/64's "overlap trick" — `grid.x = N/(16·MF·2·P)`, an extra `/2` that in those
kernels halved *column* work in exchange for `t_pad = T−3`. For the column-split kernel the
channel doesn't exist (NF=1: `grid.y = T/32` already covers all columns), so the extra `/2`
simply skipped half the rows: gate_up timing ran 17,408 of 34,816 rows. Its 23.14 ms is
**half-coverage**; full coverage ≈ 46 ms. cal66 re-times with `grid.x = N/(16·MF·P)` (identical
to the engine csgrid), all three engine prefill shapes at T=1024, 96 MB L2 flush before EVERY
timed rep, same epilogue both arms, verify = probe kernel vs engine kernel on identical buffers
(all three shapes **BIT-EXACT**: 35,651,584, 5,242,880, 5,242,880 — the cal64b geometry law
holds; only its timing grid was wrong):

| shape (T=1024) | engine (1,2,G32) | cs (2,1) P=32 | P=16 | P=8 |
|---|---|---|---|---|
| gate_up 34816×5120 | 28.77 ms | 45.95 (+60%) | 46.58 | 46.63 |
| down 5120×17408 | 14.61 ms | 24.08 (+65%) | 23.83 | 23.81 |
| o_proj 5120×5120 | 4.19 ms | 6.94 (+66%) | 6.85 | 6.86 |
| seq gate_up→down→o_proj | 45.81 ms | 77.58 (+69%) | — | — |

Full-coverage probe reproduces the engine's +67–72% to within a few percent, at every panel
(flat — matching the engine), and 23.14 × 2 = 46.3 ms lands on the measured 45.95. **Mechanism
of the loss:** column-split halves x traffic (16-token windows per warp) and keeps warp count
identical, so each row of weights is read by 2 warps instead of 1 — **weight traffic doubles**,
and at 2.125 bpw weights dominate (gate_up weights 44.6 MB vs x 5.2 MB; 27B streams ~7 GB per
pass — L2's 2 MB cannot absorb it). cal64's counter story ("halved x sectors, 975M hits") was
counting requests on a half-work kernel; the DRAM ledger it was supposed to win on is the
weights.

**Laws established (cal65/66):**
1. **The coverage law applies to the TIMING grid, not just to verify.** Verify grid, timing grid,
   and engine grid must be identical (`grid.x = N/(16·MF·P)`); the cal63/64 overlap trick is
   specific to row-split column overlap and voids any timing on column-split kernels. Any probe
   win measured with a grid mismatch is void — re-certify before believing it.
2. The engine chunks prompts at **1024 tokens**; probe T=1024 IS the engine operating point.
3. Column geometry cannot win T2 prefill on this card: the traffic budget is the weights
   (2.125 bpw), and any split that duplicates weight reads per warp loses more than it saves on
   x. The shipped (1,2) G32 raster — 16 rows × 32 cols per warp, weights read once per column
   tile — stays the winner.

**Ship decision (final):** gate stays default-off; shipped raster (median 142.6 cold-boot)
unchanged. Column-split code stays in-tree behind `NINFER_T2_A8_COLSPLIT` as the
bit-exact-certified geometry with this verdict documented; cal66 closes the column-split lane.
Files: `.qual/wmma/cal66.cu`, `/tmp/cal66-out.txt`, `run66.sh`, `run67.sh`,
`/tmp/cal65-abba.tsv`, `/tmp/cal65-profile.tsv`, engine asm dump.

## 2026-10-23 cal67 (iu4 wmma on gfx1102: certified semantics, measured 1.72× i8 rate — the last hardware lever is the user-gated 4-bit-activation lane)

cal66 closed the bit-exact GEMM lane by **saturation**: engine-faithful times give 12.7/12.5/12.8
TMAC/s on gate_up/down/o_proj. cal67's register-fed rate probe (no loads, 16 chained wmma/group,
400 groups, 112 blocks, ISA-certified 16 `v_wmma` in each loop body, 28672/28672 nonzero output
checksums) measures the **pure hardware issue peak: dense iu8 = 12.95 TMAC/s** — the engine runs at
**98% of it**. The shipped T2 prefill GEMM is at its hardware floor; no bit-exact geometry, tile,
raster, or scheduling lever remains above ~143 t/s at this bpw.

**The one lever above that floor is iu4 wmma — and it is a user-gated numerics decision, not a
free win.**

1. **Hardware fact:** clang for gfx1102 accepts `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` and
   legally emits `v_wmma_i32_16x16x16_iu4` (ISA verified). Frag sizes: a/b = 8 B/lane (2 ints =
   16 nibbles), acc = 8 ints. **No mixed i8×iu4 wmma builtins exist** (probed `i8_iu4`, `i8u_i4u`,
   `iu4_iu8` — absent), so an iu4 path needs BOTH operands 4-bit: **4-bit activations (iu4 x)** —
   a quality-gated numerics change, NOT bit-exact, requires the user's decision, not shippable
   without it.
2. **Semantics certified (cal67 leg A): 0/256 mismatches.** One T2 code word (4 B = sixteen 2-bit
   two's-complement codes) expands nibble-true into 8 bytes via `se4(c) = c>=2 ? c-4 : c`,
   byte j = `se4(code 2j) | se4(code 2j+1)<<4`; the w32 layout law
   `C(l,w) = dot(Awin[(l<16?0:17)+2w], Bwin[l])` holds for iu4 exactly as for iu8. Poison slots
   (odd slots + slot 16 = all-0xF) prove the instruction never reads them. The fragment groundwork
   for any future iu4 kernel is certified bit-exact.
3. **Measured rate (leg B, ABBA bursts, warm clocks, medians of filled reps):**
   u8 0.2266 ms = **12.95 TMAC/s**; u4 0.1321 ms = **22.22 TMAC/s**; **ratio 1.716×**. The RDNA3
   nominal 2× for iu4-vs-dense-iu8 measures **1.72× on this card** — real, reproducible (rep
   spread <0.5%), and less than nominal.
4. **Projection for the full iu4 path** (nibble-packed weights 2.125→~1.06 bpw + 4-bit x): GEMM
   MACs per token halve at a 1.72× rate → GEMM time × 0.86; GEMM is ~55% of a prefill chunk →
   **end-to-end ≈ +8–9% (~155 t/s)** from the GEMM side, plus weight-traffic headroom for the
   non-GEMM segments. **iu4 wmma alone does not reach 300 t/s** — the Prism-296 reference implies
   a full 4-bit pipeline (activations, KV, attention all 4-bit), a far larger quality-gated change.

**Probe-metrology laws (each bit cost a run):**
- Constant-fragment wmma loops get **collapsed by LLVM** (invariant frags + per-group acc re-init ⇒
  whole-timed-loop DCE). The fold must CONSUME every wmma's acc.
- LLVM **merges ILP2 chains fed identical fragments** (8 wmmas survived, not 16). Chain 1 must
  consume a permuted b frag.
- Compiler unrolls the group loop ×8 while preserving semantics — **count wmma from the ISA**, not
  the source; `#pragma unroll 1` makes the count exact.
- Sub-ms probes run at **idle clocks**; warm ~2 s of continuous GPU work and time bursts ≥10 ms.
- `hipEventElapsedTime` reads **0.0000 for burst windows on this runtime** — use host wall clock
  around burst+sync, and gate every arm on a **nonzero output checksum** before believing a time.
  (The first "u4 = 0.0000/inf" readings were host bugs: an unfilled median slot and event
  quantization — the u4 leg was running the whole time.)

**Gate (unchanged):** iu4 stays the user's quality decision. This probe gives it a measured
hardware basis: 1.72×, certified fragment semantics, ~+9% GEMM-side ceiling at 4-bit. Files:
`.qual/wmma/cal67.cu`, `/tmp/cal67/cal67`, device asm in project root (`.amdhsa_next_free_vgpr`
72/60, no spills).

## 2026-10-23 cal68 (GDN wmma recurrence: two block barriers shrink to warp scope, −5.15% rec kernel; committed 75eb1b3d — plus the #129 failure record: pre-existing, attributed, characterized)

**Barrier shrink (shipped, bit-exact).** In `chunk_recurrence_wmma_kernel<64>` warp w owns the
stripe at colbase = w*8. The snapshot writes (`store_chunk8` → `sm.snapshot`) feed only this
warp's site-2 reads (`site2_pred_history`), and the `store_residual_wmma` writes feed only this
warp's `dot_square` — every producer/consumer pair is stripe-local, so the two block-wide
`__syncthreads()` shrink to `__syncwarp()`. Only the cp.async stage barriers stay block-wide
(cross-warp stage visibility at loop head and stage reuse at chunk tail).

- cal68 probe at engine geometry (qwen-27b H=16/48, DV=64, 96 blocks, T=1024): **output and state
  hashes identical**, rec kernel **1.593 → 1.511 ms (−5.15%)**.
- Engine ABBA (RX 7600M XT, 4,232-token prompt, cold boot per leg, 3 reps base/new interleaved):
  prefill medians **143.0 vs 142.8 t/s** — the expected end-to-end impact is
  −0.082 ms × 48 layers ≈ −3.9 ms per 1024-token chunk ≈ **−0.07% of TTFT**, far below per-leg
  noise (~±2%), so ABBA's role was certification, not detection. Greedy 64-token completions
  (temp 0) are **byte-identical** (sha 0f42d7d4, 283 chars, coherent text).
- `ninfer_gated_delta_net_test`: 57-case log **byte-identical** between baseline and changed
  binaries (see failure record below).
- Committed as 75eb1b3d. Files: `.qual/gdn/cal68.cu`, `build_cal68.sh`, `ab68.sh`, `/tmp/cal68/`.

**launch_bounds refuted.** `__launch_bounds__(256, 1)` → `minBlocksPerMultiprocessor=2` is
inert here: device asm is **identical** across variants (256 VGPR, 120 B privseg, 38 scratch).
The binding constraint is LDS: `RecurrenceShared` stages + snapshot + delta = **60.6 KB**, so
occupancy is 1 block/CU regardless of the bounds hint (64 KB/CU is the LDS limit; 2 blocks
would need ≤32 KB). No register headroom to win — the kernel already uses exactly 256.

**Measured tile costs (fills the "not measured latency" gap in the value_tile heuristic):** at
engine recurrence geometry, wmma DV=64 = 1.59 ms; the SIMT path at DV=32 = **25.86 ms** and at
DV=16 = **24.74 ms** — ~16× the wmma cost. Any heuristic that picks a smaller DV for the
recurrence to save workspace trades 1.5 ms for ~25 ms per layer-chunk; the wmma-64 path is the
only viable tile. **Numerics caveat:** the wmma path is not bit-identical to the SIMT emulation
(deterministic 1–3 ULP TF32 instruction noise plus TF32→BF16 on the prediction dot) — gate
`NINFER_GDN_REC_WMMANATIVE` is default ON; SIMT stays as the `off` reference.

**#129 failure record (pre-existing on d5214b7e — attributed, NOT caused by cal68).**
`ninfer_gated_delta_net_test` fails 18 of 57 cases on this card. Attribution: the test binary
built from clean stashed HEAD (`/tmp/cal68/ninfer_tests.base`) and the binary with the barrier
change (`ninfer_tests.new`) produce **byte-identical logs** (empty diff, 57 GDN_CASE lines each,
identical GDN_PROFILE tail). The two failure clusters:

1. **Catastrophic** — `out` rel_l2 ~1e35–1e36 (garbage, not noise), on **every shape with
   `tokens % 16 == 1`**: qwen-27b 16/48 T=1025 (normalized + raw), qwen-35b-a3b 16/32 T=1025,
   kimi-k3 96/96 T=1025, glm5.3-flash 64/64 T=1025, 35b weak-decay and zero-decay T=4097,
   generic-grouped-map 5/15 T=65 (both profiles), unit-beta 1/3 T=257, raw prefill boundary
   5/15 T=17, raw uneven continuation 5/15 T=1025. The `state` checks stay mild in the same
   cases — the delta chain feeding the state is fine; the garbage is confined to the
   **output/history path**. Suppression pattern supports this: strong-decay T=65 cases are only
   mild (decay coefficients `exp(g_last−g_k)` suppress intra-chunk garbage rows) while
   ordinary/weak/zero-decay cases explode — the garbage multiplies through intra-chunk history.
   Leading hypothesis: **tail-chunk padding** — for the last chunk (`length = min(16,
   tokens − start)` = 1), prepare.cu writes a full `ControlChunk`/`QkChunk` workspace chunk and
   `load_stage` stages it verbatim via unconditional `cp_async<16,cg>`; rows 1..15 of the tail
   chunk are suspect. Next probe (cal69): smallest catastrophic case H=5/15 T=17 raw, dump the
   tail chunk's Qk/Control bytes and the out row for token 16 against reference.
2. **Mild** — state rel_l2 0.0030–0.0037 vs the 0.003 limit at prefix=1025 (≤1.23× limit;
   TF32 accumulation margin at the 64-chunk prefix) plus the raw prefill-boundary T=16 gross
   ratio 1.15. These are accumulation-margin overshoots, not garbage.

**Docs staleness:** line 252 of this file records `ninfer_gated_delta_net_test` PASS at wmma-ship
time (2026-10-22) — that was on the reference (3090) machine; the T≡1 mod 16 cluster is a
regression and/or gfx1102-specific on this card. Treat the PASS claim as stale for gfx1102.
The handoff's "pre-existing failures" framing is confirmed here **by evidence** (baseline-binary
diff), not assumed — per the standing "verify, don't trust the handoff" rule.

## 2026-10-23 GDN correction and probe review

On gfx1102, the native-WMMA GDN recurrence now keeps the output accumulator live across
its conversion and uses BF16 high-plus-residual products for the FP32 control matrices,
residuals, and scaled state update. The existing production GDN test passes all 57 cases
against its independent FP64 oracle, including normalized/raw outputs, per-head state,
continuations and partial chunks. With only the accumulator barrier, catastrophic output
errors disappeared but that same test still failed the per-head accuracy criteria. The
underlying reason for the output barrier's code-generation sensitivity is not established.

On RX 7600M XT, public running GDN, QK/value heads 16/48, T=1024, fused normalization,
cold-L2 graph replay, warmup 3 and 10 samples per run, two ABBA runs gave median times
4.490/4.504 ms before the precision corrections and 4.654/4.664 ms after (+3.6%). The
comparison baseline includes the output barrier but fails the numerical test. One same-shape
SIMT-fallback run measured 22.780 ms. This is an Op measurement, not an end-to-end prefill
claim; other shapes and full-model performance have not been measured for this change.
A broader candidate including extra prediction/history correction timed out on this workload
and was discarded; its cause was not isolated. Correcting the state update alone cost about
4% but still failed output and state criteria, and was also discarded.

The cal69 fragment diagnostic had an out-of-bounds write and used the wrong wave lanes; its
phase difference figures are invalid. Its corrected reconstruction uses only the active
16 lanes per wave; its aggregate oracle checks are not substitutes for the production test's
per-head and gross-error checks. The cal67 register-fed loop's 1.72x measurement reproduces,
but its dependent chains and host timing do not establish an instruction issue peak; the
98%-of-peak and exhausted-bit-exact-GEMM conclusions above are unproven. Cal42 uses 8192
MACs per 16x16x16 instruction instead of 4096, so its reported MAC rates are twice the
correct count and its constant-input loops also need compiler validation. Cal43 times
pre-expanded weights and pre-quantized activations with all i8 samples followed by all i4
samples, not ABBA; the 2.25-2.76x kernel-only figures exclude preparation and do not prove
end-to-end speed or model quality. Its packed representation does not establish the later
1.06-bpw projection, nor does four-bit arithmetic halve the mathematical MAC count.


## The BF16 prompt-attention candidate now has a multi-token ABBA: +25.9 % prefill, decode unchanged, output coherent

**What is in this commit** (`src/ops/softmax_attention/dense/causal_cache/`): the native BF16 WMMA QK
path for the 64-query-row prompt schedule (`bf16/tiled_mma.cuh`: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`
replaces the ldmatrix+mma sequence for that shape on HIP); the diet-path 4-warp QK changes and the LDS
64 KiB static_assert in `prompt_i8_fa2.cuh`; split plans keeping their key splits at the warp count the
shape fit to, with only the fitted stage count instantiated (`prompt_i8_fa2.cu`, `prompt_i8_fa2_plan.h`,
`small_t_i8_launch.cuh`, `bf16/template_launch.cuh`); and the test-side coverage in
`tests/ops/softmax_attention/causal_cache.cpp`. The i8 fan-out packed-select stays **production
default-off**: the diet loop keeps the original form and `-DNINFER_FA2_QK_PACK_ON` selects the packed
form for isolated experiments only.

The pre-existing ordinary causal-attention CTest abort in NVFP4 decode (`hipErrorInvalidValue`) is
**not** from this change: `/tmp/cal68/ninfer_tests.base` reproduces the identical failure on the base
tree.


Measured 14:56–15:00 on 2026-10-23, four legs inside **one** `flock /tmp/ninfer-gpu.lock` hold
(`/tmp/queue-bf16-mt.sh` → `/tmp/attn-bf16-abba-mt.sh` → `/tmp/bf16-mt-abba.log`, `rc=0`, lock
released 15:00:31). `ROCR_VISIBLE_DEVICES=1`, `ROCBLAS_TENSILE_LIBPATH=$HOME/opt/rocblas-gfx1102/library`,
`NINFER_T2_DECODE=simt`, `NINFER_T2_I8_SMALL=off`, model `bonsai2_27b_text_swift.ninfer`.
Arms `/tmp/codex-attn-serve-base` (sha `e67625fc…`) vs `/tmp/codex-attn-serve-new` (sha `2c4176bc…`);
the queue asserts the two are **not the same bytes** (`exit 23`) before the card is touched — a null pair
would produce a "win" out of nothing, so that check gates every rate number below.

The prior evidence for this pair was a server ABBA at a ~4,210-token BF16-KV prompt with
**`max_tokens=1`** (145.75 → 166.75 tok/s). That form proves the launch and the first token agree; it
cannot support a claim about model output, and the requirement was a real multi-token completion.
This run is `max_tokens=96`, `temperature=0`, prompt **5,106 tokens** (measured from the server's own
`req#1 done | … | prompt 5,106` line, not from the generator), KV cache reported by the server as
`KV 8,192 tokens, bf16, explicit`, legs base/cand/cand/base.

| leg | arm | prefill tok/s | decode tok/s | TTFT | output text sha1 | chars |
|---|---|---|---|---|---|---|
| 1 | base | **137.1** | 12.1 | 37.3 s | `50299a415d67` | 451 |
| 2 | cand | **171.6** | 12.0 | 29.8 s | `a20ac8c9052a` | 447 |
| 3 | cand | **170.8** | 12.0 | 29.9 s | `a20ac8c9052a` | 447 |
| 4 | base | **134.9** | 12.0 | 37.8 s | `50299a415d67` | 451 |

- **Prefill: base mean 136.0, cand mean 171.2 → +35.2 tok/s = +25.9 %.** The off-leg spread is 2.2 tok/s
  within base (1.6 %) and 0.8 within cand (0.5 %); the delta is ~16× the larger spread, so it is not
  inside noise, and the ABBA order is what licenses the comparison. TTFT moves 37.55 s → 29.85 s (−20.5 %),
  the same effect seen from the other side.
- **Decode: 12.1 / 12.0 / 12.0 / 12.0 tok/s — flat.** Expected: the change is in the prompt-attention
  path only, and this is the check that the win is not being paid for by a slower decode.
- **Determinism per arm: base1 == base2 and cand1 == cand2 by output sha1.** At `temperature=0` each arm
  reproduces its own 96-token completion byte-for-byte across a restart.
- **The arms are not token-identical, and that is the honest outcome.** They share a 415-character prefix
  (93 % of the shorter completion) and then differ by a lexical choice: base writes "…serves as a reliable
  **course marker** for sailors.\" Title: ", cand writes "…serves as a reliable **beacon** for sailors.\"
  Title: \"The". A BF16 WMMA path that changes the accumulation order is *expected* to land on a different
  synonym around token 90; it is not expected to be bit-exact, and bit-exactness against baseline is not
  what this candidate is being judged on. The standalone FP64/bitwise evidence stays as the numerics
  evidence; this ABBA is the serving evidence.
- **Coherence, checked not assumed:** both completions are well-formed two-sentence summaries of the
  (deliberately 156-times-repeated) lighthouse paragraph with a title starting, cut off only by the 96-token
  limit. Repetition probe: max 8-gram repeats **1**, distinct words 82 % on both arms — the model is
  summarising a repetitive prompt, not degenerating into it, which is the correct behaviour and the thing
  a 1-token completion cannot show.

Two limits to keep attached to the +25.9 %: it is measured at 5,106 prompt tokens, a longer prompt than
the 4,210-token shape of the earlier pair, so it is not a like-for-like repeat of the 145.75 → 166.75
figure (the direction agrees, the size does not transfer); and every leg ran while the census compile
(`/tmp/qfrag-census5.sh`, `nice -n 12`, single TU) was on the CPU — small, symmetric across arms, and it
inflates nothing on the candidate's side.

