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
  **Reconciled 2026-10-23 by cal70/cal71 (see the cal71 section):** the "~37%" figure is a ratio
  against the theoretical peak at the 2.33 GHz clock cal42 ran at. Against what the card is
  **measured** to sustain, 14.2 TMAC/s issued is **86% of the group-folded wall** (16.55 at the
  2.557 GHz observed under load) and **68% of the barrier-free clock-scaled extrapolation**
  (8.412 TMAC/s/GHz × 2.557 GHz = 21.5). The honest remaining scheduling headroom on the shipped
  tile is therefore **~1.5×, not ~2.7×** - and cal71 shows the tile's occupancy (130 VGPR ⇒ 15
  blocks/CU ⇒ 480 thr/CU) already sits inside the measured plateau, so most of that headroom is
  fragment/issue waste, not residency.
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

## 2026-10-23 causal prompt attention: native WMMA QK, the LDS ceiling, two gfx1102 defects

**Candidate.** Two attention changes on one working tree: the BF16 tiled-MMA prompt kernel
(`causal_cache/bf16/tiled_mma.cuh`, `bf16/template_launch.cuh`) and the INT8-family prompt kernel
(`causal_cache/prompt_i8_fa2.cuh`), which gains a native wave32 INT8 WMMA QK group
(`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`) for the `!PackedValues && !PackedKeys && Warps == 4
&& Stages == 1` shape - exactly the shape RDNA3 runs for the 27B's int8-g64 prompt. The shuffle
layout converts the native `[query row][8 keys]` fragment into the kernel's four-lane row ownership;
scores are promoted to FP32 with `__fmaf_rn`.

**Codegen (device-asm, `-fstack-count`/`.amdhsa_*`, gfx1102, same flags as the build).**
`/tmp/attn-asm-report2.py` over both trees, 124 BF16 rows, 264 INT8 rows:
- BF16 `bf16_kv_tiled_mma_kernel`: 21,341 → 14,206 instructions (−33%), 0 → 64 `wmma_*`,
  `v_perm_b32` 1,256 → 564, stack spill 1,104 B → 608 B (−45%), VGPR 256 (at the cap) in both.
  The grouped and merge kernels are byte-identical between trees.
- INT8 family: only the `int8-g64 W4 st1` kernels move - 25,404 → 21,808 instructions, 0 → 64 wmma,
  perm 2,020 → 1,902, spill 2,288 → 2,016 B, VGPR 256 both. Every `W8 st1`, every `st2`, every
  `rk8v4` and `rk4v4` row is identical base vs candidate. h16 and h24, masked and unmasked.

**End-to-end (production server, this card, one ABBA set, 4,210-token lighthouse prompt, KV default
BF16, `/tmp/queue-server-abba.log`): base 146.9 / 144.6 t/s, candidate 167.7 / 165.8 t/s - medians
145.75 → 166.75 (+14.4%)**, TTFT 28.7/29.1 s → 25.1/25.4 s, all four legs "KV 8,192 tokens, bf16,
explicit | pages 128/128", all four legs generated the same text. Scope: one prompt length, one
request per leg, one ABBA set. The candidate's INT8 op-level claims (34→23 ms, 96→65 ms, 208→143 ms
across three prompt contexts; sampled FP64 oracle 0/1536 mismatches, max_abs 1.229e-4) are the
candidate author's measurements, **not re-measured on this card**.

**INT8-KV server ABBA, same card, same prompt, `/tmp/attn-i8-server-abba.log` (inside
`/tmp/queue-i8fix.log`): base 131.1 / 148.5 t/s, candidate 170.3 / 169.7 t/s - medians 139.8 → 170.0
(+28.7%)**, TTFT 32.1/28.4 s → 24.7/24.8 s, all four legs "KV 8,192 tokens, int8, explicit | pages
128/128", all four legs generated the same text. The base legs ran on the pre-defect-B binary and
the server's own launch path never hit that window, so the server numbers are not affected by it.
Both server medians are single ABBA sets at one prompt length with one request per leg; no
multi-length benchmark backs either number.

**gfx1102 LDS law (probe3, `/tmp/attn-lds-probe3.cu`).** ROCr/HIP validates static and dynamic
shared **together** against one 65,536 B per-workgroup budget: a kernel with 19,392 B static reports
`maxDynamicSharedSizeBytes = 46,144`; requesting 65,536 max-dynamic on it is `hipErrorInvalidValue`;
a launch whose total is 84,928 B is *dispatched* by ROCr (`group_seg_size=84928`) and then aborts the
queue with `HSA_STATUS_ERROR_INVALID_ALLOCATION (0x1003)`, leaving a sticky `hipErrorIllegalAddress
(700)` for everything after it in the process. Dynamic > 65,536 on a zero-static kernel is rejected
at request time; > 1,024 threads is `hipErrorInvalidConfiguration (9)`.

**Defect B - fixed and verified.** `configure_cuda_device_once` (`src/core/device.h:60`) wraps
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`; the small-T
DynamicArena tier for `implementation_window ∈ (2054, 8198]` asked KeyBlock = 64 → 65,536 B dynamic
on top of ~19 KB static → the dispatch-abort above → the INT8 prompt test died at `device.h:60` with
rc 134. That window covers the 4,210-token lighthouse prompt. Fixed in
`causal_cache/small_t_i8_launch.cuh` under `#if defined(__HIPCC__)` by descending the tier to
`<12, 1, 32, false>` (no arena); the sm_8x path keeps `<12, 1, 64, true>` under `#else`. Verified:
the same test on the fixed binary crosses keys=8199 (case 253) with no `device.h:60` abort.

**Defect A - root-caused, fixed, verified (2026-10-23, 08:45 run).** The attention test contract queries
capacity for min=max=tokens with the launch envelope, allocates exactly that, runs, then requires
`workspace.used() == 0 && workspace.peak_used() == workspace_bytes` ("workspace query/execution
high-water mismatch", `tests/ops/softmax_attention/causal_cache.cpp:2383`). On HIP the LDS fit
descends 8-warp 2-stage → 1-stage → 4-warp 1-stage once `SmemBytes` exceeds the 64 KiB ceiling; for
int8-g64 (unpacked K **and** V) the 8-warp 1-stage shape is 32,768 + 33,792 + 2,048 = 68,608 B, so
int8-g64 is the only storage that descends to four warps (pinned by new `static_assert`s in
`prompt_i8_fa2.cuh:150-155`: 68,608 > 65,536, packed 60,928 and 52,736 fit). The launcher's split
condition was `plan.splits > 1 && fit_warps == 8`, so a fitted 4-warp shape ran **unsplit** and
allocated **zero** workspace while the query returned the plan's split bytes.

Which cases can hit that is decided by the plan, and the failure labels say the answer exactly: all
eleven mismatches are `d256-h24-kv4 int8-g64 mapping=fragmented` (T=17/128/257/258/513, keys
4,117–40,000, plus one `batch B=1 W=65`), i.e. **q_heads = 24 only**. With the test's own SM count
(16, `causal_cache.cpp:49-56`) `waves(x) = 0.85*ceil(x/16) + 0.15*x/16` is sublinear exactly when
`ctas = div_up(width,128)*q_heads` is **not** a multiple of 16: h24 gives ctas=24, `waves(24)=1.925`,
`splits=2` gives `waves(48)/2 = 1.5` → split cost `1.5*1.05 + merge < 1.925` wins for keys ≳ 200, so
the plan splits; h16 gives ctas=16,32,48 → `waves` is exactly linear, so split cost ≥ unsplit cost
and `plan.splits == 1` at every key count. That single fact explains the whole failure set:
- h24 int8-g64: plan splits → query prices split bytes, launcher (fit to 4 warps, old condition
  demanded 8) launches unsplit with zero allocation → high-water mismatch. 11 cases, all of them.
- h24 packed (rk8v4/rk4v4): plan splits, 8 warps survive the fit, splits launch → pass.
- h16 any storage: plan never splits → query returns 0 bytes, unsplit launch allocates 0 → pass.
`mapping=fragmented` is a case-table property, not causal.

The defect-B-fixed run confirms the prediction exactly (`/tmp/attn-i8fix-test.log`, 08:36, rc 1):
eleven "workspace query/execution high-water mismatch" lines, all
`d256-h24-kv4 int8-g64 mapping=fragmented` at T ∈ {17, 128, 257, 258, 513} with keys 4,117–40,000
plus one `batch B=1 W=65` - and nothing else - alongside exactly the three predicted
`prompt cases d256-h16-kv2 exercised no split launch` coverage lines (int8-g64, rk8v4, rk4v4). No
packed-storage case failed.

The three "prompt cases d256-h16-kv2 exercised no split launch" coverage lines
(`causal_cache.cpp:3730`, int8-g64 / rk8v4 / rk4v4) are therefore **not** launcher defects: they
report that no h16 prompt case planned a split, which on a 16-SM card the cost model never does
(ctas ∝ q_heads = 16 = SM count). The four-warp fix cannot and does not clear them; clearing them
would need a plan re-derivation for 16-CU devices (see the cost-model note below), not a launch fix.
Fix: splits are legal at four warps - a partial row belongs to exactly one warp at either CTA width
(`Br == Warps * 16`, eight row-pairs per warp) and the partial layout and merge kernel name no warp
count - so `fit_split = plan.splits > 1` and `static_assert(!Split || Warps == 4 || Warps == 8)`.
The stage dispatch became `if constexpr (Fit::StageCount == …)`, which also stops instantiating both
stages of every geometry (RDNA3 never runs the 2-stage ones; the prompt TU had grown to ~175 MB of
`.s`). Numerics: a split reorders the FP32 softmax merge (per-split partial max/sum), so it is not
bit-identical to unsplit - the same route the repo already runs on NVIDIA and for packed storages on
HIP, gated by the production test's own oracle, not a new numerics class. Expected result, stated
before the run: the 11 high-water mismatches clear; the h16 coverage lines stay (structural at 16
SMs), so the int8-family prompt route still reports rc 1 on this card.

**Verified on the fix build** (`/tmp/attn-i8fixA-test.log`, 08:51,
`ninfer_tests ninfer_softmax_attention_test --int8-prompt-only`, route line
`FAIL causal_softmax_attention int8-family prompt route`, **`fail_cases: 1  high-water: 0`**): every
previously-failing shape is now `ok` - int8-g64 h24 cases 9, 11, 14, 16, 17, 19, 20, 21, 29, 33, 34
(T = 17/128/257/258/513, keys 4,117–40,000) and the batch row at case 35 - with **zero** high-water
mismatch lines, and exactly the three predicted `prompt cases d256-h16-kv2 exercised no split launch`
coverage lines (int8-g64, rk8v4, rk4v4). The whole defect-A failure set is cleared by the launcher
change; the residual rc 1 is that coverage assertion, not a launch defect. 510 cases in 398 s.

**LDS bytes the 4-warp split launch asks for.** The FA2 prompt kernel declares only
`extern __shared__` (`prompt_i8_fa2.cuh:268`) - no static shared - so the full 65,536 B
per-workgroup budget is the dynamic allowance. 1-stage shapes
(`SmemBytes = Br*256 + Stage::Bytes + Br*16`, `Br = Warps*16`): int8-g64 8-warp 68,608 (over the
ceiling, descends) → **4-warp 51,200**; packed-values 8-warp 60,928 → 4-warp 43,008; packed-both
8-warp 52,736 → 4-warp 35,328. Every 4-warp shape leaves ≥ 14 KiB headroom, so a split launch at four
warps cannot hit the probe3 dispatch-abort path; the run above ran the splits and never touched
`HSA_STATUS_ERROR_INVALID_ALLOCATION`. **iu4 (4-bit activations) remains user-gated and untouched.**

**Cost-model gap (not a correctness issue).** `causal_prompt_fa2_plan` prices a split CTA grid as
`div_up(width,128) * q_heads` and waves as `0.85*whole + 0.15*fraction` (constants re-derived on the
RTX 3090's 82 SMs, 2026-10-10; plan measures 1.5% off the best forced candidate, worst regret 21%).
A four-warp split launch is `div_up(width,64) * q_heads * splits` - 2× the CTA count the split cost
prices. Unquantified on 16 CUs; the plan is a heuristic, so the fix's effect on INT8 prompt time is
an open measurement, not a deduction.

**That measurement is now closed - the fix is performance-neutral on the server** (2026-10-23,
`/tmp/queue-splitfix.log`, ABBA on `/tmp/codex-attn-serve-new` vs `/tmp/codex-attn-serve-fix`,
`--kv-dtype int8`, prompt 4,210, one lock hold for the whole set): **A1 170.8, B1 170.7, B2 170.3,
A2 169.4 tok/s**; TTFT 24.7 / 24.7 / 24.7 / 24.9 s; all four legs report the same capacity line
(`KV 8,192 tokens, int8, explicit | pages 128/128`) and the same generated text (`We`). New-vs-fix
means 170.1 vs 170.5, i.e. **-0.24%, inside the 1.4-tok/s spread the two binaries show across their
own repeated legs** - so no effect is resolvable, which is the expected outcome for shapes that
already launched unsplit on this card (the splits the fix enables fire on h24-class shapes, not the
h16 server shape). The fix therefore stands on its correctness contract (`workspace.used() == 0 &&
workspace.peak_used() == workspace_bytes` on every launch it makes legal), not on a speed claim.
**iu4 (4-bit activations) remains user-gated and untouched.**

## 2026-10-23 cal70: a clean WMMA peak probe, and the clock datum the peak question was missing

The audit above leaves the roof open: cal67's 12.95 TMAC/s is 66% of the 19.6 T MAC/s spec anchor
(line 524), which is exactly what the same anchor gives at ~1.54 GHz (`19.6 / 2.33 = 8.412` TMAC/s
per GHz; `8.412 × 1.54 = 12.96`). An ILP-only probe cannot tell "chain-latency-limited issue" from
"clock-limited pipe", so **cal70 measures both at once**: `.qual/wmma/cal70.cu`, queue
`/tmp/queue-peak.sh` → `/tmp/cal70-out.txt`, clock trace `/tmp/cal70-clock.log`. **Measured**
(lock 08:55:38 release, 9 reps per leg, legs cycled, outcheck `u8 28672/28672, bf16 28672/28672
nonzero words` - no leg optimized away, event timer alive on every leg):

| leg (7 blocks/CU, 224 thr/CU) | event median | = TMAC/s | event/wall |
| --- | --- | --- | --- |
| `u8` C2 | 0.2070 ms | **14.18** | 0.991 |
| `u8` C8 | 0.8106 ms | **14.49** | 0.998 |
| `u8` C16 | 1.5753 ms | **14.91** | 0.999 |
| `u4` C2 | 0.1160 ms | **25.32** | 0.985 |
| `u4` C8 | 0.4252 ms | **27.62** | 0.996 |
| `u4` C16 | 0.8367 ms | **28.07** | 0.997 |
| `bf16` C2 (clean arm) | 0.2024 ms | **14.50** | 0.991 |
| `bf16` C8 / C16 (spill-bound) | 3.0666 / 5.5779 ms | 3.83 / 4.21 | 0.999 / 1.000 |

Readings: ILP helps, weakly - iu8 rises 14.18 → 14.91 (+5.1%) from C2 to C16, and iu4 25.32 → 28.07
(+10.9%), so the 8-bit legs at 7 blocks/CU are not issue-bound at C2 alone; **iu4 / iu8 = 1.883× at
C16**, close to the 2× the instruction set promises; **bf16 at C2 matches iu8 to within 2%**
(14.50 vs 14.18 - same 4096-MAC-per-instruction cost, no pipe reward for the wider fragments, and
the C8/C16 bf16 legs are spill artifacts, not measurements).

**Instrument.** ILP is a template parameter (2, 8, 16 independent accumulator chains); each chain's
B fragment is a distinct permutation of `(s, lane, g, c)`, so LLVM cannot fold chains into one (it
did when they were identical), and the group fold consumes every chain, so no group is dead code.
4096 MAC per 16x16x16 instruction (cal67's accounting; cal42's 8192 doubles its rates). Register-only
compute, no LDS, no loads in the loop. `G = 112` = 16 CUs × 7 resident 32-thread blocks, 24
launches per timed window, 9 reps, legs cycled so every arm is timed right after all others ran
(whole-set boost state, not pairs). Every leg reports **both** a `hipEvent` pair recorded on
`hipStreamLegacy` and the host wall clock over the same window, and prints `event/wall` - cal67
found the event timer read 0.0000 on the u4 arm and timed host-side only, which is invisible if the
event leg is not printed. Non-triviality: an outcheck counts nonzero output words over all
`G*32*8 = 28,672` accumulator words per arm, printed before the rates, so a leg that optimized away
is visible. Three arms: iu8 (the int8 GEMM/attention lane), iu4 (the user-gated
reference), bf16 (the production KV attention lane; this Clang's builtin takes `(A, B, C)` only, as
`tiled_mma.cuh:179` calls it, with `short __vector_size(32)` fragments per `tiled_mma.cuh:163`).

**Register cost of the legs** (device asm, `--save-temps`, `.set _Z…num_vgpr` /
`.amdhsa_private_segment_fixed_size`): iu8 72 / 152 / 190 VGPR at C = 2 / 8 / 16, iu4 52 / 64 / 80,
private segment 0 for all of them - so every 8-bit leg is spill-free even at 16 chains, and a plateau
at C2 while C8/C16 hold is chain latency, not pressure. bf16 is the exception: 169 VGPR clean at
C = 2, but 256 + 1,380 B (C = 8) and 256 + 2,564 B (C = 16) of private segment, so **only the bf16
C = 2 leg is a clean bf16 measurement** and its bf16 peak must be read from that leg. `group_segment`
is 0 for every leg (no LDS in the probe).

**Clock datum, and why it gated the conclusion.** sysfs on the working card (`/sys/class/drm/card1`,
the 7600M XT; card0 is the iGPU, 606 MHz idle) read **idle** during the earlier test legs:
`hwmon/hwmon4/freq1_input = 1,300,000,000`, `freq2_input = 1,124,000,000`, `gpu_busy_percent = 30`.
That reading is not the clock a sustained WMMA loop runs at, and treating it as such inverts the
answer: the 19.6 T MAC/s anchor is quoted at 2.33 GHz (`8.412` TMAC/s per GHz), so at 1.30 GHz the
same shape would give ≈ **10.9 TMAC/s** - *below* cal67's measured 12.95 and below the engine's 12.7,
which is arithmetically impossible for a measured rate under load. The probe's in-hold 0.5 s trace resolves it:
`f1` climbs 987 → 1341 → 2161 → 2392 → **2472/2473/2474/2483 MHz** with `busy = 100` on 7 of the
last 8 samples, i.e. the card boosts to **2.39–2.48 GHz** under this load, and the idle 1.30 GHz
sample must not be used as the peak-run clock. At the observed boost clock cal70's best 8-bit leg
(14.91) is **6.04 TMAC/s per GHz** against the spec anchor's 8.412 - **72% of the clock-scaled
spec peak**. The clock caveat is therefore discharged for cal70/cal71 (in-hold trace, `busy = 100`),
while the trace is thin (0.5 s cadence, one ramp) and the per-GHz figure carries that width.
**iu4 remains user-gated and untouched.**

## 2026-10-23 cal71: occupancy is the measured variable — where the 8-bit pipe wall sits

cal70's 14.91 TMAC/s came from one fixed occupancy (112 blocks = 7 resident 32-thread blocks/CU,
224 threads/CU) and rose with ILP. cal71 asks the other axis: `.qual/wmma/cal71.cu`, queue
`/tmp/queue-cal71.sh` → `/tmp/cal71-out.txt`, 0.1 s clock trace `/tmp/cal71-clock.log`, one lock
hold 09:11:14–09:11:22, 7 reps per leg, legs cycled, outcheck `u8 159744, bf16 49152` nonzero words
(all accumulators live), `event/wall` 0.974–0.999, max observed `f1 = 2,557,000,000` with
`busy = 100` on 63 of 66 samples.

**Occupancy is a launch parameter, not a compile parameter — and putting it in the compile parameter
is harmful.** The first cal71 drafted `__launch_bounds__(32, MB)` with high `minBlocks`; the device
asm showed LLVM reacts by truncating VGPRs to the `65536/(32*MB)` cap and **spilling** (u8 C16 at
min 7: 144 VGPR + 324 B private segment; u8 C8 at min 7: 144 + 36 B; every bf16 arm at min ≥ 7:
164–3,480 B), against cal70's spill-free 152/190. The shipped probe therefore builds every kernel
with `__launch_bounds__(32, 4)` only (512-VGPR cap, satisfied with `private_segment 0` and
`group_segment 0` on all arms - asm-checked) and sets residency by the grid `G = 16 * MB`, kept at or
below each kernel's natural VGPR capacity `floor(65536/(32*vgpr))`: u8 72/152/190 VGPR → 28/13/10
blocks; u4 52/64/80 → 39/32/25; bf16 C2 169 → 12. Every block is resident in one wave, so each
window measures the occupancy it names. A 6 s `u8` warm-up precedes measurement, because cal71's
windows are ~0.2 s and cannot boost the card on their own.

**Measured medians** (event, TMAC/s at 4096 MAC per instruction; `thr/CU` = 32 × blocks-per-CU):

| blocks/CU → | 4 | 7 | 10 | 12 | 13 | 16 | 20 | 25 | 28 | 32 | 39 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `u8` C2 | | 13.45 | | | | **16.55** | | | 16.35 | | |
| `u8` C8 | | 14.62 | | | 14.31 | | | | | | |
| `u8` C16 | **16.39** | 15.25 | 14.78 | | | | | | | | |
| `u4` C2 | | 22.74 | | | | | 30.01 | | | | **30.62** |
| `u4` C8 | | 27.80 | | | | | | | | **32.51** | |
| `u4` C16 | | 27.84 | | | | | | 30.05 | | | |
| `bf16` C2 | | 13.61 | | **16.50** | | | | | | | |

**What this says about the bit-exact 8-bit lane.**

1. **The iu8 wall is ≈ 16.4–16.6 TMAC/s**, and it is reached from two opposite directions: high
   occupancy at low ILP (C2 @ 16 blocks = 16.55) and low occupancy at high ILP (C16 @ 4 blocks =
   16.39). Pushing either axis further *past* that point does not help (C2 @ 28 = 16.35; C8 @ 13 =
   14.31; C16 @ 10 = 14.78). Occupancy at fixed low ILP is worth **+23%** (13.45 at 224 thr/CU →
   16.55 at 512), which clears the probe-to-probe noise floor (cal71's own C2@7 leg reads 13.45
   where cal70 read 14.18 for the same shape, so differences under ~5% are not resolvable, and
   anything at or above +16% is).
2. **The iu4 wall is 32.51** (C8 @ 32 blocks; 30.62 at C2 @ 39), so **iu4/iu8 = 1.96× measured at
   the two walls** - the instruction set's 2× promise, confirmed at the point where both families
   saturate, and higher than cal70's 1.883× measured at one fixed low occupancy. This is the
   arithmetic cost of 4-bit *activations*, which remain **the user's gated decision, untouched.**
3. **bf16 buys nothing over iu8 in MAC terms**: 13.61 vs 13.45 at 7 blocks, 16.50 vs 16.55 at the
   plateau. The production KV-attention lane has the same 4096-MAC-per-instruction budget as the
   int8 GEMM lane, so attention efficiency is a scheduling and data-supply problem, not an
   instruction-format problem.
4. **cal71's wall is 77% of cal42's barrier-free peak.** cal71's 16.55 at the observed 2.557 GHz is
   6.47 TMAC/s per GHz against cal42's 8.412 (= 19.6 T MAC/s at 2.33 GHz), and the difference is
   structural in the probe: cal71 folds `accf += acc[c]` once per group, a barrier cal42's
   fully-independent accumulator loop does not have. Honest statement: barrier-free issue ceiling
   ≈ 8.412 TMAC/s per GHz (cal42), group-folded ceiling ≈ 16.5 TMAC/s at 2.557 GHz (cal71), and the
   shipped GEMM's 14.2 TMAC/s issued sits at **86% of the folded wall**.

**Consequence for the shipped tile.** The HIP int8 WMMA tile runs 32 threads/block at
`__launch_bounds__(32, 8)` with 130 VGPR (`t2_prefill_wmma.cuh:86`), which the driver packs to
`floor(65536/(32*130)) = 15` blocks/CU = 480 threads/CU - already inside cal71's plateau (which
spans ~512–896 threads/CU at 16.35–16.55). So the tile does **not** need a VGPR diet to reach the
occupancy sweet spot; it needs the grid to actually fill every CU with ~15 blocks (panel/wave shape),
and its remaining 14.2 → 16.5 gap is fragment waste and issue scheduling - the same conclusion the
line-534 roofline reached, now with the occupancy axis measured and closed. **iu4 remains
user-gated and untouched.**

## 2026-10-23 prefill attribution and the 300 t/s roof arithmetic

What 300 t/s costs on this card, computed from measurements already in this document (BF16-KV probe
run, 174.7 t/s, `/tmp/attn-probe-bf16.log`; probe leg 170.0 t/s INT8-KV; cal71 walls):

- **Budget.** 300 t/s = **3.33 ms/token**. Today's measured 170 t/s is **5.88 ms/token**; at the
  4,210-token prompt that is a TTFT of 24.7 s, where 300 t/s would need 14.0 s. The target is
  therefore a **−43% system-wide cut**, not a kernel-local win.
- **Where prompt time goes at 4,210 tokens** (BF16-KV probe, share of prompt layer time):
  **attention mixer 27.1%** (6,225 ms of 22,980 ms, and **quadratic** - each +1,024-token step adds
  ~530 ms), **MLP buckets 46.6%**, `gdn_mixer` 23.2%. The mixer runs at ≈**0.28 TMAC/s**, about
  **2% of the pipe peak cal71 measures** - by far the largest efficiency gap per unit of compute in
  the model, and it is the term that grows with context.
- **The GEMM floor.** Weight-MAC cost is ≈2.2e10 MAC/token ⇒ at cal71's iu8 wall (16.55 TMAC/s)
  the GEMM cannot go below **≈1.33 ms/token = 40% of the 300 t/s budget**. The shipped GEMM issues
  14.2 TMAC/s, i.e. 86% of that wall, so **all remaining bit-exact GEMM headroom is ≈0.4 ms/token
  = 7% of the budget**. Even a perfect bit-exact GEMM lands at ~250–260 t/s, not 300.
- **iu4 at its wall** (32.51 TMAC/s, 1.96×) would cut the MLP bucket by ~55% — ≈2.74 → ~1.20
  ms/token, −26% of the total ⇒ **~230 t/s**. GEMM alone, even with 4-bit activations, does not
  reach 300; the quadratic attention term has to shrink too. And iu4 is a **numerics change the
  user gates** — measured as a pipe reference here, shipped nowhere.
- **Existence proof.** Prism runs **296 t/s prefill on this same card** (documented earlier in this
  file), so 300-class numbers are physically reachable on gfx1102; the gap is our kernel mix, not
  the silicon.

**Priority order this arithmetic implies** (bit-exact, no user-gated numerics): (1) attention-mixer
efficiency — 27% of prompt time at 2% of measured pipe peak, quadratic; (2) `gdn_mixer` at 23%;
(3) GEMM panel/issue headroom, worth ≤7% of the budget; (4) the **gate_up routing gap below**,
which is a plumbing fix with a numerics-visible (not bit-exact) boundary that the user must see
before it ships.

## 2026-10-23 T2 GEMM arithmetic map: gate_up never takes the integer WMMA tile

An Explore pass over the T2 linear routing (entry `src/ops/linear/linear.cpp:115-121`), verified by
hand on the citations below:

- **`gate_up_proj` (34816×5120) is A16 by construction.** The model layer sets the integer route at
  `src/models/qwen3_5/execution/parameters.cpp:139`
  (`integer_route(out.gate_up, {Q4_G64_FP16, T2_G128_FP16}, 34816, 5120)`), but the SwiGLU wrapper
  calls the **3-arg `linear` overload**
  (`src/ops/wrapper/linear_swiglu.cpp:161-170`, the `if (t2_weight)` branch:
  `linear(x, gate_up_weight, gate_up, stream);`), and that overload hard-codes
  `LinearPolicy::A16Only` with `workspace == nullptr`
  (`src/ops/linear/linear.cpp:209-212`) - the wrapper's own comment says "**A16 by construction**".
  With no workspace, the integer route is set and then **ignored**: gate_up lands on
  `launch_t2_mma_r64_c128`, the A16 BF16 tile
  (`src/ops/linear/t2/t2_rowsplit_gemm_mma.cuh`; BF16 tensor-core ops at `:249` and `:266`, with a
  weight→BF16 expansion in shared memory at `:188-200`). Standalone, that shape measures
  **239 ms (A16) vs 28.7 ms on the WMMA int8 tile** at T=1024.
- Everything else in the T2 path takes the hardware-WMMA int8 tile: `down_proj`
  (`parameters.cpp:140`), `o_proj` (`:183`), `gdn_output` (`:198`), the input projections,
  `linear_add`, and `lm_head` → `t2_a8_linear` →
  `src/ops/linear/t2/t2_prefill_wmma.cuh` (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` at
  `:151-174`; PRMT-only 2-bit→s8 decode `wmma_code_frag` at `:58-67` with
  `kTable = 0xFFFE0100u`; fp16 scale fold at `:180-203`). Activations are int8 codes plus one fp16
  scale per token (`t2_a8.h:51-56`); all T2 entry points require BF16 x/out.
- **There is no shipped 4-bit-activation path**: `LinearPolicy::AllowA4` throws
  (`src/ops/linear/t2_dispatch.cpp:88-93`), and `wmma_i32_16x16x16_iu4_w32` appears only in the
  `.qual/wmma` probes. The iu4 numbers in this document are pipe references for the user's gated
  decision, nothing more.
- Env knobs in force: `NINFER_T2_DECODE=simt` (T==1 decode only), `NINFER_T2_I8_SMALL=off` (default
  on HIP), `NINFER_T2_A8_COLSPLIT=off` (−27% at engine).

**Status: this is a report, not a shipped change.** Routing gate_up onto the integer WMMA tile
needs (a) a workspace threaded through the SwiGLU wrapper, and (b) a numerics sign-off, because the
integer tile consumes int8-quantized activations where the A16 tile consumes BF16 — outputs move, so
it is **not bit-exact** even though the weights stay identical. That is the user's call, like iu4.
**iu4 (4-bit activations) remains user-gated and untouched.**

## 2026-10-23 attention device-asm cost table: every prompt MMA kernel spills — and the int8 prompt test fails identically on base

The roof section above calls the attention mixer "the largest efficiency target" (~27% of prompt time
at ~0.28 TMAC/s, about 2% of the pipe peak cal71 measures) without a mechanism. This section supplies
one, measured from the device asm (`.amdhsa_*` kernel metadata in the gfx1102 code objects built with
`-c --save-temps`; parsed from the `amdhsa.kernels:` YAML block, fields `.vgpr_spill_count`,
`.sgpr_spill_count`, `.private_segment_fixed_size`, `.group_segment_fixed_size`).

**Table 1 — i8 FA2, base tree** (`src/ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cu`,
`causal_attention_prompt_i8_fa2_kernel<CausalAttentionGeometry<24,4,1,41>, PagedKVDirectMetadata, …>`
= d256-h24-kv4 and `<16,2,2,41>` = d256-h16-kv2; both geometry families carry the full arm set L∈{4,8}
× three bool flags):

| kernel family | vgpr_spill_count | sgpr_spill | private_segment | group_segment |
|---|---|---|---|---|
| `causal_attention_prompt_i8_fa2_kernel`, every arm | **444 – 493** | 0 | **2,208 – 2,352 B/thread** | **0** |
| `causal_attention_prompt_i8_fa2_merge_kernel` | 0 | 0 | 0 | 0 |

- The i8 FA2 kernel is **register-limited, not LDS-limited**: every arm requests zero shared memory and
  spills 444–493 VGPRs (~1.7–1.9 KB/thread of live state) to per-thread scratch. At the shipped 32
  threads that is 70.7–75.3 KB private per block; at 16 blocks/CU it is ~1.1 MB/CU of scratch that the
  inner loop streams through. That is a concrete mechanism for the 2%-of-pipe figure: the kernel issues
  WMMA at full rate while reloading hundreds of accumulator/fragment words from scratch between issues.
- The split-merge epilogue on the same geometries is completely clean (0/0/0), so all the pressure is
  in the main kernel's live state, not in the reduction.

**Table 2 — bf16** (`.../causal_cache/bf16/`, from the `launch.cu` code object):

| kernel | vgpr_spill_count | private_segment | group_segment |
|---|---|---|---|
| `bf16_kv_tiled_mma_kernel<CausalGeometry<256,24,4> / <256,16,2>, Bf16KvTiledMmaSchedule<64,32,1>>` | **266** | 1,104 B | 0 – 33,152 |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<16,32,2,2,1>>` | **95 – 99** | 1,024 – 1,056 B | 1,064 – 33,152 |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<32,32,1,2,…>>` | **164 – 165** | 1,088 – 1,104 B | 32,912 |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<64,32,1,2,…>>` | **163 – 166** | 1,088 – 1,136 B | 32,912 – 33,152 |
| `bf16_kv_merge_kernel<…, Bf16KvMergeSchedule<256,256>>` | 0 | 0 | 32,912 – 33,152 |

So the pattern is uniform across the prompt-attention MMA families on this card: **the main kernels
spill (95–493 VGPRs, 1–2.35 KB/thread private), the merge kernels do not.** The bf16 arms are
LDS-bound in a way the i8 arms are not (32.9–33.2 KB group on most arms, within the 64 KiB
`kBf16CausalLdsCeilingBytes` law in `bf16/lds_fit.cuh`), the i8 arms are purely register-bound.

**Candidate delta (asm-real).** In the candidate tree exactly two FA2 arms differ from base, both the
`…Li4ELb0ELb0ELb0…` (L=4, both flags off) variants: d256-h24-kv4 `493 → 437` spills with
`2,288 → 2,016` B private, d256-h16-kv2 `493 → 434` spills with `2,288 → 2,016` B private. All other
arms' metadata are identical between trees. So the candidate's spill diet is real at compile time on
the two arms it targeted (−11% spills, −272 B/thread private) and inert elsewhere.

**The ABBA on the int8 prompt test, checked rather than excused.** `/tmp/attn-i8-abba.log`, four legs
inside one lock hold, 07:55:47 → 08:02:20:

- base1, cand1, cand2, base2 each report `ok_cases=94 fail_cases=10` and exit **134** (SIGABRT).
- The per-leg failure signatures are **byte-identical** (extracted signature lists hash equal): the
  same ten cases (9, 11, 14, 16, 17, 20, 21, 29, 33, 34 — int8-g64 d256-h24-kv4 prompt cases at
  history 4,100 / 33,000 with tokens 17/128/257/258/513, a1 and a3), the same ten
  `causal_softmax_attention* … mapping=fragmented … workspace query/execution high-water mismatch`
  lines, the same `int8-g64 prompt cases d256-h16-kv2 exercised no split launch` assertion, and the
  same aborting `src/core/device.h:60: CUDA_CHECK(result) failed: hipErrorInvalidValue: invalid
  argument`.

Two conclusions, both useful:

1. **On this test, cand and base are indistinguishable.** The ABBA is symmetric (4/4 identical legs),
   so the working tree introduced no measurable regression here — and the cand is equally *not*
   validated by it. A test that aborts identically on both trees cannot discriminate in this area. The
   handoff's "pre-existing failures" label is correct for these ten cases, and it is now backed by a
   four-leg byte-equal signature rather than by assertion.
2. **The pre-existing defect is the actionable finding.** On the HIP build the int8 prompt path (a)
   trips the workspace query/execution high-water contract on `mapping=fragmented` at
   `keys=4117/4357/33017/33128/33257/33258/33513/40000` — the query leg's advertised peak and the
   execution leg's used high-water disagree — and (b) reaches a `hipErrorInvalidValue` launch that
   aborts the test before the h16 "no split launch" coverage case can report. Under the gfx1102 launch
   law (64 KiB shared per workgroup, 1,024 threads) `hipErrorInvalidValue` at a launch is exactly what
   an over-subscribed LDS or thread-count request looks like from the host side.

**That diagnostic step is closed - both halves, by the two runs above.** The `hipErrorInvalidValue` at
`src/core/device.h:60` was Defect B (the small-T `DynamicArena` tier asking 65,536 B dynamic on top of
its static shared), fixed under `#if defined(__HIPCC__)` by descending that tier to
`<12, 1, 32, false>`; the ten high-water mismatches were Defect A (a 4-warp-fitted int8-g64 shape
launching unsplit while the query priced the plan's split bytes), fixed by `fit_split =
plan.splits > 1`. Both are documented with their verification runs at "Defect B - fixed and verified"
and "Defect A - root-caused, fixed, verified" above. The 10-vs-11 case-count difference between the
ABBA legs and the 08:36 run is the abort, not the fix: the ABBA legs died at `device.h:60` before
reaching the 11th case (the `batch B=1 W=65` row), the fixed binary crosses it.

What the ten-case classification came down to, stated plainly: every failing shape is a **planned split
that never launched**, and the plan splits exactly when `ctas = div_up(width,128) * q_heads` is not a
multiple of the SM count - on a 16-CU card that means q_heads = 24 and not 16. `mapping=fragmented`
and the key tiers are case-table consequences, not causes.

**Residual open items in this area, unchanged by the fixes:** (1) the i8 FA2 spill diet stays
asm-plausible only - 443–493 spills and 2.2–2.35 KB private per thread on every prompt-attention main
MMA arm, register-limited (`group_segment 0` on all arms), with the candidate's −11% spill / −272
B/thread delta measured at compile time; it needs an engine ABBA now that the test no longer aborts,
which is the first run that can discriminate it. (2) Any bf16 diet has to respect the `lds_fit.cuh`
descent law, not just the VGPR count. (3) The plan's wave model still prices a 4-warp split CTA grid at
`div_up(width,128) * q_heads` while the launch is `div_up(width,64) * q_heads * splits` - scoped by the
width bound recorded with the knob below to *short-query launches over long caches*, which is the only
place a split can be afforded at this card's 8,192-token KV capacity, so it is a verify/small-prompt
item and not a 300 t/s long-prompt lever. **iu4 (4-bit activations) remains user-gated and untouched.**

**Item (3), worked out from the plan code (analysis, no measurement in it).** `causal_prompt_fa2_plan`
prices the unsplit baseline at `narrow_ctas = div_up(width,64) * q_heads` with the narrow sweep rate
0.65, and every split candidate at `ctas = div_up(width,128) * q_heads` times `splits` with rate 1.0
(`prompt_i8_fa2_plan.h:111-125`) - i.e. it prices splits as the **eight-warp** shape. On this card the
int8-g64 8-warp 1-stage shape cannot fit 64 KiB, so the launcher descends to 4 warps and the real grid
is `div_up(width,64) * q_heads * splits` (`prompt_i8_fa2.cu`, `grid = {div_up(width, Shape::Br),
QHeads, Split ? plan.splits : 1}`) sweeping at the narrow rate. Both halves of the model error point
the same way for the h24 rows that split here: at width 128, splits 1, the plan's baseline is
`waves(48) * 0.65 * keys = 1.95 keys` - exactly what the 4-warp launch pays - while the plan's split
cell is `waves(24*2) * 1.0 * keys / 2 = 1.5 keys` + merge, where the launch the plan picked actually
pays `waves(96) * 0.65 * keys / 2 = 1.95 keys` + merge. So on a 16-CU card the plan's split cells are
optimistic by roughly the 2x CTA-count and the 1.54x rate term, and a planned split is at best
break-even against unsplit: the plan buys the merge (`kCausalPromptFa2SplitPerRowHead * width * splits
* q_heads`, plus the FP32 merge kernel and its partials traffic) for a sweep saving that the fitted
shape does not deliver. That is the arithmetic behind the split-fix server ABBA measuring neutral
(170.8/170.7/170.3/169.4) on this prompt, and it predicts the actionable version of item (3): price
the split candidates at the shape the launcher will actually fit - `Br = 64`, rate 0.65 - which needs
the plan to see the LDS fit it currently ignores, then re-derive the constants on this card the way
:47-57 describes doing on the 3090. Test it by forcing split counts on h24 shapes: nothing in `dense/causal_cache/` reads an env
variable today (grep for `getenv` there is empty), so the forced sweep needs a host-side
"NINFER_FA2_SPLITS=n" override in the launcher - the same shape of knob the GEMM cal probes use - and
should be read against measured time, not against the plan's own predictions. The override belongs in
`causal_prompt_fa2_plan` itself, not in the launcher, because three consumers read the plan (the
launch, the workspace capacity query, the launch-shape query) and a knob in only one of them is
exactly the shape of Defect A: a query that prices bytes the launch never takes, or a launch that
takes bytes the query never priced. It clamps to the rules the plan's own loop enforces (width >= 17,
`splits <= pages / 8`, `splits <= 8`) so a forced leg cannot ask the kernel for a page range that
does not exist. Sweep: one binary, `NINFER_FA2_SPLITS` in {1,2,4,8}, ABBA-paired leg order 1-4-2-8-8-
2-4-1, 4,210-token lighthouse prompt, all eight legs in one lock hold (`/tmp/splits-sweep.sh`), read
as per-leg prefill t/s and generated-text equality.

**Making item (1) A/B-able: `NINFER_FA2_IU8_W32_OFF`.** The diet was landed as an additive
`if constexpr` arm in `prompt_i8_fa2.cuh` (the G64 score block for `!PackedValues && !PackedKeys &&
Warps == 4 && Stages == 1`), so an engine A/B had to diff two source trees, which is what made the
08:36 ABBA uninterpretable: with the suite aborting, nothing separated "the diet is neutral" from
"the legs never ran". Both the wave32 fragment `using`s and that `if constexpr` arm are now guarded by
`#if defined(__HIPCC__) && !defined(NINFER_FA2_IU8_W32_OFF)`, so one tree builds both arms: define the
macro and the block vanishes, leaving the pre-diet `af[GroupKc][4]` path that is already the sm_8x
path. Arm 2 (diet off) is a copy of the working tree with `#define NINFER_FA2_IU8_W32_OFF 1` ahead of
the `.cuh` include in `prompt_i8_fa2.cu` (the only file that includes it); arm 1 (diet on) is the
same tree without it. Both arms carry the Defect A/B fixes, so the legs differ in exactly one thing -
how the G64 score block is issued - and the arm pair is reproducible without a git dance.
The int8-prompt suite run that confirms the fixes is simultaneously the diet's correctness gate: the
FP64 oracle in `tests/ops/softmax_attention/causal_cache.cpp` is what decides whether the wave32
accumulation is exact, and that run is the first that can finish long enough to answer.

**That gate ran, and the diet cleared it.** `/tmp/i8diag.log` (locked 09:28:33 → 09:35:07, 398 s,
`--int8-prompt-only` with the workspace instrumentation plus `AMD_LOG_LEVEL=3`): **0 high-water
mismatch lines, 0 `hipErrorInvalidValue` / `device.h:60` aborts**, 510 cases reached - 504 `ok` case
lines plus the 6 `masked batch row over 33K envelope` coverage markers - and the single `FAIL` is the
known `d256-h16-kv2 exercised no split launch` coverage assertion, printed once per storage. The
instrumentation is what the run was for, and it came back clean: the queried and executed workspace
high-water agree on every case, including the five h24 int8-g64 rows and the `batch B=1 W=65` row that
made Defect A visible, so the `fit_split = plan.splits > 1` launch really does allocate what the plan
priced. The log also shows the descended shape launching as designed -
`__hipPushCallConfiguration({4,24,1},{128,1,1},51200, ...)` on
`causal_attention_prompt_i8_fa2_kernel<CausalAttentionGeometry<24,4,1,41>, PagedKVBatchMetadata<false>,
4, false, false, false, 1>` - the 4-warp 51,200 B arm the 64 KiB ceiling forces.
Because that binary was built from the tree with the diet in it, the same 510 cases double as the
diet's correctness check: every case verifies its output, so the wave32 `wmma_i32_16x16x16_iu8_w32`
score assembly is exact against the FP64 oracle on the shapes it covers. What it does **not** settle
is speed - that is the engine ABBA on the arm pair, which needs no source change to run.

**Keeping the diagnostic prints (decided, not measured).** The high-water *contract* in
`tests/ops/softmax_attention/causal_cache.cpp` - `workspace.used() != 0 ||
workspace.peak_used() != workspace_bytes` → `++failures` - predates this work. What was added is the
*text* of the failure line at all six sites that carry the contract (both `run_a1_case` overloads,
both `run_a3_case` overloads, both `run_batch_case` overloads: 18 lines added, 6 reflowed), printing
`queried=`, `executed_peak=`, `executed_used=`. The call is to keep it. It prints only on a branch
that is already failing the case, so a clean run emits exactly what it emitted before - the 09:28
confirmation log has zero such lines, which is itself the result the run was for. And the bare
`mismatch` sentence cannot separate the two failure shapes: Defect A is `queried > executed_peak`
(the plan priced split bytes for a launch that never split and never allocated), Defect B is
`executed` demand above the 64 KiB ceiling (the small-T arena tier asking KeyBlock = 64). The three
numbers are what located plan-versus-launcher as the split point in the first place, and every future
plan change on this card needs that discrimination back. Cost: one `std::cerr` on a path that has
already failed the case.

**The plan knob landed: `NINFER_FA2_SPLITS` (implementation record - no measurement in it).** The
override is in `causal_prompt_fa2_plan` exactly where the item-(3) analysis put it.
`causal_prompt_fa2_forced_splits()` (`prompt_i8_fa2_plan.h:99-109`) reads the variable once per
process into a function-local `static const`, so the launch (`prompt_i8_fa2.cu:103`), the workspace
capacity query and the launch-shape query (`causal_softmax_attention.cpp:547`, `:638`) cannot disagree
about the split count - the Defect A shape, a query pricing bytes the launch never takes. Parsing is
strict (`strtol` with a full-string check) and the accepted range is `1..kCausalPromptFa2MaxSplits`;
unset, empty, unparseable, and out-of-range all mean "no override", so a typo in an A/B script leaves
the shipped route in force instead of silently probing something else. The forced branch
(`prompt_i8_fa2_plan.h:131-145`) skips the cost model and keeps the plan's route facts: a forced count
above 1 returns the same `{8, splits}` the candidate loop returns for a split launch (the launcher
then does its own LDS descent, so a forced leg measures the *fitted* shape, which is the point of the
sweep), and a forced count of 1 returns the baseline `{width <= 64 ? 8 : 4, 1}`. The gate is
`forced > 0`, not `> 1`: the sweep's `s=1` leg is a true unsplit control, not a leg that falls
through to the model. The clamp walks the count down to the largest feasible one under the loop's own
limits (`width >= kCausalPromptFa2MinSplitWidth`, `pages >= splits * 8`, split partials within
`kCausalPromptFa2SplitBudgetBytes`), so a forced leg can never request a page range that does not
exist.

**Reading the sweep: grid z is the proof, not the env var.** Each leg's route is read from its own
server log under `AMD_LOG_LEVEL=3`, pairing the `hipLaunchKernel ( ptr, {gx,1,gz}, {threads,1,1},
ptr, lds, ... )` line with the `ShaderName :` line naming the FA2 kernel two lines later - z is the
split count that actually launched and the `lds` column names the shape the 64 KiB ceiling fitted.
(A first attempt paired on `__hipPushCallConfiguration` with a no-space pattern and matched nothing:
AMD_LOG prints those lines with spaces, and grid/block/LDS are on the `hipLaunchKernel` line.) A leg
whose table shows only z=1 never split, so a knob that was set is not a knob that took effect, and the
eight-leg table is what makes the t/s numbers mean something. Expect the sweep to separate two
hypotheses that the item-(3) arithmetic cannot tell apart on its own: if `s=1` matches the best forced
count, the plan's split cells are optimistic exactly as the fitted-shape analysis predicts and pricing
splits at `Br = 64`, rate 0.65 is the fix; if a forced count is clearly faster than `s=1`, the model's
*direction* is right and the regret sits in the constants, which is the `:47-57` re-derivation, not
the shape. Measured legs go here when the log exists.

**Splits are structurally unavailable on long prompts on this card - the split sweep's geometry had to
change because of it (found by arithmetic, before any leg ran).** `width` in the plan is `q.ne[2]`
(`prompt_i8_fa2.cu:102`), the query token count, and a prompt is launched as one call at its full
token count, so the split budget is a bound on *query width*, not on keys: the partials are
`256 * q_heads * width * splits` FP32 plus `2 * q_heads * width * splits` more, i.e. about
`24,768 * width * splits` bytes against `kCausalPromptFa2SplitBudgetBytes = 64 MiB`, giving
**width * splits <= ~2,709**. A 4,210-token prompt therefore takes `splits = 1` for every q_heads and
every split count, and so does the 7,000-token cache-filling request below. That is the answer to a
question the item-(3) analysis left open from the server side: it explains the neutral split-fix
server ABBA on the lighthouse prompt (170.8/170.7/170.3/169.4) without any hypothesis about the cost
model - the prompt the ABBA measured can never split, in either tree, whatever the plan decides. It
also says the split lever is not where the 300 t/s long-prompt work is, on this card and at this
capacity; the plan's split cells matter for launches whose width is small and whose keys are long.
The sweep's legs are therefore a **short query over a warm cache**: capacity here is 8,192 int8 KV
tokens (server `capacity | KV` line), so each leg warms ~7,000 tokens (identical in every leg, and
forced to `splits = 1` by the same bound, so cache state entering the probe is identical by
construction) and measures a ~260-token query over ~7,270 keys, where all counts 1-8 are feasible
(49-51 MiB of partials, 114 pages >= 8 * 8). The clamp in the knob is what makes this safe rather
than loud: an unreachable forced count walks down to the largest feasible one, so the leg launches
something the plan priced, never a page range that does not exist. `tests/ops/softmax_attention`
cases at 257/513 tokens over 33K keys reach counts the engine cannot reach at 8,192-token capacity -
the test probes the kernel, the engine probes what is reachable at capacity, and a result from one is
not a result for the other.

**A cache-warmed probe is only reachable if its tail is novel - the sweep's own first-draft bug, caught
before any leg ran.** Engine cache matching is prefix matching: a request's `visible_keys` are the
matched cached prefix and its query width is the unmatched suffix. The first draft built the probe as
the first 8 units of the same generator that built the 215-unit warm body, i.e. the probe was a
**prefix of the warm text**, so the engine absorbs it whole: width ~12, under
`kCausalPromptFa2MinSplitWidth = 17`, which means **no leg splits, in either tree, whatever the plan
decides** - eight legs measuring the same unsplit route, reported as a split sweep. The fixed probe is
the warm text plus a generated tail in *different* sentences: keys ≈ the warm length, width ≈ tail plus
instruction ≈ 275, which is the launch the sweep asks about. The leg prints its own geometry check, and
both fields have to read right: `cache N (%)` near 96% says the probe's keys rode the warm prefix, and
the probe request's `prompt N` says the width that ran. A leg whose `cache` line reads 0% warmed
nothing; a leg whose probe line reads the same `prompt N` as its warm line was absorbed, and its t/s is
a warm-restart number, not a split-leg number. Same class of defect as the v1 width-bound problem: the
probe could not answer the sweep's question in either draft, and both times arithmetic on the code
caught it **before the legs consumed shared-card time**.

**Phase B: legs with no override measure the shipped plan at the width its own pricing picks (the claim
is arithmetic; the legs are facts about a launch).** At width ~1,024 over ~1,024 keys the plan's split
cell is affordable (`1,024 * 2 = 2,048 <= ~2,709`; 16 pages >= 8 * 2) and its cost model *chooses* it:
baseline `div_up(1024,64) * 24 * 0.65 * keys = 15.6 * keys` against split cell
`waves(div_up(1024,128) * 24 * 2) * 1.0 * keys / 2 + (1/24) * 1024 * 2 * 24 = 12 * keys + 2,048`, which
wins for keys > 569. The launcher cannot fit the 8-warp shape that cell priced - Defect A's shape,
generalized to every width over 64 - so what launches is the descended 4-warp grid
`div_up(1024,64) * 24 * 2 = 768` CTAs, costing `waves(768) * 0.65 * keys / 2 = 15.6 * keys`: exactly
what the unsplit route pays, **plus** the merge kernel and its partials traffic. Predicted by the plan's
own constants, then: at the widths where the plan chooses to split, splits cost the baseline's price and
buy a merge - the split cells are net cost, not net gain, at the fitted shape. The `unset` legs
(no `NINFER_FA2_SPLITS` in the environment, same binary) are what test that without a knob: their grid
z **is** the shipped plan's own choice - z=2 means the priced-and-descended split launch happened and
its cost is in the leg's t/s, z=1 means the model's feasibility loop rejected the split on its own.
Paired order `unset-1-2-2-1-unset` (`/tmp/splits-sweep2.sh` phase B, cold 1,024-token request, cache 0
by construction). Measured legs go here when the log exists.

**Reading the diet ABBA (instrument decided before the legs ran).** `/tmp/parse-diet-abba.py` turns
`/tmp/attn-diet-abba.log` into the leg table - per leg: the `req#N done` prefill, the best windowed
prefill, the SHA1 of the generated text, the per-chunk `tokens=... total=...` buckets - and then
`off = mean(off1, off2)`, `on = mean(on1, on2)`, the paired delta, **printed against the off-to-off
spread as the noise yardstick**. That last pair is the point of the script: the card's probe-to-probe
noise floor is ~5% (recorded above), and the two off legs of a paired set are the same binary on the
same prompt, so their spread *is* a measurement of that floor for this set. A delta smaller than the
off-to-off spread is not a result, and text equality is required before any of the numbers mean
anything - a leg that produced different text changed the computation, not the schedule. (Verified on
a synthetic log first; the real legs ran 10:43-10:46 and are read in the "Diet ABBA, measured" section
below. Instrument fix found on the real log: the request line is timestamp-prefixed
(`2026-10-23 10:43:59.441  INFO  req#1 done | ...`), so the parser must match `req#` **inside** the line -
matching the line's start reported `nan` for every leg's request prefill and silently left only the
windowed numbers, which on this server are the 5.0 s buckets (`prefill 204.8 tok/s (1,024 tok)`) and are
**not** the request-level metric.)

**Sweep and ABBA parsing instruments (`/tmp/parse-splits2.py`, `/tmp/parse-diet-abba.py`).** Both
parsers are grouped by **what launched**, not by what the script asked for: the sweep table pools legs
by grid z, so a leg whose forced count clamped (env=8, z=2) joins the z=2 population instead of
inflating a `splits=8` population that never launched. The sweep parser also prints the probe geometry
with each leg - `prompt N`, `cache N (%)`, and their difference as the query width - so a leg whose
probe was absorbed (cache ≈ prompt, width ≈ 12) is visible in the table as the v2 geometry bug, in the
run, instead of being read as a slow split. Both scripts are in `/tmp`, run with `python3 -I`, and
tested against synthetic logs that reproduce the scripts' print format.

**Chain status comes from the log and `pgrep`, never from a completion notification.** Observed today:
the diet-ABBA queue reported "finished (exit code 0)" with a tail line that matched no print in its own
script, while `pgrep` showed `bash /tmp/queue-diet-abba.sh` alive inside its deadline loop, the off arm
still compiling (last line `[off3] configure rc=0`), no `/tmp/ninfer-serve-dietoff`, and no leg logs.
The queue was not finished and nothing had been measured. The same rule in the opposite direction
(earlier this session): an "exit 0" on a `set -e` script that died early. The only statuses that count
are the chain's own sentinel lines (`ALL DONE`, `DIET ABBA SET`, `rc=`) and the binaries/logs they are
accompanied by - which is why every queue script prints a sentinel and every leg prints its proof
lines.

## 2026-10-23 lever board: where the 300 t/s thrust stands, lane by lane

For the next session: what is shipped, what is measured, what is queued, and what is the user's to
decide. Nothing here is newer than the sections it cites; the queue lines describe jobs, not results.

- **KV / GEMM lane - shipped and measured.** G32 panel median 142.6-143 t/s baseline; BF16-KV 146.9 →
  166.75 (+14.4%); INT8-KV 139.8 → 170.0 (+28.7%). cal64b certified the column-split GEMM at −25% on
  its probe, and cal65/cal66 closed it: the probe win was a coverage artifact once the verify gate ran
  the engine kernel on identical buffers. The GEMM lane's bit-exact headroom is the roof item above -
  **≤7% of the budget** - so this lane is not where 300 t/s comes from.
- **Attention lane - the active lane.** Native WMMA QK landed; the two gfx1102 defects (LDS-ceiling
  small-T tier, plan-versus-launcher split pricing) are fixed and suite-verified with the workspace
  instrumentation clean at 510 cases; the register diet (wave32 `iu8` G64 score block) cleared the same
  oracle and its engine ABBA is **measured neutral** (177.0 base vs 176.3 candidate, −0.40% paired,
  inside the set's 1.81% off-to-off spread, text identical on all four legs; keeps −11.9% frame and
  −9.3% code on the served shape - certification, not a win); the plan knob `NINFER_FA2_SPLITS` landed and its sweep v2 is
  **measured flat and inconclusive-by-instrument** (phase B, the usable half: 1,077-token cold legs
  172.95 unset vs 173.10 forced-1 vs 173.35 forced-2, every delta inside the sweep's own 0.46% spread,
  text identical on all 14 legs; phase A failed its geometry check - all eight probes read `cache 0.0%`,
  so they measured a cold 7,329-token prefill; and grid `z` printed zero lines in every leg, so nothing
  proves the override reached a launch). Scoped by the width bound to short-query launches over long
  caches - which is the launch this sweep never produced. This is the lane the
  roof arithmetic points at: 27.1% of prompt time running at ~2% of measured pipe peak, and the only
  term that grows quadratically with context.
- **GDN lane - one win shipped.** cal68's warp-scope barriers: −5.15% on the recurrence kernel,
  committed 75eb1b3d. `gdn_mixer` is 23.2% of prompt time, so this lane has measured value and no
  closure.
- **User-gated, untouched.** iu4 (4-bit activations): certified semantics and 1.72-2.76× on the
  dominant GEMMs, a numerics change, **not bit-exact**, and it ships only on the user's explicit
  decision. Its roof contribution (~230 t/s alone) is recorded above. Same for the gate_up integer-tile
  routing gap: plumbing with a numerics-visible boundary the user has to see first.
- **The target's status, stated plainly.** Bit-exact work as measured lands at ~250-260 t/s; 300
  needs the quadratic attention term to shrink, not just the GEMMs. Prism's 296 t/s on this silicon
  says the ceiling is not the card. The diet ABBA and the split sweep are tonight's attempts at the
  attention term; the width bound already says splits are not a long-prompt lever here, so the diet and
  the mixer's raw efficiency are what the next session should push.

### Occupancy-2 LDS target for the int8 FA2 shape - register gate now measured, LDS still the binding term

The roof arithmetic says the attention mixer runs at ~2% of the measured pipe peak; this is the LDS
arithmetic behind why, and the shape target it implies. Arithmetic only: no code changed, no leg run.
Every number below is verified three ways - a standalone arithmetic program (`/tmp/lds_calc.cpp`), the
shipped `static_assert`s in `prompt_i8_fa2.cuh` (lines 131-138, 153-158, 128-129), and the observed
launch from the suite log (`lds 51200`).

**The decomposition is exact.** Constants: head dim `D = 256` (`prompt_common.cuh:19`), tile width
`Bc = 64`, pinned to the KV page size by `static_assert(kCausalPromptFa2Bc == kPagedKVPageSize)`
(line 128), group 64 ⇒ `Groups = 4` (line 129). The served int8-g64 shape is
`<4 warps, unpacked V, unpacked K, 1 stage>`:

| term | bytes |
|---|---|
| Q tile (Br = 4·16 = 64 rows × 256 codes) | 16,384 |
| K tile (64 keys × 256 codes) | 16,384 |
| V tile (unpacked int8) | 16,384 |
| K scales (64 × 4 groups × 2) | 512 |
| V scales (64 × 4 groups × 2) | 512 |
| Q group scales (1-stage dedicated region, 64 × 4 × 4) | 1,024 |
| **SmemBytes** | **51,200** |

**Zero static LDS.** The kernel declares only `extern __shared__` (`.cuh:271`) - no static
`__shared__` arrays - so the per-workgroup LDS is exactly `SmemBytes`, and the 64 KiB ceiling applies
to it directly. gfx1102 gives a workgroup ≤65,536 B and a CU 65,536 B total, so **2 workgroups per CU
needs ≤32,768 B each**. All three shipped 1-stage shapes are above that: int8-g64 4-warp = 51,200;
packed-values 8-warp = 58,112; packed-values 4-warp = 41,728. Every shape on this card is
**1 workgroup per CU**.

**Measured agent facts (`rocminfo`, CPU-side, no lock, today).** gfx1102 reports **SIMDs per CU = 2**
(not 4 - this die's CU is 2 × 32-wide), **Max Waves Per CU = 32**, **Max Work-item Per CU = 1024**, and
the GROUP segment pool = **64 KB, non-allocatable, granule 0, alignment 0** (the runtime carves LDS per
workgroup; nothing in the agent report forces a rounding penalty, which is a fact about the report, not
proof about a second resident workgroup). So the shipped int8-g64 shape runs **4 warps = 2 waves per
SIMD**, and the packed 8-warp shape runs 4 waves per SIMD. The "2% of peak" reading is therefore not
"one wave per SIMD"; it is 2 waves per SIMD with a 1-stage sequential copy-then-compute loop and no
second resident workgroup whose phases could overlap the first's stalls.

**The register-file budget for 2 WG/CU - measured, and it is NOT the binding term.** 2 workgroups ×
128 threads = 8 waves per CU = **4 waves per SIMD** on this die (2 SIMDs/CU). The SIMD's vector file is
32,768 32-bit VGPRs (128 KB) allocated per wave in granules of 8, so 4 co-resident wave32 waves bound
the kernel at **exactly 256 VGPRs per lane** - the ISA per-lane ceiling, not the 128 an earlier draft
of this section claimed (that draft assumed a 16,384-VGPR SIMD; it was wrong, and the compiler's own
occupancy figure disproves it, below). So **registers permit 2 WG/CU with zero slack, and the served
kernel already sits exactly at that gate** - which is a measured fact, not architecture arithmetic.

Device asm dump, build flags, `-c --save-temps`, gfx1102 (`/tmp/fa2asm/…gfx1102-03f33253.s.tmp`, diet
**on**; the diet-off twin with `-DNINFER_FA2_IU8_W32_OFF` in `/tmp/fa2asm_off/…6c935eb0.s.tmp`). Read
from each kernel's own `.amdhsa_kernel <name>` block (`.amdhsa_next_free_vgpr`,
`.amdhsa_next_free_sgpr`, `.amdhsa_private_segment_fixed_size`) with `/tmp/parse-amdhsa.py` - an
earlier version of that parser paired the `.AMDGPU.csdata` comment blocks with the last `.globl` in
document order and **mislabeled rows** (42 rows for 36 symbols; csdata blocks are not emitted in
`.globl` order), so every number below is from the name-bound blocks, cross-checked against the csdata
fields where labels agreed. Load-bearing rows:

| shape (diet on) | VGPR | SGPR | scratch B/lane | scratch B/wave | static LDS |
|---|---|---|---|---|---|
| **G24/4 4w S0 PV0 PK0 st1 (served int8-g64)** | **256** | 40 | **2,016** | 64,512 | 0 |
| G24/4 4w S1 PV0 PK0 st1 (split variant) | 256 | 52 | 2,016 | 64,512 | 0 |
| G24/4 8w S0/S1 PV0 PK0 st1 | 256 | 38/42 | 2,240 | 71,680 | 0 |
| G24/4 4w S0/S1 PV1 PK1 st1 (packed) | 256 | 37/41 | 2,288 | 73,216 | 0 |
| merge kernel (split reduce) | 10 | 22 | 0 | 0 | 0 |

Every FA2 **math** kernel instantiation on this card - both geometries (G24/4 = group 64, G16/2 =
group 8), 4 and 8 warps, split and unsplit, packed and unpacked - compiles to **256 VGPRs (the ceiling)
with 2,016-2,336 B/lane of private segment (64-75 KB per wave of scratch traffic)**, at compiler
Occupancy 4 (waves/SIMD). Only the merge kernels are light (10 VGPR, 0 scratch, Occupancy 16). The
compiler prints Occupancy 4 for a 256-VGPR wave32 kernel only if it models 32,768 VGPRs per SIMD
(32,768 / (256·32) = 4), which is the second independent confirmation of the file size - and the same
4 waves/SIMD is what a 2-WG/CU workgroup needs.

**What that does to the plan.** (a) 2 WG/CU is register-legal today: 8 waves × 256 × 32 = 65,536 = the
whole 2-SIMD file, so **LDS is the sole binding term** (51,200 × 2 = 102,400 > 65,536). (b) Levers (1)
and (2) below *add* ~40 registers to a kernel that is already at 256 and spilling ~2 KB/lane, so they
cannot be paid for as designed - the allocator has nothing left to give and the added registers become
more scratch, not fewer LDS bytes. (c) The highest-headroom bit-exact lever the dump reveals is
**spill removal itself**: 64-75 KB/wave of global round-trip traffic per workgroup per stage step, on a
kernel that has LDS headroom it is not using. That is measurable with this same tool (scratch → 0 is a
one-number test) before any shape surgery.

**Where the spills live - located, not just counted.** `/tmp/spillscan2.py` reads loop membership off
clang's own annotations (the AMDGPU back end prints it on each block-label line - `.LBB1_596:
; =>This Loop Header: Depth=1`, `.LBB1_602: ;   in Loop: Header=BB1_596 Depth=1` - so every instruction
inherits its block's loop; scratch accesses print as `scratch_load/store_* …, off offset:N`; kernels are
delimited by `.section .text.<mangled>`, and the emitted symbol `.LBB1_596` and the comment's
`Header=BB1_596` are the *same* loop and must be canonicalized or one loop splits into two rows).
Full-file run: `/tmp/spillscan-full2.txt`. Served shape: **1,670 scratch instructions of 21,765 (7.7%)**,
and **686 of them - 41% of the kernel's entire scratch traffic - sit in one loop**, header `1_596`
(Depth 1, 129 blocks, 3,789 instructions), which is **all loads, zero stores**: 686 loads (563 `b32` +
123 `b64`) into the frame's upper band, **offsets 608..1988**, 269 distinct constant offsets against base
`off`. Per lane per tile iteration that is **3,236 B/lane = 101 KB/wave of VRAM reads**. The loop's
identity comes from its mix - `v_cndmask_b32_e64×808, scratch_load_b32×563, s_waitcnt×480,
s_delay_alu×345, ds_store_b8×239, s_clause×171, s_and_saveexec_b32×129` - the per-tile K/V staging loop
with the causal peel/predication. The 984 out-of-loop scratch ops (386 distinct store offsets, 0..2000)
are the prologue filling the frame.

**What it reloads.** `/tmp/spilltrace.py` names each slot's producer (nearest earlier instruction
writing the stored registers; dual-issue `::` lines split, indentation stripped, `v[23:24]` range
destinations accepted - a parser that misses any of the three returns `?` for everything, as an early
draft did):
- **123 of the 686 reads are a single `b64` slot, offset 760**, filled once in the prologue by
  `scratch_store_b64 off, v[23:24] ; 8-byte Folded Spill`, whose producer pair is
  `v_dual_mov_b32 v24, 0 :: v_dual_and_b32 v5, 15, v127` - the value is `(lane & 15, 0)`: two ALU ops
  from the lane id, loop-invariant, re-read from VRAM 123× per lane per tile iteration.
- Slots **776..804** (8 consecutive words, 28 reads each = 224 of the 686) are each written once by
  `v_add_nc_u32 v0, vX, vY` - eight loop-invariant address/index sums.
- Producer census over all 686: **vector ALU 279, `ds_bpermute_b32` 220, lane-id-derived 131,
  `ds_load_b32` 56**; **369 of 686 (54%) read slots written exactly once outside the loop**, i.e. loop
  -invariant. Slots 856..884 (7 reads each) trace to `ds_load_b32` at LDS offsets 50176..50188 - the
  scale region (Q/K/V tiles occupy 0..49152, K-scale 49152..49664, V-scale 49664..50176, Q-scale
  50176..51200, matching the 51,200 pinned above) - real data, not coordinates.

**Consequence.** Most of the served kernel's scratch traffic is **not live data**: it is loop-invariant
lane/coordinate/index terms the allocator pushed to VRAM at the 256-VGPR ceiling and re-fetches every
tile iteration, and every reload drags a `s_waitcnt` with it (480 in that loop). Rematerializing a term
is ~2 ALU ops and 2 registers, and the loop already issues 808 predication `v_cndmask`s, so ALU slots
are not the scarce resource there. Measurable test for any diet attempt, same tool: **scratch loads in
loop `1_596` → 0** (686 today), with 2,016 B/lane as the coarse companion. The concentration is
shape-specific, which narrows the target: **41% of scratch inside that one loop for the 4-warp unpacked
rows, 14% for 8-warp rows, 8-13% for packed-values rows, 0 for merge kernels** - the diet's target
shape is the one that carries the reloads. Caveats: device asm carries no `.loc`/`.file` directives, so
attribution is to loops and instruction mixes, never source lines; all counts are static, per lane, per
iteration - no dynamic profiling was run.

**The same trace on the diet-off twin: the diet moves reloads around, it does not remove them.** Run
identically on `/tmp/fa2asm_off/prompt_i8_fa2-hip-amdgcn-amd-amdhsa-gfx1102.s` (the diet-off dump is
`…gfx1102.s`, regenerated with `-save-temps=cwd`; a `.s.tmp` name for this arm is stale). Served shape,
off arm: **1,744 scratch of 25,345 instructions (6.9%)**, **375 in loop `1_596` = 22%**, same loop header
as the diet-on arm and the same fingerprint - identical `ds_store_b8×239` in the mix, 136 vs 129 blocks -
so the two dumps name the same loop, not two look-alikes. The off arm's hot loop is **all loads, 24
distinct offsets, 780..2264, 1,992 B/lane/iteration (62 KB/wave-iter)**, and **371 of 375 reads (99%) hit
slots written exactly once** (20 of its 24 read slots are write-once) - almost the entire hot loop is
loop-invariant reload.

Lane attribution needed fixing before that number was trustworthy. The lane-id register is **dump-relative**
- clang calls it `v127` in the diet-on dump and `v124` in the diet-off twin - so a hardcoded register name
reports 0% lane-derived on one arm. The robust rule: on GCN, **v0-v2 arrive initialized with the workitem
thread ID**, and each served kernel's *first* body instruction is `v_mov_b32_e32 v127, v0` (on) /
`v_mov_b32_e32 v124, v0` (off), 11 lines into the section; every lane-shaped term reads that copy.
`/tmp/spilltrace.py` now seeds the lane set from v0-v2, propagates it through `v_mov`/`v_dual_mov`, adds
`v_mbcnt` destinations, and **drops a register on any other write** (registers get reused, so a register's
lane-ness is per program point, and dual-issue halves are applied against the state before the line).
Corrected shares: **off 126 of 375 (34%) lane-derived**, census `vector ALU 249, LANE-derived 126`;
**on 131 of 686 (19%)**, census unchanged (`vector ALU 279, ds_bpermute_b32 220, LANE-derived 131,
ds_load_b32 56`).

Side by side, served shape (`G24/4:4w S0 PV0 PK0 st1`):

| | diet-off | diet-on |
|---|---|---|
| private segment | 2,288 B/lane | 2,016 B/lane (**−11.9%**) |
| scratch instructions | 1,744 of 25,345 (6.9%) | 1,670 of 21,765 (7.7%) |
| scratch inside loop `1_596` | 375 (**22%**) | 686 (**41%**) |
| hot-loop bytes/iteration | 1,992 B/lane (62 KB/wave) | 3,236 B/lane (101 KB/wave) |
| loop-invariant share of hot reads | 371/375 (99%) | 369/686 (54%) |
| lane-derived share of hot reads | 126/375 (34%) | 131/686 (19%) |

So the diet cuts the frame and total scratch ops while **raising hot-loop reloads by +83% (375 → 686) and
hot-loop bytes per tile iteration by +62% (1,992 → 3,236)**. Two caveats keep that honest: the diet's hot
loop is also 2.15× the instructions (3,789 vs 1,761), so shares are taken over different denominators; and
the diet's added reloads are specifically the shuffle/LDS-fed terms - the 220 `ds_bpermute_b32` and 56
`ds_load_b32` producers appear **only** in the diet-on census, the off arm's hot reads are pure vector ALU
plus lane-derived. The single hottest reload is the same artifact in both arms: a `b64` pair written once in
the prologue and re-read **123× per lane per tile iteration** (on: slot 760, `(lane & 15, 0)`; off: slot
780, from `v_subrev_nc_u32_e32 v25, 64, v124`, i.e. `64 − tid`). A lane-derived pair that costs 2 ALU ops to
rematerialize, reloaded 123 times per tile iteration in *both* builds, is allocator behavior at the
256-VGPR ceiling and not diet-induced - which lowers the prior that the diet alone wins the ABBA, and makes
"scratch loads in loop `1_596` → 0" the shared target for both arms.

In issue-slot terms, arithmetic on those counts (static, no dynamic profiling): each scratch access is one
32-lane wavefront request, so the hot loop issues **686 requests per wave-tile-iteration in the diet-on
build against 375 in diet-off (+311, 39.8 KB extra read per wave-iteration)**. Measured against the loops'
own sizes - 3,789 vs 1,761 wavefront instructions - the **reload density is 18.1% diet-on vs 21.3%
diet-off**: the diet's hot loop does more reloads per iteration, at a *lower* reload-per-instruction
density, with `s_waitcnt` in 480 of 686 loads (on) and 252 of 375 (off). So the diet's ABBA risk is stall
behavior under the tight waits, not LSU issue pressure, and the ratio is 311 requests against a
3,789-instruction body - a per-tile-iteration quantity whose token-level cost needs the loop's trip count,
which is dynamic and was not measured here.

**The two bit-exact levers, and where the bytes land.**

1. **Q out of LDS into registers.** Br = 64 rows × 256 codes = 16,384 codes over 128 lanes = 128
   codes/lane = 32 `b32` registers, plus 8 fp32 group scales/lane. Removes Q (16,384) and the 1-stage
   Q-scale region (1,024): **51,200 → 33,792, which misses the 2-WG bound by 1,024 B.**
2. **K/V scales out of LDS into registers.** Each warp touches 16 keys × 4 groups per stage = 64 fp16
   scales = 2 regs/lane. Removes exactly 1,024. With (1): **32,768 B - exactly the 2-WG bound, zero
   slack ⇒ 8 warps/CU from 2 workgroups.**
3. (Bigger, costlier) **Bc = 32** halves the stage to 16,896; with (1) that is 3 workgroups = 12
   warps/CU. Pinned by the page-size assert: it is a page-size change (cache layout, materialize,
   allocator) or a tile/page decoupling in the kernel - a much wider blast radius than (1)+(2).

**The other half of the same shape: the exposed copy.** The 1-stage descent is not just narrow LDS,
it is a *sequential* loop - copy the page's K/V tiles from global, then compute, with no second stage
to overlap (source comment lines 99-101: 2-stage exists on sm_8x only because the 2-stage shapes exceed
this card's ceiling). Occupancy 2 hides that latency *between* workgroups. The occupancy-1 alternative
is a **register prefetch stage**: load the next tile into registers while the current tile computes,
then write LDS. Arithmetic: one tile (K) is 16,384 B over 128 lanes = 128 B/lane = **32 `b32`
registers**; K+V both = 64. The measured baseline decides this one too, and it says the same thing:
the kernel is already at 256 with 2,016 B/lane spilled, so at 1 WG/CU (2 waves/SIMD here, gate 512 ⇒
256 real, ceiling) the 64 registers exist **only by displacing spills** - which is the same work as the
spill diet, paid in the same currency - and at 2 WG/CU the gate is 256 with zero slack, so the prefetch
stage cannot be added there either. Both occupancy routes are therefore register-limited by the *same*
fact: the kernel's live set. Fix the live set (spill diet first, measurably) and the prefetch stage
becomes affordable out of the registers the diet frees; skip that step and neither lever is fundable.
Bit-exact by construction (it moves
bytes, it changes no arithmetic), so it belongs to the same class as the diet: build behind a
kill-switch, oracle-check, ABBA.

**Non-levers, named so nobody re-derives them.** Halving the V tile (PackedValues, 25,344-byte stage,
4-warp Q-in-regs = 2 WG/CU) is a KV **storage** format (int4 V with group-128 scales), not a kernel
knob - on the served int8-g64 cache it means re-quantizing V, i.e. a numerics change in the iu4 class
that ships only on the user's decision. Two stages stay dead on this card: unpacked K+V alone are
67,584 and with Q 83,968 > 65,536 (the descent comment at lines 140-148).

**What must be measured before any of this is code.**
- The score block gets its WMMA A fragment from `ldmatrix` out of LDS today; a register-filled Q
  fragment needs the per-lane fragment layout proven, not assumed (fragment-layout probes exist as
  precedent).
- ~~2 WG/CU ⇒ ≤128 VGPRs/lane, so (1) is affordable only if the kernel sits under ~90~~ **answered by
  measurement: the gate is 256/lane and the kernel is at 256** - so (1) and (2) are not fundable as
  designed (see the register-file budget above). The dump's answer also carries the diet's own cost,
  which was part of the question: **the diet's `iu8` W32 path touches only the 4-warp unpacked rows**
  (served shape 2,016 B/lane diet-on vs 2,288 diet-off, −11.9%, and 123,192 vs 135,840 B of code,
  −9.3%), while **every 8-warp row and every packed (PV1/PK1) row is byte-identical between the arms**
  (2,240 B/lane, same code length). So the diet is register-and-code-cheap exactly on the shape the
  ABBA is measuring, and neutral everywhere else - a favorable prior for the leg that was queued. That
  leg ran and the numbers decided (see "Diet ABBA, measured"): −0.40% paired, inside the set's own
  1.81% off-to-off spread, text equal on all four legs.
- 2 × 32,768 = 65,536 with **no slack**: ROCr's per-workgroup LDS rounding/alignment for a second
  resident workgroup is a driver fact to measure (device-asm metadata + a 2-CPU occupancy probe), not
  to assume. The LDS-fit law so far only ever proves things *don't* fit (Defect A's 84,928 abort).

**Diet ABBA, measured 2026-10-23 10:43-10:46 (`/tmp/attn-diet-abba.log`, one lock hold, four legs
off1/on1/on2/off2, cold server boot per leg, 4,210-token lighthouse prompt, `max_tokens=1`, KV bf16
8,192 tokens, cache 0 (0.0%)).** Candidate `/tmp/ninfer-serve-dieton` (the landed `iu8` W32 G64 score
path) against base `/tmp/ninfer-serve-dietoff` (`-DNINFER_FA2_IU8_W32_OFF`), same tree one macro apart:

| leg | binary | request prefill | TTFT |
|---|---|---|---|
| off1 | diet-off | **178.6 t/s** | 23.6 s |
| on1 | diet-on | 176.6 t/s | 23.9 s |
| on2 | diet-on | 176.0 t/s | 23.9 s |
| off2 | diet-off | 175.4 t/s | 24.0 s |

off mean **177.0** (spread 3.20 = 1.81% of mean, this set's own noise yardstick), on mean **176.3**,
paired delta **−0.40%**. The delta is inside the off-to-off spread, so it is not a result - the diet is
**not measurable end-to-end on this prompt**. Text was identical on all four legs (SHA1 `a24ae9faf6`,
`"We"`), so the legs compared schedules, not computations: the diet is bit-exact at the served path, as
its class claims. The point estimate's sign matches the static prediction above (the diet's hot loop
carries +311 more scratch requests per wave-tile-iteration than the base's), at ~4× smaller magnitude
than the noise floor, so the honest reading is the certification reading: the candidate is **neutral
end-to-end while keeping its frame and code savings** (−11.9% private segment, −9.3% code on the served
shape), not a t/s win. The diet stays as landed; it is not the lever that moves the mixer, and the
attention lane's next bit-exact target is the shared one named above - **scratch loads in loop `1_596` →
0** - which applies to both arms (the hottest reload, the prologue-written `(lane & 15, 0)`-shaped
`b64` pair re-read 123× per lane per tile iteration, is present in both builds).

**Splits sweep v2, measured 2026-10-23 10:51-11:07 (`/tmp/knobs-sweep.log`, `[knobs] LOCK ACQUIRED
10:51:12` → `LOCK RELEASED 11:07:35`, one 16.4-min hold, 14 server legs, `=== SPLIT SWEEP v2 DONE
11:07:35 ===`).** One binary (the knobs rebuild finished `build rc=0 10:51:12`), so this is a config
sweep, not a build A/B. Text: **all 14 legs SHA1 `a24ae9faf6` = `"We"`** - the same hash the diet ABBA
printed, so the sweep compared routes, not computations.

**Phase A failed its own geometry check on all eight legs, so its numbers answer nothing.** The design
required two fields to read right: `cache N (%)` near 96% (the probe rode the warm prefix) and a probe
`prompt N` well under the warm's. Measured: every probe request reads **`cache 0 (0.0%)`, `prompt
7,329`** - the warm request (7,042) cached nothing the probe reused, so the flag the design named ("a leg
whose `cache` line reads 0% warmed nothing") fired, and what ran was a full cold 7,329-token prefill, not
the short-query-over-long-cache launch the width bound scopes. The pooled leg table (s1 155.4, s2 154.75,
s4 154.75, s8 154.6; 0.5% spread) is a cold-prefill measurement of a 7,329-token prompt; it is the third
draft of the same probe-geometry defect and is not evidence about splits.

**Phase B is the usable half, and it is flat.** Cold 1,077-token request, cache 0 by construction, paired
`unset-1-2-2-1-unset`:

| legs | forced splits | request prefill | mean |
|---|---|---|---|
| B1, B6 | unset (plan's own choice) | 173.0 / 172.9 | **172.95** (spread 0.10 = 0.06%) |
| B2, B5 | 1 | 172.9 / 173.3 | 173.10 (+0.09%) |
| B3, B4 | 2 | 173.7 / 173.0 | 173.35 (+0.23%) |

Whole-sweep spread 172.9-173.7 = 0.46%; every paired delta sits inside it, and inside the ~5% probe-to-
probe floor recorded above. **Reading: at this width the split count does not move the leg** - forcing 1
(the route the width bound says the launcher descends to anyway) is indistinguishable from the plan's own
choice, and 2 buys nothing measurable.

**Both phases share one instrument gap, and it bounds what the flatness can mean: grid `z` was never
observed.** The log's `--- prompt_i8_fa2 launch shapes (grid z = splits) ---` section printed **zero
lines on all 14 legs**, so the parser reports `launched z=[]` for every leg and pools by the env tag - by
what the script asked for, not by what launched, which is exactly the grouping rule this doc requires of
a sweep. A `z=2` launch is the launch the sweep asks about (the plan's own `prompt_i8_fa2_plan.h:131-144`
forces `{8, splits}` and the launcher descends its grid to 4 warps, so `z` is the split count); with no
`z` printed, the flat table cannot distinguish "splits don't matter at this width" from "the override
never reached the launch". **No conclusion about the launcher is available from this run.** The next split
leg prints its own grid `z` and page/width record before any timing is read; a leg that cannot show its
`z` is not a split leg.

**Two instrument traps recorded from this run.** (1) The sweep's text extraction pipes a JSON response
file into `python3` as *source*, so every log text section is a traceback (`NameError: name 'null' is not
defined`) - text equality had to be read out-of-band from the raw response files; fix the sweep to parse
JSON as JSON. (2) One 16.4-min hold covered all 14 legs. It is defensible as one measurement phase, but
phase A and phase B are independent measurement sets (only phase B's paired order needs one hold), so a
rerun releases between phases and takes the lock per phase.

Status: design target recorded, arithmetic pinned, **the register/spill facts measured** (device asm,
both diet arms, served shape + split + 8-warp + packed, both geometries) **and the spills located to a
named loop with their producers** (686 of 1,670 scratch ops in loop `1_596`, all loads, 54% of them
reloads of slots written once outside the loop, hottest a `(lane & 15, 0)` pair read 123×/iteration),
**and the diet's engine ABBA run**: neutral (−0.40%, inside the set's 1.81% spread, text equal) - the
diet stays landed with its −11.9% frame / −9.3% code and no t/s gain. Register gate for 2 WG/CU =
256/lane, kernel at 256 with 2,016 B/lane scratch ⇒ LDS is the sole binding term, levers (1)/(2) are
unfundable as designed, and the ordered next steps are: (i) the spill diet as a measured lever of its
own - the metric is now **scratch loads in loop `1_596` (686 → 0)**, with scratch B/lane as the coarse
companion, and it is an **allocator** target, not the iu8 kill-switch (the 123×-reload sits in both
arms); (ii) lever (3) `Bc = 32` as the only LDS-shape change consistent with the register facts;
(iii) the split sweep v2 - **run, and closed as instrument-bound, not as a result**: phase B flat
(172.95 / 173.10 / 173.35 for unset / 1 / 2, inside the sweep's own 0.46% spread, text equal on all 14
legs), phase A void (probes read `cache 0.0%`), and **grid `z` unobserved in every leg**, so the sweep
cannot say whether the override reached a launch. Re-running it is only worth the card time **after** a
leg can print its own `z`; see "Splits sweep v2, measured".
(iv) the diet's own named producer census (see "Naming the spilled values"): the diet-on hot loop's 686
reads split **286 vector ALU / 220 `ds_bpermute_b32` / 124 lane-derived / 56 `ds_load_b32`**, and the
`ds_bpermute` class exists **only in the diet arm** - it is the `qk_softmax` fragment conversion
(`prompt_i8_fa2.cuh:602-614`), which shuffles all 8 accumulator elements of all 4 tiles (32 live
registers per tile per group) and consumes 8 of them. Shrinking that fan-out is bit-exact, is the same
kill-switch class as the diet, and is **gated on the accumulator's per-lane ownership map being proven
by a probe** - a naive "shuffle only element `j`" rewrite is not equivalent, because `__shfl` evaluates
its operand at the *source* lane and `j = 2 * lid` is dynamic, so the source lane's `j` is not the
destination's and the rewrite silently changes values. No probe in the tree covers
`wmma_i32_16x16x16_iu8_w32` (checked: `tests/test_mma_ldmatrix.cu` does not mention WMMA/shfl, and the
builtin appears only in `t2_prefill_wmma.cuh` and `prompt_i8_fa2.cuh`), so the diet's conversion is
certified end-to-end (text equality) but its layout is **assumed, not proven** - the probe is the gate.

**2026-10-23: the gate is open, and the lever it gates does not exist.** Two things, one of them a
correction to the paragraph above.

*The map is proven, by a probe that was in the tree all along.* The "no probe in the tree" check scoped
itself to `tests/`; the probe battery `.qual/wmma/cal16{c,e,f}.cu` **does** cover
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (δ-sweep by construction: one byte of one operand set to
`0x10`, other operand all-ones, 512 probes per side, printing every lit `(lane, word)` cell -
`cal16c.cu`, 10/10 battery, run 2026-10-21, recorded above ("2026-10-21 verification") and in
`decode-30tps-plan.md:285-291`). Pinned there: `C(lane l, word w) = dot(Awin[(l<16?0:17)+2w], Bwin[l])`,
with the hardware's operand-role swap (`Bwin` slot l = A row l; `Awin` window = B column), so the product
lands at `(A·B)(l&15, (l>=16?8:0)+w)`. In the diet's operand roles (first operand = keys, second =
query) it reads, for the diet's fragment `dots`:

```
dots[l][w] = score(query = row_base + (l & 15), key = (l >= 16 ? 8 : 0) + w)
```

which is exactly what the diet's own comment claims ("native [query row][8 keys] ownership",
`prompt_i8_fa2.cuh:600-614`) and what both consumers assume: `row0 = row_base + gid` /
`row1 = row_base + gid + 8` with `q_scale_r[0/1]` (`:528-539`) pair with the fan-out's sources
`gid, gid+8, gid+16, gid+24` (queries `gid`, `gid+8`; keys `0..7`, `8..15`), and the diet-off arm's
target layout is the same `keya = nt*8 + 2*lid` × `{row0, row1}` assignment the `mma_s8` arm produces
(`:631-662`). So the diet's conversion was never assumed - it was derivable from a measured map, and the
derivation agrees with it. **Item (iv)'s gate is closed with no new probe needed.**

*The derivation also kills the lever.* With the map, the fan-out's demand is explicit: per `nt16` each
lane `(gid, lid)` needs exactly 8 of the 32 accumulator values - words `{2*lid, 2*lid+1}` from source
lanes `{gid, gid+8, gid+16, gid+24}` (the shipped 32 shuffles fetch all 8 words from those 4 lanes and
keep those 8). A shuffle instruction delivers a **lane-uniform slot index** - `__shfl(v, src)` evaluates
`v` at the source lane, and the slot a loop indexes is fixed per instruction; the per-lane freedom is
the *source* lane, not the word. So a round can only collect, per instruction, one fixed word from each
lane's chosen source. The needed word index is `2*lid + {0,1}`: lane-dependent, 4 ways. The sources'
`src & 3 = gid & 3`, never the consumer's `lid`, so the source-lane freedom cannot absorb the
lane-dependence either. Conclusion (static derivation from the measured map, not a measurement):
**the 32-shuffle fan-out is at its floor for shuffle-only re-indexing** - the 8-value floor is reachable
only by lane-dependent slot addressing, which is the dynamic-index pattern that produces the very
spills this lever was meant to remove. A naive shrink is not just non-equivalent (as recorded above),
it is unavailable. The 24 dead shuffles are live registers, and the register allocator is keeping them
because it cannot re-address the accumulator; that is (i)'s allocator problem, not a shfl-count problem.

**Probe run, 2026-10-23 (`cal72`, `.qual/wmma/cal72.cu`, gfx1102, 64 random-i8 seeds, 2 tiles, wave32,
lock held for the single launch).** The probe reproduces the diet's loads verbatim (swizzled LDS rows at
`(row*DB16 + swz(row, col))*2`, the 4-chunk `iu8` chain with `input_c = false`, the shipped
`rows0..3` / `j = 2*lid` fan-out) and compares the 8 consumer values per lane against an independent
host-side dot product, with bytes 64..127 of every row filled with random junk so an over-read shows as
a mismatch rather than a pass:

```
v0 (shipped addressing) vs host reference from the derived map: 0 mismatches  -> MAP PROVEN
v1/v2/v3 (lane-congruent addressing candidates) vs shipped: 32640, 32640, 32256 of 32768  -> not equivalent
```

32,768 comparisons, **0 mismatches** for the shipped addressing: the map
`dots[l][w] = score(query = row_base + (l&15), key = (l>=16?8:0)+w)` is proven *in the diet's own
addressing*, not just in cal16's synthetic shape. The diet's conversion is now certified by construction,
not only end-to-end.

**All three addressing candidates fail, on all 32 lanes** - the layout-change version of the lever is
dead exactly where the shuffle argument says it must be: no per-lane row permutation puts the 8 needed
values into lane-uniform slots, because the consumer's word demand is `2*lid + {0,1}` (4-way lane
dependent) while the slot a shuffle reads is fixed per instruction, and the sources' `lane & 3` is
`gid & 3`, never the consumer's `lid`. Item (iv) is therefore **closed as measured, with no lever
available**: the 32-shuffle fan-out is at its floor, its 24 dead shuffles are register-allocator
artifacts, and the lever that can act on them is (i) - the allocator target (scratch loads in loop
`1_596`, 686 → 0), not a shfl-count rewrite. No code change from this item.

**Fan-out A/B, 2026-10-23 (`cal73` + a same-tree static pair).** Item (iv)'s floor argument is about
*shuffle count*, and it holds: the packed variant measures **the same 32 shuffles per `nt16`** as the
shipped fan-out (`4 rows × 8 dots` vs `2 halves × 2 groups × 8 dots`), so it was never a shuffle-count
lever. What it changes is the *indexing*: the shipped fan-out writes 4 row arrays and reads them at the
dynamic index `j = 2 * lid`, which is the pattern item (iv) named as the spill producer. So the packed
tree was built to test the indexing, bit-exactly.

*Bit-exactness (probe, GPU, lock held for one launch, `.qual/wmma/cal73.cu`):*
`packed-select vs shipped fan-out: 0/32768 mismatches (0/32 lanes affected) -> BIT-EXACT`, and
`vs host reference … 0 mismatches`. (The first run of this probe was half-wrong: the output slot formula
was written `packed[4r + 2h]`; the fan-out's slots are h-major, `packed[2r + 4h]`.)

*A correction first - the earlier pair in `/tmp/attn-asm` is not a fan-out A/B.* That driver compares
HEAD against the working tree, and **HEAD has no diet path at all** (no `NINFER_FA2_IU8_W32_OFF` gate, no
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` call in `HEAD:…/prompt_i8_fa2.cuh`). Its dumps therefore
measure *diet OFF* against *diet ON packed*, which is a **path switch**: the base TU's 114 sections all
carry `v_dot4_i32_iu8 ×512` with **zero** `v_wmma`, the cand's diet sections carry `v_wmma ×64` with zero
dot4. The hot-loop numbers read out of that pair (base `3_596` 1761 instrs / 375 loads vs cand `1_596`
3178 / 685, +360 cndmasks) are **two different loops in two code paths** and settled nothing about the
fan-out. Instrument rule, recorded the hard way: *an arm pair is a lever test only if both arms emit the
same lowering class in the kernel under test* - check the mnemonic census (`v_wmma` / `v_dot4` per
section) before reading any loop number.

*The correct pair* (`/tmp/pack-asm-cost.sh`: same working tree, same flags, `-save-temps=obj`,
gfx1102, **compile-only, no GPU, no lock**; the single variable is `-DNINFER_FA2_QK_PACK_OFF`). Arm F =
shipped fan-out, arm P = packed select tree. Census: **both arms diet-ON** - 156 sections, `v_wmma` 768,
`v_dot4` 30720, `ds_bpermute_b32` 86760, identical totals.

Measured, per kernel (`/tmp/kernmeta.py`, pairing by mangled name; 78 kernels):

```
non-diet kernels byte-identical between arms:  66/66
diet kernels with fewer instructions:          12/12   (-8635 instructions total; range -693..-743)
diet vgpr 256 -> 256   ds_bpermute 1341 -> 1341   v_wmma 64 -> 64   priv 2016 -> 2032 B
diet scratch loads -32 total, scratch stores +70 total
```

Served shape (`G24/4` + `PagedKVDirectMetadata`, tail `Li4ELb0ELb0ELb0ELi1E`), whole kernel
21762 → 21021 instructions, and the hot loop, at **verified equal work** (both bodies carry
`ds_store_b8:239`, `global_load:1` → same unroll):

| served hot loop `1_596` | F (shipped fan-out) | P (packed) |
|---|---|---|
| instructions | 3789 | **3178** (−16.1%) |
| folded reloads | 274 | **158** (−42%) |
| `v_cndmask` | 812 | 364 |
| scratch loads | 686 | 685 |
| scratch stores | 0 | 0 |
| hottest scratch offset | `0x2f8`: 123 | `0x2f4`: 123 |

Read plainly: the packed fan-out **removes 611 instructions and 116 folded reloads per iteration of the
served hot loop at identical shuffle, LDS and WMMA traffic, with no VGPR change** (256 → 256; the
`s*gpr` move is `40 → 48` in `S0` and `52 → 47` in `S1`, private segment +16 B). The allocator's hot
site is untouched: the same single offset is reloaded 123× in both arms and the loop's scratch loads go
**686 → 685**, so **lever (i)'s target (686 → 0) is NOT reached by this change** - item (iv)'s verdict
stands, and (i) remains an allocator problem.

**Server ABBA, 2026-10-23 (`/tmp/pack-abba.log`, F-P-P-F, one GPU-lock hold, 4,210-token lighthouse
prompt, BF16 KV, `max_tokens=1`):** fan-out 178.4 / 175.6 = **177.0 tok/s**; packed 176.7 / 176.0 =
**176.35 tok/s**. All four responses were `"We"`. The packed point estimate is −0.37%, while the
fan-out arm alone spans 2.8 tok/s (1.6%); it is not a performance result and certainly not a win.
Keep the original fan-out as production default; the packed tree is retained only behind
`NINFER_FA2_QK_PACK_ON` for a future isolated experiment. This closes the fan-out lever.

The two binaries were built from the same source at 13:27 with the single compile-time selection;
the dynamic default is the original form. `iu4` remains a user-gated numerics change, untouched.

### Lever (3) `Bc = 32`: the sites are now named, and it buys zero occupancy on its own (static, no GPU)

Verified in source, not assumed. `kCausalPromptFa2Bc = 64` (`prompt_i8_fa2.cuh:79`) is pinned by
`static_assert(kCausalPromptFa2Bc == kPagedKVPageSize)` (line 128), and `kPagedKVPageSize = 64` is a
`inline constexpr` in `src/core/paged_kv_cache.h:22`. The tile == page identity is load-bearing in the
copy path, all inside `issue_tile`:

- `prompt_i8_fa2.cuh:314` - `key_blocks = max_query_abs / Bc + 1` (a page count derived from `Bc`);
- `:337` - `const int page = block_table[kb]` (one tile index is one page index);
- `:338` - `valid = min(Bc, max_query_abs + 1 - kb * Bc)` (the causal tail is page-expressed);
- `:361-399` - both code planes are addressed from `kv_cache_int8_quant_code_index(page, kv_head, 0, 0)`,
  i.e. the in-tile offset is a *within-page* byte offset with no page term;
- `:401-418` - the scale rows are loaded by `tid < Bc` / `tid < 2 * Bc` against the same `page`.

So `Bc = 32` is not a constant swap: `kb` stops being a page index and every one of those five sites
needs `page = block_table[kb >> 1]` plus a half-page intra-offset term, and the `SmemBytes`
static_asserts (lines 131-138) must be recomputed. That is exactly the "tile/page decoupling in the
kernel" branch of the lever-(3) entry above; the *page-size* branch (cache layout, materialize,
allocator) is wider and is not needed for the kernel-side version, since a 32-key tile is a
contiguous half of a page in both planes.

**The occupancy arithmetic, which is the part that decides whether to pay for it.** In the served
shape (4 warps, unpacked, 1 stage) `Bc = 32` gives `Stage::Bytes = 8192 + 8192 + 256 + 256 = 16,896`
and `SmemBytes = 16384 + 16896 + 1024 = 34,304`. Two workgroups need 68,608 B against the 65,536 B
per-group ceiling this card reports, so **`Bc = 32` alone stays at 1 WG/CU and buys nothing**: the
3 WG/CU figure quoted in lever (3) is `Bc = 32` *plus* lever (1) (Q and Q-scales out of LDS:
16,896 × 3 = 50,688 ≤ 65,536). Lever (1) is already recorded as unfundable as designed at the 256-VGPR
gate. Order of dependency therefore inverts what the lever list reads as: (3) is gated *behind* (1),
not beside it - spending any card time on the page/tile decoupling only pays off after the live set is
demonstrably shrunk. Recorded so nobody re-derives the five sites.

### Lever (i) candidate, named from source: the diet's Q fragment loads are loop-invariant (static, no GPU, nothing measured)

Item (iv)'s result above says the fan-out lever leaves the allocator's hot site alone, so (i) needs its
own target. The source names one. In the diet QK loop (`prompt_i8_fa2.cuh:586-599`) the Q fragment is
loaded per `(grp, nt16, kk)` at

```
&k_b16[key_row * DB16 + causal_prompt_swz(key_row, col)]     // keys: depends on nt16 - varies
&q_b16[query_row * DB16 + causal_prompt_swz(query_row, col)]  // Q:   query_row = row_base + (lane & 15)
```

with `col = grp * 32 + kk * 8`. The Q address depends on **neither `nt16` nor `kb`**: Q is encoded once
per launch by the CTA-wide rotation (`:464-510`), published by the `__syncthreads()` at `:515`, and
never rewritten - which is exactly why the per-row group scales are already hoisted to registers right
after that barrier (`q_scale_r`, `:535-540`). So per tile iteration the loop redoes `Groups × (Bc/16) ×
4 = 4 × 2 × 4 = 32` identical LDS reads, from **16 distinct addresses per lane** (`Groups × 4`, 16 B
each), each carrying the swizzle address math. The hottest spilled pair in the served hot loop -
`(lane & 15, 0)`, slot `0x2f8` in the fan-out arm and `0x2f4` in the packed arm, 123 reloads per
iteration in both - is that address.

The candidate hoists those 16 fragments to registers at the `q_scale_r` point and consumes them in
place of the load: same values, same order, same 64 `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` calls,
so bit-exact by construction. Gated behind `-DNINFER_FA2_QFRAG_REG` *and* the diet's own
`if constexpr (!PackedValues && !PackedKeys && Warps == 4 && Stages == 1)`, so the shipped tree is
inert without the define and the 66 non-diet kernels cannot change.

**The trade, stated before measuring it, because it is a trade at a hard gate:** 16 fragments × 16 B =
**64 VGPRs** become live across the whole tile sweep at the 256-VGPR gate that already spills 2,016 B
per lane. The bet is that removing 32 LDS reads per tile per lane plus their swizzle math (the hot
spilled pair) beats paying the allocator up to 256 B per lane of extra private segment. That is exactly
the kind of bet the compiler answers and the card does not always agree with, so the gate is, in order:

1. compile-only census on the served instantiation - **`sc_ld` in loop `1_596` must fall** (target 685
   → 0), the private segment must not blow up, and `ds_bpermute` (1341) and `v_wmma` (64) must be
   unchanged, i.e. the mnemonic census first, the loop number second;
2. only then a GPU ABBA (arms built from one tree, one lock hold, text equality first, a delta inside
   the F-to-F spread is not a result).

Tooling, all CPU-only, none of it touching a live tree: patch `/tmp/patch-qfrag.py` (three asserted
anchors), arm tree `/home/vermont/git/ninfer-qfrag` (define armed in its `prompt_i8_fa2.cu`), snapshot
census `/tmp/qfrag-census.sh` (copies both arms' sources under `/tmp/qfrag-census/{base,cand}` and
compiles with `-save-temps=obj`; it asserts the cand arm has the define and the base arm has *neither*
the define nor the hoist), verdict driver `/tmp/qfrag-verdict.sh` (`kernmeta.py` pair + `spillscan2.py`
on the `Li4ELb0ELb0ELb0ELi1E` tail).

**Discipline note, recorded because the tool invites a mistake:** the older `/tmp/attn-asm-cost.sh`
produces the same kind of pair by doing `git show HEAD:<header> > <header>` in the live tree and
restoring on exit. That is safe only when no build is reading the tree. It must not run while
`/tmp/build-fanout2.sh`'s packed phase rebuilds from `/home/vermont/git/ninfer-3090` (it would compile
HEAD's no-diet headers into the ABBA's packed arm), nor against `/home/vermont/git/ninfer-fanout`
mid-build. The snapshot census exists so this hazard is structural, not a habit. (Its hipcub `-I`
points at `$REPO/opt/hipcub-local/usr/include`, which does not exist on this box; hipcub lives at
`/home/vermont/opt/hipcub-local/usr`.)

Nothing measured for this lever yet - no loop number, no t/s, no VGPR claim. `iu4` stays user-gated.

### Packed vs fan-out: the GPU ABBA ran, the result is NEUTRAL, and the -16.1% static win did not show (measured 13:27-13:30)

One lock hold for the whole set (`/tmp/ninfer-gpu.lock` acquired 13:27:30, released 13:30:16, rc=0),
arms built from one tree, order F,P,P,F, `prompt 4,210`, ports 18961-18964, `ROCR_VISIBLE_DEVICES=1`,
`NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off`. F = shipped 4-row fragment fan-out
(`-DNINFER_FA2_QK_PACK_OFF`), P = packed compare-and-select (the live tree). Arm hashes differ
(`f50a34ad…` / `563f0d0d…`), which the queue verifies before it will start.

| leg | arm | prefill tok/s |
|---|---|---|
| f1 | fan-out | 178.2 |
| p1 | packed  | 176.3 |
| p2 | packed  | 175.4 |
| f2 | fan-out | 175.4 |

- **Text equality: all four legs emitted the identical string** (`sha1 a24ae9faf6…`) - the check passed.
- F mean **176.80**, P mean **175.85**, **paired delta −0.54%**; noise yardstick = F-to-F spread
  **2.80 = 1.58%**. The delta sits inside the spread, so by the standing rule **this is not a result.**
- Standing rule "revert the packed fan-out if it loses": **it did not lose** - it measured neutral.
  Nothing is reverted and nothing is committed; the packed form stays in the working tree behind its
  existing kill switch so a later set can re-measure it.

**Two instrument facts worth carrying, both discovered in this set:**

1. *The windowed prefill metric is quantized and cannot discriminate.* Every leg reported exactly
   `prefill 204.8 tok/s (1,024 tok)` = 1,024 / 5.0 s, because the whole probe prompt lands inside one
   5 s window. The request line (`req#1 done … prefill`) is the only usable reading at this prompt
   size; the `throughput` line is decorative and parsing it as signal would be an error.
2. *`output limit` = 1 token makes "text equality" a much weaker check than it sounds.* Four identical
   one-token outputs (`"We"`) prove the kernels launch, complete, and agree on the first sampled token.
   They prove nothing about logit equality downstream. A numerics-quality claim for the diet path needs
   a leg with a real generation length; that has not been run.

**Why this is the expected outcome given the statics:** the packed arm removed 611 hot-loop instructions
(3,789 → 3,178, −16.1%) and 116 folded reloads (274 → 158, −42%) while leaving `ds_bpermute` (1,341),
`v_wmma` (64), scratch loads (686 → 685) and VGPR (256 → 256) unchanged. A −16% cut in *instruction
count* that moves t/s by −0.5% ± 1.6% says the diet tile loop is **not instruction-issue-bound at this
operating point** - the cuts landed on work that was hidden behind LDS latency/barriers, not on the
critical path. That is precisely what lever (i) (scratch *traffic*, not instruction count) targets, and
it is why the Q-fragment census is the next measurement, not another instruction trim.

Resolution bound: with a 1.58% spread at n=2+2, a real effect smaller than ~1.6% is unmeasurable here.
To resolve ~0.5% needs a wider set (4+4 legs at the same prompt is ~6 min of card time per set) and a
generation long enough to be a text check.

### Lever (i) is now quantified, not hypothesized: the diet hot loop reloads 685 values and writes none (measured static, 13:36)

From the census' fresh base dump (`/tmp/qfrag-census/base/obj/prompt_i8_fa2-hip-amdgcn-amd-amdhsa-gfx1102.s`,
compiled 13:36 from the live tree's sources, served instantiation `CausalAttentionGeometry<24,4,1,41>` +
`PagedKVDirectMetadata` + `Li4ELb0ELb0ELb0ELi1E`; it reproduces the packed arm's earlier numbers exactly -
3,178 instrs, 685 in-loop scratch ops, hottest slot 756 = `0x2f4` ×123 - so its provenance is attributed):

**`loop 1_596`: 685 scratch ops = 685 loads, 0 stores.** Every scratch op inside the served hot loop is a
*reload of a slot that was written outside the loop*. That is the structural signature of
rematerializable spill traffic - the values are computed once, then re-fetched from VRAM every tile
iteration - and it is the whole premise of lever (i) confirmed on the compiler's own output.

Named by producer (`/tmp/spilltrace.py`, lane register identified per dump from `v_mbcnt`, not hardcoded):

| slots | reads/lane/iter | producer | what it is |
|---|---|---|---|
| **756** | **123** | `v_dual_and_b32 v5, 15, v126` → **`lane & 15`** | the Q swizzle **row index** - the lever-(i) candidate's own address term |
| 772-800 (9) | 28-29 each | `v_add_nc_u32` pairs | **address sums** (row × DB16 + swizzle offsets), computed once, re-read 28× per iteration |
| 852-880 (8) | 7 each | `ds_load_b32 … offset:50176+` | LDS loads, re-read every iteration |
| 892, 896 | 7 each | `v_cndmask_b32_e32 v2, -1, v*, vcc_lo` | causal-mask compare results |

The two top rows - the `lane & 15` slot and the nine address sums - account for **123 + ~252 = ~375 of
the 685 reloads (55%)**, and all of them are address/index terms, not data: free to recompute in ALU,
and exactly the terms the diet Q-fragment hoist deletes (the Q fragment load *is* that address math).
Slot 756's 123 reads cost 984 B/lane/iteration (123 × 8 B), 121,032 B/wave/iteration.

So the target for the cand arm is no longer "some spills should fall": **the Q-fragment hoist should
remove the loads in slots 756 and 772-800**, i.e. 685 → ~310 in-loop scratch loads, with the mnemonic
census (`ds_bpermute` 1,341, `v_wmma` 64) unchanged and the private segment not enlarged. If the cand
dump does *not* move those slots, the lever is dead on the compiler's own evidence and no GPU time is
owed to it. (Cand arm still compiling as of 13:40; no cand numbers claimed.)

**Tooling note, learned here:** `/tmp/spilltrace.py` matches sections with `mangled.startswith(want)`, so
the argument must be the leading `_ZN6ninfer3ops37causal_attention_prompt_i8_fa2_kernel…` prefix - a
mid-string shape substring silently yields "section not found".

**Census v1 was an instrument fault, and the cand numbers above are therefore NOT claimed.** The pair
came back byte-for-byte identical (109,523,160 B both, hot loop 685/3178, same hottest slots). A
"no change" result at that fidelity is not a lever outcome - it is the signature of an arm that never
compiled the candidate. Cause: the census staged snapshots at `$OUT/<arm>/src/src/ops/...` (it used the
source-relative path under a `src/` root), while `prompt_i8_fa2.cu` includes
`"ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cuh"` - relative to `src/`. That path did not
exist in the snapshot, the include **fell through to `-I$REPO/src`**, and *both* arms compiled the live
tree's packed header. The `-DNINFER_FA2_QFRAG_REG` define sat in cand's `.cu` with nothing to gate.

Three fixes, all of them assertions rather than habits (`/tmp/qfrag-census.sh` v2):
1. stage at the **include-relative layout** `$OUT/<arm>/src/ops/softmax_attention/dense/causal_cache/`,
   and refuse to run (`exit 5`) if that file is absent;
2. compile with **`-H`** (clang's include graph trace) and require, from the compiler's own output, that
   the header resolved under `$OUT/<arm>/src/…` - anything else is `exit 9`;
3. require the two arms' device asm to **differ by sha1** (`exit 10` if equal). A candidate arm that
   cannot change codegen is a measurement failure, and it must fail loudly at compile time rather than
   arrive at the verdict stage looking like a null result.

The faulty dumps are quarantined, not deleted, so nothing downstream reads them as a result. **The base
arm's own numbers are unaffected**: base's intended header *is* the live tree's packed header, which is
exactly what it compiled, so the 685-load / slot-756 census above stands as a measurement of the packed
form.

Two instrument facts from the re-run (measured 14:05-14:08), both of which would have *faked* a verdict:

- **Quarantine by a name the globs cannot match.** The first quarantine renamed the dumps
  `prompt_i8_fa2-hip-…-gfx1102-FAULTY-INSTRUMENT.s`, which still matched the census's own
  `prompt_i8_fa2-hip-*.s` glob. `BASE_S=$(ls …)` then held two paths, `sha1sum < "$BASE_S"` read neither
  (the value contains a newline), and two empty digests compare **equal** - so the sha1-inequality
  assertion would have fired `exit 10` and reported "arms byte-identical, candidate did not reach
  codegen" about a pair that had not been compared at all. An assertion whose failure mode is produced by
  the assertion's own input handling is not an assertion. They are now
  `prompt_i8_fa2-FAULTY-INSTRUMENT-v1.s-QUARANTINED`: every glob in the census and the verdict script
  resolves to exactly one file.
- **clang's `-H` prints `<dots> <path>`, not `> path`.** The v2 assertion grepped GCC's
  `^> <path>$` form, matched nothing, and aborted with "base compiled the header from , NOT its own
  snapshot" - a fault in the *test*, on an arm that the compiler's own trace shows compiled its snapshot
  (`/tmp/qfrag-census/base/obj/compile.log` line 190:
  `. /tmp/qfrag-census/base/src/ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cuh`). The
  pattern is now `^[. ]+ <path>$`. Same shape as the layout bug: an instrument that reports a fault
  should be checked against the compiler's output before the fault is believed.

## Census v4 is a confounded pair, and its candidate fails lever (i)'s own stated conditions

Measured 14:39-14:51 on 2026-10-23. No GPU time, no lock; both numbers below are from device asm.

**The confound.** Census v4 staged base from `/home/vermont/git/ninfer-3090` and cand from
`/home/vermont/git/ninfer-qfrag`. Between the two snapshots the user made the direct fan-out the
production form in the 3090 tree (`prompt_i8_fa2.cuh:617`, `#if !defined(NINFER_FA2_QK_PACK_ON)`),
and the qfrag tree predates that inversion (`…:647`, `#if defined(NINFER_FA2_QK_PACK_OFF)`). Reading
the gate region closes the question of *where* they differ: the `QK_PACK` gate wraps the **accumulator
readout** (the `__shfl` fan-out of `dots` into `rows0..rows3` and its dynamic `j = 2 * lid` index),
which sits *below* the Q fragment load. So census v4's pair moved the readout form AND the Q-fragment
hoist in one step, and its `arms differ: 0f0f24d35d base / 1d3cd78388 cand` line is not attributable
to lever (i). It stays on disk as a record of a confounded pair, not as a result.

**Re-based.** `/tmp/sync-qfrag.sh` copies `prompt_i8_fa2.cu/.cuh/prompt_i8_fa2_plan.h` from the 3090
tree into the qfrag tree, re-applies `/tmp/patch-qfrag.py`, re-arms the define, and asserts: the
fan-out gate present (`exit 20`), the packed-default gate gone (`exit 21`), 3 header gate markers
(`exit 24`), the define in the `.cu` (`exit 25`), and - new, from this segment's compile failure -
that `q_frag[Groups][4]` is declared at **function** scope (`exit 26`). The first version declared it
inside the `if constexpr` fill block; its consumer is the `qk_softmax` lambda ~70 lines later, so the
candidate died with `error: use of undeclared identifier 'q_frag'` at the wmma call site
(`cand/obj/compile.log` line 791) and, because the qfrag tree's `.cu` carries the define armed, the
same bug broke `cmake --build` on that tree. After the sync the header-vs-3090 diff is **purely
additive**: 32 added lines, 0 removed - the fan-out form is untouched.

**The candidate's static numbers (census v4 pair, served `G24/4 B0 w4 S0 PV0 PK0 st1`, diet tail
`Li4ELb0ELb0ELb0ELi1E`):**

| | vgpr | priv | instrs | sc_LD | sc_ST | bpermute | iu8 |
|---|---|---|---|---|---|---|---|
| base (fan-out, no lever) | 256 | 2032 | 21,021 | 1185 | 486 | 1341 | 64 |
| cand (packed readout + hoist) | 256 | **2208** | 21,050 | 1181 | 481 | 1341 | 64 |

Hot loop `1_596`: base 685 in-loop scratch loads / 3,178 instrs / **158** folded reloads, hottest
offset `(756, 123)`; cand 679 / 3,159 / **176** reloads, hottest offset `(752, 123)`.

Three things follow, and all three are unfavorable:
1. **The private segment grows by 176 bytes (2032 → 2208).** Lever (i)'s own gate says the diet path
   must achieve its scratch-load cut "with the private segment not enlarged". It is enlarged.
2. **The hoist does not touch the loop's hottest slot.** Base reloads the slot at `0x2f8` 123 times
   per iteration; cand reloads the slot at `0x2f0` 123 times per iteration. Same count, same
   structure, shifted offset. The premise of lever (i) was that the `lane & 15` swizzle address *is*
   that hottest pair; removing 32 Q fragment loads from the loop leaves the pair intact, because the
   reload is fed by the loop's other address traffic, not only by the Q loads.
3. **In-loop scratch loads moved 685 → 679 (−0.9%), with reloads up (+18).** The lever's target was
   ~685 → ~310. A 0.9 % movement paid for by a 176-byte private-segment increase, with the static
   instruction-count-to-speed translation already established as unreliable (packed lost its server
   ABBA at −0.54 % while cutting 611 instructions), is not the reduction that owes card time.

**Census v5 is the single-lever test** (`/tmp/qfrag-census5.sh`): both arms compile **the same source
bytes** - the re-based qfrag header with the inert-without-the-define hoist, plus the 3090 tree's
`.cu` (which does *not* define the lever, so the pair cannot be null) - and differ only by
`-DNINFER_FA2_QFRAG_REG` on the candidate's command line. Assertions: staged layout (`exit 5`),
staged `.cu` must not pre-define the lever (`exit 7`), 3 markers in the staged header (`exit 6`),
staged header must be the fan-out-default form (`exit 27`), compile rc (`exit 11`), `-H` resolution
(`exit 9`), missing dump (`exit 28`), sha1 inequality (`exit 10`). Baseline for the comparison is
now measured on the fan-out form itself - **685 in-loop loads / 158 reloads**, i.e. the packed form's
figure is not the reference the served shape should be judged against.

## The BF16 prompt-attention candidate now has a multi-token ABBA: +25.9 % prefill, decode unchanged, output coherent

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

## The Q-fragment register hoist fails its own static gate: the served hot loop gets worse, no GPU time is owed

The census v5 pair (`/tmp/qfrag-census5.sh`, log `/tmp/qfrag-census5.log`) is the first clean instrument
for lever (i): both arms compile **the same staged source bytes** (staged `sha=3c26c778f3`), the *only*
difference is `-DNINFER_FA2_QFRAG_REG` on the candidate command line, and the instrument's own assertions
all passed — both arms `rc=0`; `-H` shows each arm resolving
`/tmp/qfrag-census5/src/ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cuh`, i.e. its own staged
header, not the live tree's; and the device asm differs (`a4d5b33d97` base / `58a6f58b85` cand). Header
inclusion proven, candidate/base asm proven different, exactly one lever between them.

Verdict on the served diet shape `G24/4 w4 S0 PV0 PK0 st1` (`/tmp/qfrag-verdict5.log`):

| | base | cand (+QFRAG_REG) | Δ |
|---|---|---|---|
| private segment | 2016 B | **2240 B** | **+224 B** |
| total instrs | 21762 | 21800 | +38 |
| scratch loads (whole kernel) | 1188 | 1196 | +8 |
| **hot loop `1_596` in-loop scratch loads** | **686** | **693** | **+7** |
| hot loop reloads | 274 | 280 | +6 |
| hottest hot-loop spill pair | (760, 123) | (756, 123) | ~same |

The gate was: card time is owed only if the candidate materially reduces the served hot loop's ~685
in-loop scratch loads (toward ~310). It **increases** them, by 7, with reloads up 6 and the private
(spill) segment up 224 B. The lever's premise — that the 32 per-tile Q-fragment LDS reads and their
swizzle math are removable live state — is right about the reads being loop-invariant, but the kernel
is **already compiled at the 256-VGPR ceiling** in this shape. A "register hoist" under a saturated VGPR
budget is a register *demand* increase with nowhere to go: the 64 VGPRs of fragments cannot displace the
in-loop loads, they add live state the allocator must spill, and the spills land in the hot loop. That
is the mechanism the numbers show, and it explains why the static instruction count moved only −28
whilst the scratch traffic that the gate measures moved the wrong way.

The packed shapes are the control: every `PV1` row in the metadata pair is **byte-identical** between
arms (priv 2320/2320, 21179 instrs/21179, etc.), because the hoist is gated to the diet form
(`!PackedValues && !PackedKeys && Warps == 4 && Stages == 1`). Only diet rows move, and the diet rows
that take the packed-route form are untouched — so the movement is the lever's own doing, not noise,
and the gate behaved as specified.

**Outcome: lever (i), in this pure-addition form, is closed without card time.** Per the rule that
static instruction count alone does not translate to speed, the census was run on the gate metric
(in-loop scratch loads), and it is unfavourable, so it is recorded as unfavourable and the GPU is not
used for it. A future attempt at the same idea has to *fund* the fragments — pair the hoist with a
matching release of live state in the same loop (the `(lane & 15, 0)` pair at slot 0x2f8 is the known
address), not add 64 VGPRs on top of a 256-VGPR kernel.

### 2026-10-23 serve-now: fresh confirmation on the committed b78f2c94 binary

`build-hip/apps/ninfer-serve` (built 14:54 from the committed tree, production defaults
`NINFER_FA2_QK_PACK_ON` off) measured on the served 4,210-token BF16-KV lighthouse prompt, one
flock hold, port 18941, `ROCR_VISIBLE_DEVICES=1`, `NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off`:

| rep | cache | TTFT | rate |
|---|---|---|---|
| 1 (`max_tokens=1`, cold) | 0.0% | 23.6 s | **prefill 178.4 tok/s** |
| 2 (`max_tokens=32`) | 100.0% checkpoint | 4.14 ms | **decode 13.0 tok/s** |
| 3 (`max_tokens=1`) | 99.8% checkpoint | 1.1 s | 6.3 tok/s (tail re-prefill, not a prefill leg) |

- Prefill 178.4 t/s sits above the qualification ABBA candidate leg (171.2) and well above the
  baseline leg (136.0); decode 13.0 t/s is above the recorded 12.0–12.1. Single-pair fresh numbers,
  not an ABBA — treat as confirmation of the committed state, not as a new delta.
- Coherence: rep1 and rep3 completions are byte-identical text at temperature 0; rep2's 32-token
  completion is on-topic English ("We need to respond to user: "Summarize the paragraph above in one
  sentence." …"). Target 300+ t/s prefill remains open.
- Instrument note (honest record): the serve script's coherence print passed the response JSON as
  Python's *script* path, so the heredoc never ran and nothing printed; the texts above were
  recovered from the saved `serve-now_rep*.json` after the lock was released. The timings and shas
  are unaffected.

### 2026-10-23 census6 base anchor (lever (j), rematerialization)

The census6 base arm (staged source sha `e3f227c686`, inertness of the `NINFER_FA2_QFRAG_RMAT` gate
proven before compiling: define-off == committed header code lines) reproduces the served-shape
baseline exactly — `G24/4:4w S0 PV0 PK0 st1`, hot loop `1_596`: **686 in-loop scratch loads**
(563 b32 + 123 b64), 274 folded reloads, 3,789 in-loop instructions, 1,670 total scratch refs,
private segment stores spanning 0..2000. The instrument measures, so the candidate arm's number,
when it lands, is comparable to 686 on the gate.

**Census6 result (15:23): lever (j) rematerialization is UNFAVOURABLE — closed without card time.**
Both arms rc=0, headers resolved to the staged copy, asm shas differ (`41e721fa61` base /
`640184b394` cand) — every instrument gate passed, and the gate metric moved the wrong way:
served hot loop `1_596` in-loop scratch loads **686 → 873 (+187, +27%)**. The cand dump shows the
mechanism: the diet pair at base offset 760 (123 b64 loads, the known `(lane & 15, 0)` pair)
*split into two b32 loads per iteration* (offsets 748 and 764, 122 hits each); 244 new
`scratch_load_b128` appeared; total scratch refs 1670 → 1908; the served kernel's private segment
grew 2016 → 2192 B. The empty asm barrier forced the allocator to treat lane-derived chains as
redefined, and the recomputed chains needed live values that the allocator then spilled: at the hard
256-VGPR ceiling the equilibrium is zero-sum, every liveness perturbation pays in scratch.
Forms closed so far: **(i)** pure-addition Q-fragment hoist (+7 loads, +224 B private), **(j)**
lane barrier at lambda scope with the fan-out readout levers (+187 loads, +176 B private). Next
form **(k)**: the same barrier applied surgically to the diet QK row terms *only* (`key_row`,
`query_row` — the chain that feeds the 0x2f8 pair), with no fan-out levers, minimal blast radius.

**Census7 first attempt (15:36): infra fault, NOT a lever verdict.** The instrument gates all
passed before compiling — lever (k) patch applied to a staged copy (sha `937bf181a9`), inertness
proven with the lever-(k) gate (`INERT: define off == committed header`), base arm rc=0 with its
dump complete (109,969,713 B). The candidate arm then died in the backend: `rc=1`, `fatal error:
error in backend: IO failure on output stream: Disk quota exceeded` — `/tmp` (16 G tmpfs) had
reached 80% and the `-save-temps=obj` pair (~400 MB) did not fit. No candidate dump, no verdict.
Fix: census7 re-run with its working directory on the root filesystem (286 G free); census6's spent
110 MB asm dumps (verdict recorded above, scans preserved in `/tmp/q6-*-scan.txt`) deleted to
reclaim room. The base-arm baseline of 686 is unaffected by the re-run — same staged source,
same compiler, same flags, so the cand number stays comparable to 686 on the gate.

**Census7 rerun result (15:49): lever (k) is UNFAVOURABLE — closed without card time.** Rerun
clean: staged sha `14c8c1b8a3` (the 15:26 attempt's sha differed only because its staging dir
pre-existed with leftover files — the staged bytes are identical across attempts, verified by
`diff`), inertness proven, both arms rc=0, headers resolved to the staged copy, asm shas differ
(`76add54bde` base / `3c4696448a` cand), base arm reproduces the anchor exactly (`1_596`: 686
loads, 563 b32 + 123 b64, 274 reloads, 3,789 instrs). Candidate `1_596`: **709 in-loop loads
(586 b32 + 123 b64), +23 (+3.4%)**, 281 reloads, 3,774 in-loop instrs (−15), total scratch refs
1,667 (−3). Mechanism: the lever targeted the 123× pair reload at offset 760, and the pair did
*not move* — b64 count stays exactly 123, the pair just re-anchored to offset 748 as the private
layout shifted (stores 386→389 distinct offsets, max 2000→2012); what the barrier produced is
+23 b32 reloads of the address terms' own inputs (`nt16`, `row_base`, lane terms now live across
more of the loop). The premise fails: the pair is not a register-pressure artifact — barriering
the row terms frees nothing for it, it only costs their recomputation.
**Forms (i), (j), (k) close the register-barrier family: at a hard 256-VGPR ceiling with LDS as
the sole occupancy binding term, liveness games move scratch around, never down.** The pair must
be attacked structurally — reduce what is simultaneously live in the QK→softmax→PV chain, which
is the shape of the diet `Bc`/tile geometry, not a compiler hint. Next: static census of the diet
tile geometry (key-block chunk `Bc = 64` → 32): the gate is LDS ≤ 32,768 B (unlocks 3 WG/CU vs
today's 2, LDS 51,200 B being the binding term) *and* in-loop loads not materially above 686.

**Static probe of the diet tile geometry (16:05): served launch is 51,200 B = 1 WG/CU, and only
16-row staging doubles occupancy.** Compile-time probe `/home/vermont/tmp/bc32-static/
probe_diet_smem.cu` reproduces the header's law from the constants (D=256, Br=64, Groups=4):
`SmemBytes(Bc) = 17,408 + 528·Bc` (Q tile 16,384 + Q scales 1,024 + stage), asserting every value
against the header at compile time. `Bc=64` → **51,200** — the served diet launch, **1 WG/CU**
(the sentence above saying "today's 2" was wrong: 65,536 / 51,200 = 1; LDS is the sole binding
term). `Bc=32` → 34,304 → **still 1 WG/CU**: 2×34,304 = 68,608 > 65,536 — the fixed Q tile blocks
the halving, and 3 WG/CU (≤ 21,845 B) is unreachable while it stays. `Bc=16` → 25,856 →
**2 WG/CU: resident waves 4 → 8 per CU.** The 2-WG line in tile rows: 528·bc + 17,408 ≤ 32,768 →
bc ≤ 29.1; WMMA's 16×16 tiles require bc % 16 == 0, so **16 rows is the only sub-page tile that
doubles occupancy**. The register file fits: 2 WG × 128 lanes × 256 VGPR = 65,536 = exactly the
gfx1102 VGPR file, and the served kernel is already ≤ 256 VGPR/lane with its 2,016 B/lane private.
Math is untouched at 16-row granularity: same WMMA total (4 groups × 1 tile × 4 k-steps per chunk ×
4 chunks), same scores and online softmax, finer causal/alpha granularity (16 keys, page-law
respects the per-tile bound), zero-fill past `max_query_abs` preserved per chunk.

**Lever (l) instrument (16:20): `/tmp/patch-bc16.py`, define `NINFER_FA2_DIET_BC16`.** Diet-only
quarter-page staging: stage struct parameterized on tile rows (16-row diet chunk = 8,448 B vs the
33,792 B page), diet-only `QKNt = 2` / `PVKs = 1` (body sites shrink for free), an `issue_chunk
(kb, c16)` staging the page's rows `[16c, 16c+16)` with the same swizzle, 16-byte cp law, and
zero-fill, `qk_softmax(kb, c16)` with chunk-local k0/full_tile/V-scale scan, and a chunked 1-stage
sweep gated on `Shape::Diet16` alone — non-diet shapes and the S2 overlap path keep the committed
bytes verbatim in the `#else` arms (11 paired gates + 2 pure additions = 13 gate lines; no base-arm
line is added anywhere). First strip run **caught an instrument fault** (S7's candidate text
re-emitted its own anchor, duplicating `cp_commit(); };` into the base arm — the inertness proof
working exactly as designed); S7 became a single paired gate and the re-run proves `INERT: define
off == committed header (code lines)`. Census8 (`/tmp/rmat-census8.sh`) reuses census7's proven
instrument (identical staged bytes for both arms, define on the candidate command line only, 13
marker count, header-resolution and fan-out-default checks, base-vs-census7 asm cross-check on the
686 anchor). The 25,856 B byte law is pinned by static_assert in the header's candidate arm
(diet `SmemBytes == 25856 ≤ 32,768`; non-diet `Stage::Bytes == 33792`): if the law breaks the
candidate simply fails to compile. **Gate owed before card time: served loop in-loop scratch loads
materially below 686, private/VGPR not worse.** Compile only, no lock.

**Census8 run 2 (16:08): base arm clean, cand arm failed on two cand-only faults — fixed, no GPU time spent.**
The base (no-define) arm compiled rc=0 from the staged header (16:08:14) — the instrument is sound and the
committed anchor is reproducible. The candidate arm (rc=1, 16:08:16) failed exactly twice, both inside gated
arms: (1) S4's `SmemBytes <= kCausalPromptFa2LdsCeilingBytes / 2` sat at the line-138 insertion point, before
the constant's definition at line 149 — fixed by moving the S4 anchor to `kCausalPromptFa2LdsCeilingBytes =
64 * 1024;` itself (dry-verified: definition 187, lever assert 196, after it); (2) the `qk_softmax` signature
gate added a second parameter, breaking the committed Stages==2 overlap sweep's one-arg `qk_softmax(kb)` calls
(header lines 925/940) — fixed with a default argument, `[&](int kb, int c16 = 0)`: for every non-diet shape
the sweep is the committed one (Diet16 requires Stages == 1, so c16 = 0 there), and `k0 = kb*Bc + 0*16 =
kb*Bc` reproduces the committed k0 verbatim. Dry-run after both fixes: patch applies (11 gate pairs), strip
proves `INERT`, 13 markers, signature and S4 placement verified in the staged text. Census8 relaunched
compile-only, no lock.

**Census8 verdict (16:20): FAVORABLE — the diet chunk path has NO in-loop scratch loads. Gate passed; GPU ABBA is owed.**
Run 3 (staged sha `46811a7e0f`): base rc=0 16:16:07, cand rc=0 16:20:55, asm differs (`c9d5201c08` / `90738400d9`) —
the define reaches codegen. **Anchor check:** census8 base asm sha differs from census7's `76add54bde` because census7
staged the pre-b78f2c94 header (the WMMA/split-LDS commit added 122 header lines after census7 staged its copy); the
served loop's census on census8's base reproduces the anchor exactly — loop `1_596`, **686 in-loop scratch loads, 274
reloads, 3,789 in-loop instrs, hottest (760,×123)** — so 686 stands on the committed text.
**Candidate, served shape (G24/4 4w S0 PV0 PK0 st1): in-loop scratch loads 0** — "no scratch in any annotated loop";
kernel-wide scratch 1,670 → 112 (32 offsets 0..496, all OUTSIDE loops, 48 b128 + 64 b64), total instrs 21,765 → 11,459.
4w S1 diet shape identical (0 in-loop, 112 total). **Non-diet shapes byte-identical to base** (8w loops 207, PV1 loops
181/97 — untouched, as designed). Served-kernel metadata: **VGPR 256 → 240** (2 WG/CU fits 61,440 of 65,536, headroom
4,096), **private 2,016 → 528 B/lane**, numbered SGPR 40 → 57 (≤ 102/lane cap; not gate material). The 25,856 B byte
law is compile-proven: the cand arm's `static_assert(SmemBytes == 25856 ≤ 32768)` passed → 2 WG/CU at launch, resident
waves 4 → 8 per CU, VGPR/SGPR/private all fit the 2-WG allocation.
Caveats owed at the card: the diet path rescales the alpha chain 4× per page (16-key granularity) so cand is NOT
bitwise-identical to baseline — coherence, not bit-identity, is the bar; barriers per page 3 → 12 (per-chunk publish);
four chunk cp_wait/syncthreads round-trips per page add latency the census cannot price. **Gate wording
("in-loop scratch loads materially below 686, private/VGPR not worse") is satisfied with margin. Next: build with
`-DNINFER_FA2_DIET_BC16`, multi-token ABBA under flock — prefill t/s AND coherent text, all four legs, unfavorable
included.**

### 2026-10-23 lever (l) A/B build: object-swap protocol (no GPU; tree restored after)

The candidate binary was built surgically, not by a full rebuild, so the ONLY delta vs the
anchor binary is the served attention TU:
- CAND object = census8's proven candidate `prompt_i8_fa2.o` (sha `0d3d62b6a6`; staged-bytes arm,
  define on the command line, asm sha `90738400d9`). Symbol table diffed against the committed
  object (`e9bd4fb67e`): **identical symbol sets** - the lever adds only member constants
  (`Diet16`), the mangled kernel names and host stubs are unchanged, so the swap is link-safe.
- Swap: copy object to `CMakeFiles/ninfer_ops.dir/.../prompt_i8_fa2.cu.o`, `ar r` into
  `libninfer_ops.a`, `cmake --build build-hip --target ninfer-serve` (link-only). Result
  409,124,072 B vs baseline 409,939,176 B (−815 KB, consistent with the leaner kernel: instrs
  21,765 -> 11,459, private 2,016 -> 528 B/lane). Saved as `/home/vermont/tmp/bc16-ab/ninfer-serve.cand`
  (`8d467b7abe`). Baseline arm = the 14:54 committed-tree binary copy (`4f6d892f91`), which is the
  serve-now anchor (prefill 178.4 / decode 13.0).
- **Determinism note (measured):** relinking the restored pristine archive yields `0ab1eec3ada4`,
  byte-DIFFERENT from the 14:54 binary `4f6d892f91` - same size, identical symbol sets, and all
  139,393 differing bytes fall in the gaps BETWEEN allocated ELF sections (padding; verified by
  mapping every diff offset against the section table). Functional arms; the A/B therefore uses the
  preserved 14:54 copy as the base leg, not the relink.
- Tree state after the build: baseline object back in place (`e9bd4fb67e`, sha-verified), pristine
  archive restored byte-for-byte, `apps/ninfer-serve` = relinked baseline. No source-tree changes.
- Next under lock: multi-token ABBA (`/home/vermont/tmp/bc16-ab/abba-bc16.sh`, w=1024 prompt ->
  4,210 tokens, max_tokens=128, temp 0, 4 legs B,C,C,B in ONE flock hold).

### 2026-10-23 lever (l) multi-token ABBA: **NULL on speed — closed; coherence and text-exactness pass**

`/home/vermont/tmp/bc16-ab/abba-bc16.sh bc16mt 128`, ONE flock hold 16:26:45-16:30:00, legs B,C,C,B,
served 4,215-token lighthouse prompt, `max_tokens=128`, temperature 0, env
`NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off ROCR_VISIBLE_DEVICES=1`.
Arms: BASE = committed b78f2c94 binary `4f6d892f91` (the serve-now anchor);
CAND = object-swap build `8d467b7abe` (census8 candidate object `0d3d62b6a6`; served diet kernel
VGPR 240, zero in-loop scratch loads, 2 WG/CU byte law - all verified in the census).

| leg | TTFT | prefill tok/s | decode tok/s | completion sha1 |
|---|---|---|---|---|
| base1 | 23.6 s | **178.8** | 13.1 | 77f1779eb831 |
| cand1 | 23.9 s | 176.7 | 13.0 | 77f1779eb831 |
| cand2 | 24.0 s | 175.7 | 12.9 | 77f1779eb831 |
| base2 | 24.0 s | **175.5** | 12.9 | 77f1779eb831 |

- **Prefill: base mean 177.15, cand mean 176.2, delta -0.95 t/s (-0.5%)** - inside the base's own
  3.3 t/s leg spread, with both cand legs strictly between the base legs and TTFT drifting
  monotonically 23.6 -> 24.0 s across the set (thermal ordering: the first server on a just-released
  card is fastest). A delta inside off-leg spread is not a result. Decode 12.9-13.1: unchanged
  (decode runs the T2 simt kernel - the lever does not touch it).
- **Coherence: PASS on all four legs** - 128 tokens each, 93 words, 63% distinct, max 8-gram repeat
  1, on-topic English. **The completion is token-IDENTICAL across all four legs** (sha 77f1779eb831,
  574 chars): the 4x-per-page alpha rescale did not move a single sampled token on this prompt at
  temperature 0. The census's warning about non-bit-identity was the honest caveat; the measured
  outcome is bit-exact at the token level.
- **Mechanism of the null (for the record):** the census gate metric moved enormously (in-loop
  scratch loads 686 -> 0, instrs 21,765 -> 11,459, private 2,016 -> 528 B/lane, VGPR 256 -> 240,
  2 WG/CU at launch) and served speed did not move at all. The served diet prefill is NOT
  spill-bound: at the serial 1-stage discipline the per-chunk `cp_wait<0>` + `__syncthreads` latency
  chain is the floor, and lever (l) pays that chain 4x per page (12 barrier pairs vs 3) while
  removing the spills it never paid for. The 2-WG occupancy doubles waves-per-CU but each WG is an
  independent CTA with its own barrier chain, so the chain does not shorten. This is the
  "static instruction count alone does not translate to speed" rule, confirmed from the opposite
  direction: a massive spill elimination that changes nothing measurable.
- Disposition: lever (l) stays define-gated in `/tmp/patch-bc16.py`, NOT applied to the tree
  (source tree verified clean; only docs dirty). Arms preserved at `/home/vermont/tmp/bc16-ab/`
  (`ninfer-serve.base` `4f6d892f91`, `ninfer-serve.cand` `8d467b7abe`). A follow-up on the same
  geometry must attack the barrier/latency chain (overlap chunk c16+1's copy with chunk c16's
  compute - a 2-stage diet pipeline, 2 x 8,448 B stage). Statics, corrected twice: the header
  overlays the Q scales onto the stage when Stages==2 (line 116: `SmemBytes = QBytes +
  Stages*Stage::Bytes + (Stages == 1 ? QScaleBytes : 0)`), so the diet 2-stage total is 16,384 +
  2 x 8,448 = **33,280 B - 512 B OVER the 32,768 B 2-WG/CU line**, landing 1 WG/CU. A 2-WG
  variant exists on paper: move the 2 x 256 B chunk scales out of LDS into registers (16 rows x 8
  groups, sub-register-per-lane sized) -> 16,384 + 16,384 = exactly 32,768 = 2 WG/CU, pipelined
  and spill-free. Route: the launcher (`prompt_i8_fa2.cu` lines 39-83) picks Stages from
  `Fit::StageCount` at compile time - the plan's requested 2-stage shape branches to `launch<w,2>`
  when the fit type keeps 2 stages, so a 2-stage diet needs the Fit descent to keep Stages=2 for
  the diet geometry, not a new launch site. The bet for a 2-stage lever is spill-free + pipelined
  vs the baseline's serial + spilling at the same 1 WG (and occupancy per the scale-placement
  choice). The chunk staging is sound, its serialization is the cost.
- Target 300+ t/s prefill remains open; committed state stands at ~177 t/s fresh-measure mean.

### 2026-10-23 lever (l2) census v9: gate PASSES (686 in-loop scratch loads -> 0); the launcher also carried dead kernels — statics only, GPU verdict pending

`/tmp/rmat-census9.sh`, log `/tmp/rmat-census9.log`. Compile only, no lock. Lever (l2) = lever (l)'s
diet math (certified token-exact) staged through **two** LDS stages so chunk c16+1's copy overlaps
chunk c16's compute — one barrier per chunk instead of the serial `cp_wait<0>` + `__syncthreads`
chain lever (l) identified as the floor. Source = committed tree + `/tmp/patch-l2.py` (header) +
`/tmp/patch-l2cu.py` (launcher), every site define-gated on `NINFER_FA2_DIET_L2`.

- Instrument: both arms compile **the same staged bytes** (source sha `3e005884ce`); the only
  difference is `-DNINFER_FA2_DIET_L2` on the cand command line. Gate markers 21/21 across `.cu` +
  `.cuh`; `/tmp/rmat-strip.py` proves **both** files reproduce the committed bytes with the define off
  (so the base arm is production code); `-H` confirms each arm compiled the staged header.
- **Byte law / Fit, measured by compile-only probe** (`/tmp/l2probe/probe.cu`, `-fsyntax-only` against
  the staged header; values read out of `implicit instantiation of undefined template 'Show<N>'`
  diagnostics): `SmemBytes` 4w st2 = **33,280**, 4w st1 = 25,856; 8w st2 = 100,352, 8w st1 = 68,608;
  8w PV1 st2 = 84,992, PV1 st1 = 60,928 (non-diet values identical in both arms); 8w
  `Stage::Bytes` = 33,792 with `Rows` = 64. `CausalPromptFa2LdsFit`: 4w **StageCount 1 -> 2** (the
  diet shape fits under the 65,536 ceiling so Fit keeps 2 stages — the served route becomes a new
  `launch<4,2,split>` instantiation), 8w StageCount stays 1, 8w Fit descends to a 4-warp st1 shape.
  `Diet16`: 4w = 1, 8w = 0. Lever (l)'s note that 33,280 lands 1 WG/CU (512 B over the 32,768
  2-WG/CU line) is confirmed: occupancy is unchanged, l2 buys overlap, not occupancy.
- **Served-shape census (gate owed before GPU time):**

  | served G24/4 4w S0 PV0 PK0 | base (st1 diet) | cand (st2 diet) |
  |---|---|---|
  | in-loop scratch loads | **686** | **0** |
  | kernel instrs | 21,765 | 12,162 |
  | scratch instrs | 1,670 | 112 |
  | VGPR / SGPR | 256 / 40 | 238 / 54 |
  | private B/lane | 2,016 | 528 |

  Base arm reproduces the census8 committed anchor exactly (21,765 instrs, hot loop `1_596` 3,789
  instrs, 808 cndmask, 686 in-loop loads). Its asm is *not* byte-identical to census8's saved
  anchor: the diff is only the `__hip_cuid_*` TU-hash symbol and its `.type/.size/.globl/.addrsig_sym`
  lines — the gate wrapper changes source bytes, hence the TU hash, not codegen. Classified benign.
- **Incidental finding, independent of the lever — the launcher instantiated kernels no route can
  launch.** `launch_fitted` chose the warp leg with a runtime `if (fit_warps == 4)` where
  `fit_warps = Fit::WarpCount`, a compile-time constant, so **both** `run<4>` and `run<8>` instantiate
  for every Fit. Host-side evidence: the committed binary has `launch_fitted` instantiations only for
  Fit shapes W4 {PV0/PK0, PV1/PK0, PV1/PK1} st1 and W8 {PV1/PK0, PV1/PK1} st1 — **no W8 PV0/PK0
  instantiation** — yet its device asm contains W8 Splt0/1 PV0/PK0 st1 kernels. They were dead code.
  Once a Fit has StageCount == 2 (what l2 does), the same branch materializes 8 live-looking 8-warp
  st2 kernels (VGPR 256 / private 2,208 B/lane) that can never launch, because in that instantiation
  `fit_warps == Fit::WarpCount == 4`. Fixed at the source in the lever: `constexpr int fit_warps` +
  `if constexpr`. Section census (`/home/vermont/tmp/secscan/secscan.py`, name-level):
  base 156 = cand 156 kernel sections; **+8 diet W4 PV0/PK0 st2, −8 dead W8 PV0/PK0 st1**; packed
  routes untouched; **zero W8 st2 sections**.
- Gate verdict: the handoff condition for spending GPU time (reduce the served loop's ~685 scratch
  loads) is met — 686 → 0, with VGPR and private bytes both improved. GPU time is owed.

### 2026-10-23 lever (l2) ABBA set `l2mt`: **valid 4-leg ABBA, NULL on speed — and confounded by a tree rebuild**

(Correction, same day: an earlier version of this section said the set had 3 legs and no DONE line.
That was a **premature log read**, not a lost leg — all four legs and `DONE 17:38:29` are in
`/tmp/l2mt_abba.log`. Lesson kept: read a background run's log only after its completion
notification, and read the wrapper output, not just the `tee` log.)

`/home/vermont/tmp/l2-ab/abba-l2.sh l2mt 128`, ONE flock hold 17:35:16–17:37:41, same lighthouse
prompt (4,215 tokens), `max_tokens=128`, temperature 0, same env as lever (l)'s set.

| leg | TTFT | prefill tok/s | decode tok/s | completion sha1 |
|---|---|---|---|---|
| base1 | 23.6 s | 179.0 | 13.1 | 77f1779eb831 |
| cand1 | 23.9 s | 176.6 | 13.0 | 77f1779eb831 |
| cand2 | 24.0 s | 175.5 | 13.0 | 77f1779eb831 |
| base2 | 24.0 s | 175.6 | 12.9 | 77f1779eb831 |

Arithmetic: base mean **177.3** (179.0, 175.6), cand mean **176.05** (176.6, 175.5), Δ **−1.25 t/s
(−0.7%)**, against a base off-leg spread of **3.4 t/s**. The lever's own legs are the two middle
positions, the base legs the outer two, and the fastest leg of the whole set is the *first* one
(179.0) — the known first-leg advantage. **Δ sits inside the spread: this is a NULL, not a loss, and
it is not a win.**

- **One reason this set cannot carry a verdict: the confound.** BASE was the 14:54 committed binary while CAND linked the 16:25 tree archive —
  `find -newer` shows the tree rebuilt (`libninfer_ops.a` 16:25:25, `apps/ninfer-serve` 16:25:26)
  *after* the base arm binary was made, and the arm-build symbol diff showed 15 removals in
  **unrelated** TUs (q5/q8/q4 mma stubs, serve chat-template, xgrammar helpers). A tree-rebuild delta
  of unmeasured size sits inside the comparison.
- What the set does show, kept for the record: coherence passes on every leg (93 words, 63% distinct,
  max 8-gram repeat 1) and the completion is **token-identical** to lever (l)'s set (sha
  `77f1779eb831`, 574 chars) — the diet math moves no sampled token, now confirmed across a
  pipelined variant. Prefill ordering (179.0 > 176.6 > 175.5 > 175.6, monotone with the leg order
  through leg 3) matches the thermal drift pattern already recorded, so nothing here separates the
  lever from noise.
- **The strongest result in this set is negative and general.** Lever (l2) delivered every static
  prize the handoff asked for — served loop's scratch loads **686 → 0**, private **2,016 → 528
  B/lane**, VGPR **256 → 238**, `Stages 1 → 2` — and the served prefill rate did not move at all
  (−0.7%, inside spread). That is the handoff's warning, measured directly on the served shape:
  **static instruction count does not translate to speed.** The served FA2 kernel is not bound by
  those scratch loads. Any future lever must be justified by a measured share of prefill time, not
  by instruction counts.
- **Clean pair built to replace it** (`/home/vermont/tmp/l2-ab/build-arms-clean.sh`): both arms link
  **the same 16:25 tree archive** (`6791fe2212`) with the tree's own `link.txt`.
  BASE = `ninfer-serve.base2` = the tree's own 16:25 binary `0ab1eec3ad` (committed FA2 object
  `e9bd4fb67e`, no swap). CAND = `ninfer-serve.cand2` `62dc1e3d3b` = a copy of that archive with only
  `prompt_i8_fa2.o` replaced by the census cand object `b420769879`. FA2 symbol diff +79/−15 with
  **zero** removed `launch_fitted` instantiations; whole-binary diff +63/−3.
  Residual gap being closed (`/home/vermont/tmp/l2-ab/compile-treeflags.sh`): the census objects were
  compiled with the census flag line, which adds `-D__HIP_ROCclr__=1` and omits
  `-Wimplicit-fallthrough -Werror=implicit-fallthrough -ffile-prefix-map=…=.` relative to the tree's
  `flags.make` — the shape of the +79/−15 host-symbol delta. Both arms are being recompiled with the
  tree's exact flags and the per-kernel device sections compared; if the device asm is
  section-identical to the census arms, the flag delta is proven device-neutral and the linked pair
  differs only by the lever define.
- Next under lock: 4-leg ABBA on those arms (`/home/vermont/tmp/l2-ab/abba-l2b.sh`), prefill t/s and
  coherence, one flock hold. Target 300+ t/s prefill remains open.

### 2026-10-23 **the served attention route is BFloat16 — levers (l), (l2) and the qfrag census all measured a kernel the server never launches**

This is the mechanism behind every FA2 NULL in this file. It is a source reading, not a measurement,
and it is checkable line by line:

- `src/serve/serve_options.h:58` — `KvCacheStorage kv_cache = KvCacheStorage::BFloat16;`. No ABBA
  runner in `/home/vermont/tmp/*/` passes `--kv-dtype`, so every lever (l)/(l2)/qfrag run used the
  default.
- `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:461` —
  `if (storage == KvCacheStorage::BFloat16) return CausalAttentionRoute::Bf16;`.
- `src/ops/softmax_attention/dense/causal_cache/prompt.cu:11` — "BFloat16 storage never reaches
  these functions", i.e. the int8 FA2 prompt kernels.
- `src/ops/softmax_attention/dense/causal_cache/bf16/plan.cpp` — with
  `width > kGroupedPrefillMaxWidth (= 256)` the `Tiled128` instance is chosen only when
  `heads == 16 && width >= 1024 && width % 128 == 0 && visible-width >= 8192`; this model has
  `heads == 24` (model header: `num_attention_heads 24`, `num_key_value_heads 4`, `head_dim 256`),
  so prompt attention takes **`Tiled64`**.
- `src/serve/serve_options.h:42` and `include/ninfer/types.h:370` — `prefill_chunk = 1024`, so the
  served prefill really does run `width = 1024` chunks.

Therefore the kernel the served server launches for prompt attention is
`bf16_kv_tiled_mma_kernel<CausalGeometry<256,24,4>, Bf16KvTiledMmaSchedule<64,32,1>>`, and
`causal_attention_prompt_i8_fa2_kernel` — the kernel lever (i), (j), (l), (l2) and the qfrag census
all worked on, and the kernel whose in-loop scratch loads we drove 686 → 0 — **is not on the served
route at all** under default options. Every one of those verdicts is a verdict about an unserved
kernel. That does not retroactively change any recorded number, and all of those unfavorable results
stay in this file exactly as recorded; it does say why they were NULLs, and it retargets the search.

The occupancy term, from `bf16/schedule.cuh:69`:

```
bf16_kv_tiled_shared_bytes = (kQueryRows + 2 * kKeyRows) * kHeadDim * 2
```

`<64,32,1>` at `kHeadDim = 256` is `(64 + 64) * 512 = 65,536 B` — the entire RDNA3 LDS budget, so
exactly **one CTA per CU**. (Docs Table 2's row for this kernel — "266 spills, 1,104 B private" —
does **not** reproduce; the census section below measures 19–20 spills and 608 B private on this
exact instance, compiled with the tree's own flags. The 266/1,104 reading is kept in Table 2 as
recorded, and is superseded for this row.)
`bf16/lds_fit.cuh` descends only to `kBf16CausalLdsCeilingBytes = 64*1024` (halving KeyRows first,
then QueryRows); it never aims at the 32 KiB line, i.e. it never buys a second block per CU.
`(32 + 2*16) * 512 = 32,768 B` is exactly half, which is the 2-CTA/CU line. This is the first lever
with a measured time share behind it (next section); its target kernel is the served one.

### 2026-10-23 measured prefill breakdown (`NINFER_PREFILL_PROFILE=1` on the committed base arm)

No new code: the tree's own `PrefillProbe` (`src/models/qwen3_5/execution/text.cpp:269`) on
`/home/vermont/tmp/l2-ab/ninfer-serve.base2` (`0ab1eec3ad`, the tree's own 16:25 build, committed FA2
object `e9bd4fb67e`) — the same binary the clean ABBA uses as BASE. Runner
`/home/vermont/tmp/prefill-prof/pprof.sh`, ONE flock hold 17:44:11–17:45:28, lighthouse prompt
(4,215 tokens), `max_tokens=8`, log `/tmp/pp_pprof1.log`. Two identical requests: **177.7** and
**176.0** prefill t/s; the probe's own bucket totals agree between the runs within 1.4%.

Every number below is probe output. Shares are computed from the four 1,024-token chunks, which are
91% of prefill time.

| bucket | chunk 1 | chunk 2 | chunk 3 | chunk 4 | sum (4 chunks) | share |
|---|---|---|---|---|---|---|
| `gdn_mixer` | 1,247.2 | 1,243.8 | 1,247.6 | 1,247.9 | 4,986.5 ms | 23.1% |
| `gdn_mlp` | 2,065.0 | 2,065.7 | 2,065.6 | 2,065.8 | 8,262.1 ms | 38.3% |
| `attn_mixer` | 613.3 | **1,160.5** | **1,684.5** | **2,217.4** | 5,675.7 ms | 26.3% |
| `attn_mlp` | 666.6 | 656.3 | 661.8 | 660.5 | 2,645.2 ms | 12.3% |
| chunk total | 4,592.3 | 5,126.4 | 5,659.7 | 6,191.8 | 21,569.5 ms | 100% |

- **The quadratic term is real and it is attention.** `attn_mixer` grows **+547, +524, +533 ms** per
  added 1,024-token chunk while every other bucket is flat to within 0.5%. GEMM/MLP (the two `*_mlp`
  buckets) = **50.6%**, mixers = **49.4%**. These measured shares reproduce the roofline section's
  46.6 / 27.1 / 23.2% on the committed binary with the tree's own probe — the roofline section is
  confirmed, not assumed.
- **Attention alone cannot reach 300 t/s.** Deleting *all* prefill attention time (the entire
  `attn_mixer` bucket) leaves 18.9 s of probe-measured call time for 4,215 tokens ≈ **223 t/s**. Any
  route to 300 has to move the GDN and MLP buckets too; attention is a necessary, not sufficient,
  lever. (Probe overhead: the 9 calls sum to 24.5 s against a reported TTFT of 23.7 s, so the probe
  adds ~3.5%; shares are unaffected, absolute ms are ~3.5% high.)
- **The GDN mixer's own split says where its 23.1% goes.** Per sampled layer at T=1,024
  (n=16): `proj+conv 13.43`, `out 4.91`, `rec 4.74`, `nc 2.59`, `norm 0.23` ms = **25.9 ms**, against
  1,247.2 / 48 GDN layers = **26.0 ms/layer** for the whole bucket. The fine split accounts for the
  bucket to 0.3%, so `proj+conv` alone is **52% of the GDN mixer** ≈ 14% of chunk time, and the
  recurrent `rec` term is only 4.7 ms/layer.
- **Attention's own efficiency, derived from the above:** the model has 64 layers, 48 GDN + 16 full
  attention (3:1 `layer_types`, read from the model header). Chunk 4's 2,217.4 ms over 16 layers =
  138.6 ms/layer for ~3,670,000 causal query-key pairs × 24 heads × 256 dims × 2 passes ≈ 45 GMAC,
  i.e. **≈0.33 TMAC/s** — the same order as the roofline section's 0.28 TMAC/s mixer figure, ≈2% of
  the 16.55 TMAC/s wall. The served bf16 kernel (1 CTA/CU at 65,536 B dynamic LDS, 4 warps, 19–20
  VGPR spills, 26.3% of prefill time) is the most under-utilized large block identified so far.
- **Small prefill calls are expensive, and they are 12% of TTFT.** Same run, the 5 non-chunk calls:
  51 tokens → 410.1 ms, 2 → 415.2, 112 (the 4,215-tokens remainder) → 1,044.6, 5 → 647.2, 2 → 457.4.
  **172 tokens cost 2,974 ms = 17.3 ms/token**, against 5.27 ms/token for the chunked part — a 3.3×
  per-token penalty. At T=2 the buckets read `gdn_mlp 207.9`, `attn_mlp 69.2`, `gdn_mixer 100.0`: at
  T≤5 the MLP weight traffic dominates, and it costs as much as a 1,024-token chunk.

### 2026-10-23 **compile-only metadata census of the SERVED attention kernel — two levers closed before any GPU time**

No GPU, no lock: `bf16/launch.cu` compiled once with the tree's own flags
(`-O3 -DNDEBUG -std=gnu++17→gnu++20 -DNDEBUG -x hip --offload-arch=gfx1102 -save-temps=obj`, defines
`-DNINFER_BACKEND_HIP=1 -DNINFER_SM8X_COMPAT=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1`), `rc=0` at
17:50:19, script `/home/vermont/tmp/bf16-census/census.sh`, parser
`/home/vermont/tmp/bf16-census/parse2.py`, object
`/home/vermont/tmp/bf16-census/obj/launch-hip-amdgcn-amd-amdhsa-gfx1102.s`.

Parser note, because it produced two wrong readings first: this ROCm/LLVM emits kernel metadata **twice
and in two formats** — tab-indented `.amdhsa_kernel` asm directives (`.amdhsa_group_segment_fixed_size`,
`.amdhsa_private_segment_fixed_size`, …) near the kernel bodies, and a space-indented trailing table
(`.vgpr_count:`, `.vgpr_spill_count:`, `.private_segment_fixed_size:`, `.group_segment_fixed_size:`,
`.max_flat_workgroup_size:`) that has **no** `.amdhsa_metadata:` label line, so `yaml.safe_load` dies on
the tabs. VGPR and spill counts exist **only** in the trailing table. Within one entry the emit order is
`… .group_segment_fixed_size, .kernarg_*, .max_flat_workgroup_size, .name, .private_segment_fixed_size,
.sgpr_*, .symbol, .vgpr_* …`, so anchoring on `.name:` and reading every key after it silently attaches
the **next** entry's group size to the current kernel. Both effects were measured and fixed; the numbers
below come from the corrected split (pre-`.name:` keys vs post-`.name:` keys).

| instance | vgpr | vgpr spill | sgpr spill | private | group LDS | warps |
|---|---|---|---|---|---|---|
| **`bf16_kv_tiled_mma_kernel<…, Bf16KvTiledMmaSchedule<64,32,1>>`** (served, all 3 metadata variants × both geometries) | 256 | **19 – 20** | 0 | **608 B** | 0 (dynamic, 65,536 B set at launch) | 4 (128 thr) |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<32,32,1,2,0>>` (prefill w ≤ 256, not served at chunk 1024) | 256 | 164 – 165 | 0 | 1,088 – 1,104 B | 32,912 | 4 |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<64,32,1,2,0>>` (same) | 256 | 163 – 166 | 0 | 1,088 – 1,136 B | 32,912 | 4 |
| `bf16_kv_grouped_mma_kernel<…, Bf16KvGroupedMmaSchedule<16,32,2,2,1>>` (decode) | 256 | 95 – 99 | 0 | 1,024 – 1,056 B | 33,152 | 4 |
| `bf16_kv_merge_kernel<…, Bf16KvMergeSchedule<256,256>>` | — | 15 | 0 | 0 | 1,064 | 8 |

- **Docs Table 2's first row does not reproduce.** It lists the served tiled kernel at "266 spills,
  1,104 B private, group 0 – 33,152". The same TU compiled with the tree's flags gives **19 – 20 spills
  and 608 B** on that exact instance; 1,104 B and 163–166 spills belong to the *grouped* prefill
  instances, and 33,152 B belongs to the grouped decode instance. Table 2 stays in this file exactly as
  recorded; for the tiled row it is superseded by the measured numbers above.
- **The spill lever on the served kernel is dead, and it died statically.** 19 – 20 spilled VGPRs is
  608 B/thread of scratch; a CTA is 4 warps, so 2,432 B per CTA. The i8 FA2 kernel whose in-loop scratch
  traffic we spent levers (l)/(l2)/qfrag chasing carries 444–493 spills and 2.2–2.35 KB/thread — the
  served bf16 kernel is ~4× below that per thread and lives in a kernel that already runs 4 warps in one
  65,536 B LDS block. A ~76 B/lane spill reload cannot explain 0.33 TMAC/s against a 16.55 TMAC/s wall.
  Per the standing rule ("static instruction count alone did not translate to speed", "do not spend GPU
  time unless the served loop's ~685 scratch loads"): **no GPU time spent on this lever.**
- **The LDS-halving / occupancy lever is also dead, also statically.** `(32 + 2*16) * 512 = 32,768 B`
  would indeed fit 2 CTAs per CU, but `bf16/schedule.cuh` sets `kWarps = QueryTile / 16`, so a
  `<32,16,·>` CTA is 2 warps and 2 CTAs per CU is **4 warps per CU — exactly today's occupancy**, and
  `vgpr_count` is already at the 256 cap for every one of these instances (registers bind as hard as
  LDS does). Meanwhile the grid in `bf16/template_launch.cuh` is
  `KVHeads * ceil(width * GroupSize / kQueryRows)`, so halving `kQueryRows` from 64 to 32 **doubles** the
  CTA count per layer-call (at width 1,024: `4 * ceil(6144/64) = 384` → `4 * ceil(6144/32) = 768`), which
  doubles K/V re-read traffic — ~805 MB → ~1,611 MB per layer-chunk at avg visible 2,048, ≈143 ms →
  ≈286 ms across the 16 attention layers against a measured 613 ms for that chunk's whole `attn_mixer`
  bucket. Net: no added warps, more traffic. Deprioritized before GPU spend.
- `(also) Bf16KvInstance::Tiled128 = Bf16KvTiledMmaSchedule<128,32>` descends through
  `bf16/lds_fit.cuh` to the same `<64,32,1>` as `Tiled64` on gfx1102, so the `Tiled128` route buys
  nothing on this card — it is the same kernel the census measured.

### 2026-10-23 compile-flag confound in the l2 pair (verdict from the tree-flags control compile)

`/tmp/claude-1000/…/tasks/bgidbttd6.output`, exit 0, compiled 17:44:40 (base) and 17:49:04 (cand) with
the tree's exact flags, compared section-by-section against the earlier census-flags objects:

- **BASE:** 78 device sections (tree flags) vs 78 (census flags); **1** device-asm section differs —
  `causal_attention_prompt_i8_fa2_kernel<CausalAttentionGeometry<16,2,2,41>, PagedKVBatchMetadata<1>, 8, …>`.
- **CAND (`-DNINFER_FA2_DIET_L2`):** 66 device sections (tree flags) vs 78 (census flags); **77**
  device-asm sections differ.
- Objects: base(tree-flags) `e99e2929b0`, cand(tree-flags) `a2b63a8633`, cand(census-flags) `b420769879`.

So the flag delta **reaches device code**, not just the driver: the `base2`/`cand2` pair used by the l2
ABBA runs is confounded at the level of the machine code being timed, and its nulls cannot be attributed
to the lever alone. A confound-free l2 verdict needs a pair rebuilt from the tree-flags objects. It is
**deprioritized, not deleted**: after the route discovery above, l2's target kernel
(`causal_attention_prompt_i8_fa2_kernel`) is not on the served route, so GPU time spent cleaning up that
comparison buys a verdict about an unserved kernel. All l2 unfavorable results stay in this file exactly
as recorded, and this paragraph records *why* the pair was not rebuilt — it is not a re-litigation of the
measurement itself.

## 2026-10-23 machine-code census of the SERVED kernel — a correction, and where its work actually goes

All of this is compile-only: the served `bf16_kv_tiled_mma_kernel<…, Bf16KvTiledMmaSchedule<64,32,1>>`
body, section lines 8–21923 of
`/home/vermont/tmp/bf16-census/obj/launch-hip-amdgcn-amd-amdhsa-gfx1102.s`, **14,202 instructions**,
counter `/home/vermont/tmp/bf16-census/` (same flags as above, `rc=0`, no GPU, no lock).

**Correction, stated first because it invalidates an earlier claim.** In the session log I reported the
served kernel as containing "zero tensor-core instructions". **That was false.** My mnemonic prefix list
was `v_mfma`, `v_mma`, `v_mac`, `v_dot`, `v_bfmm` — and RDNA3's bf16 matrix mnemonic is
`v_wmma_f32_16x16x16_bf16`, which does not start with `v_mma`. The served body contains **64 of them**.
gfx1102 executes wave32 bf16 WMMA. Every statement built on the "no tensor core" reading is void.

| class | count | share | note |
|---|---|---|---|
| `v_wmma_f32_16x16x16_bf16` | **64** | 0.5% | see arithmetic below — exactly minimal, **no WMMA inflation** |
| `ds_bpermute_b32` | **812** | 5.7% | 12.7× the WMMA count; runs = 2×(exactly 64) + 526 singletons + 79 pairs; **498 of the runs sit inside `;;#ASMSTART`/`;;#ASMEND` inline-asm regions**; 128 lie within 70 instructions after a `v_wmma` |
| `ds_load_*` | 192 | 1.4% | with `ds_store_*` 84 |
| `s_delay_alu` | **1,630** | 11.5% | fixed-latency VALU padding the scheduler inserted — 1 in 9 instructions is scheduler filler |
| `v_cmp_ne_u32` | 1,210 | 8.5% | + 491 `v_cndmask`, 170 `s_and_saveexec`, 242 `s_cbranch_exec`: heavy per-lane divergence |
| `s_waitcnt` | 582 | 4.1% | 206 `vmcnt(0)`, **94 `lgkmcnt(0)` (full LDS/global drain)**, 87 `lgkmcnt(1)`, 51 `lgkmcnt(3)`, 29 `lgkmcnt(2)`; **18 sites are `lgkmcnt(0)` feeding a `v_wmma` directly** |
| FMA family | 620 + 518 + 366 dual + 176 mul + 162 add3 = 1,842 | 13.0% | softmax exp/pack/alpha, not the contractions |
| scratch | 201 loads, 135 stores | — | 19–20 spills × 2 `step<true>`/`step<false>` instantiations. **The handoff's "~685 scratch loads" does not reproduce in this body** (336 scratch ops total here); kept as recorded, superseded for this kernel |

The WMMA count is *arithmetically* minimal, which is what makes the shuffle count the finding. Per warp
per key tile at the served instance (16 query rows per warp, `Bc = 32` keys, `D = 128`):
QK = 16 × 32 × 128 = 65,536 MACs = 16 wave32 WMMA; PV = 16 × 128 × 32 = 65,536 = 16 WMMA. 32 per warp per
tile × 2 tile variants (full + masked) = **64, exactly the measured number**. Nothing is being wasted on
the matrix side; the waste is all in moving fragments between layouts.

Two structural facts, each verified by reading the source:

1. **HIP `cp.async` is fictional.** `src/ops/common/memory.cuh:47-82`: under `__HIPCC__`,
   `cp_async<Bytes>` is a synchronous `store_vec(load_vec(gmem))`, `cp_commit()` is `{}`, and
   `cp_wait<Groups>()` is only a `static_assert`. The tree's own comment says "RDNA3 has no async-copy
   instruction to route it to… The overlap is what is lost, not correctness." Consequence for the served
   per-tile schedule: V staging (`tiled_mma.cuh:148`) and K(kb+1) staging (240–245) each sit behind a
   `cp_wait<0>()` + `__syncthreads()` that do nothing, so **both global round trips are paid in full**,
   in place, separated by barriers, on a CTA that is 4 warps in 1 CTA/CU. The 94 `lgkmcnt(0)` drains and
   the 18 drain→WMMA pairs above are that schedule made visible. Verified non-finding: the served staging
   path `bf16/tile_io.cuh` uses `cp_async<16, Cache::cg>` plus an explicit zero `store_vec`, and does
   **not** call `cp_async_zfill`, so that function's HIP byte-loop form is not a served-route cost.
2. **QK was hand-ported to native WMMA; PV was not.** `tiled_mma.cuh:161-196` is inside
   `#if defined(__HIPCC__)` and calls `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` directly — then pays
   32 `__shfl` per `nt16` (64 per lane per key tile; the two exactly-64 `ds_bpermute` runs) to move the
   WMMA C fragment into the `score[nt][0..3]` m16n8k16 layout. The PV site (`tiled_mma.cuh:344-376`) has
   **no HIP branch at all**: per lane per key tile it issues `PVLoads = PVKs * PVHalf = 32`
   `ldmatrix_x4_t` and 64 `mma_bf16` calls, and both are raw PTX strings —
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`src/ops/common/mma.cuh:482`) and
   `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` (`mma.cuh:470`). gfx1102 has neither instruction, so
   the PV path runs through ROCm's PTX translation: m16n8k16 lowered onto 16×16×16 hardware with fragment
   shuffles (16 WMMA per warp per tile — the arithmetic above accounts for them), `ldmatrix…trans`
   lowered to LDS reads plus permutes. That is where most of the 498 `;;#ASMSTART`-delimited
   `ds_bpermute` runs come from, and it explains the 1,630 `s_delay_alu` padding and the divergence
   counts: the translator is repacking bf16 fragments per lane inside the innermost loop.
   Control evidence in the same object: `bf16_kv_grouped_mma_kernel` (the small-`w`/decode route, not
   served at chunk 1024) has the same `mma_bf16` PTX with **0 WMMA and 2,503–2,535 FMA-family
   instructions** plus 1,608–1,624 `ds_bpermute` — i.e. there the same fragment form is lowered to an FMA
   loop. Static instruction counts do not rank those two (a loop's count is not its work); this is
   recorded as a profile, not as a speed claim.

Per lane per key tile, measured (not estimated): **32 WMMA contractions** (minimal) against **~64
`ds_bpermute` from the tree's QK→m16n8 packing plus a few hundred more from the translated PV fragments
and the warp reductions**, 96 LDS loads, 2 full drains behind 4 barriers, 4 warps per CU, 1 CTA per CU.
On RDNA3 every wave32 `__shfl` is a `ds_bpermute` — an LDS-pipe instruction, not the 1-cycle VALU shuffle
it is on wave64 parts — so the served softmax path's dominant per-tile movement is LDS-pipe traffic, and
`cp_wait` being a no-op means none of it is overlapped.

### Open lever, sketched and NOT validated — no GPU time spent

PV is the only matrix product in the served kernel that reaches hardware exclusively through the PTX
translation layer; QK shows the native route in the same file. Porting PV the way QK was ported — direct
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` fed from the QK accumulator — would delete the tree's 64
permutes per lane per tile *and* the translator's fragment shuffles. Two open questions must be settled
before a single GPU cycle:

- **Layout.** WMMA C per lane is 8 fp32 contiguous along N for a fixed row; WMMA A per lane is 8 bf16
  contiguous along K for a fixed row; P's K axis *is* QK's N axis, so the mapping appears identical and P
  could be packed in place with no lane movement. That "appears" is derived from the tree's own shuffle
  pattern (`gid`, `gid+8`, `gid+16`, `gid+24` gathers at `j = 2 * lid`), which is evidence about layout,
  not proof of it — it has to be confirmed against the gfx11 w32 fragment layout with a numeric check.
- **The B operand.** PV's B (V, keys × dims) needs 8 contiguous *keys* per lane for a fixed dim group.
  LDS holds V key-major in 256 B rows, so a native B read is 8 strided 2-byte LDS reads per fragment
  (128 per lane per tile — worse than what is there now). Making B native needs V **transposed in LDS**,
  and transposed staging costs scalar LDS stores on the global→LDS path, which is already synchronous
  (finding 1). That is the lever's real cost, and the reason it is not being presented as a quick win.

Scope honesty: `attn_mixer` is 26.3% of prefill, so this lever cannot reach 300 t/s by itself — deleting
all attention time ceilings at ~223 t/s. `gdn_mixer` (23.1%) calls the same WMMA builtin
(`src/ops/linear_attention/gated_delta_net/chunked/recurrence_wmma.cuh:214`), so the same
native-vs-translated question applies there; the census TU above contains only the bf16 causal-cache
translation unit, so **the GDN kernels' shuffle profile is not yet measured** and its census is the next
compile-only step.

## 2026-10-23 GDN recurrence census — the native-WMMA port is already proven on this hardware, and the lever's scope is 26.3%, not 49.4%

Compile-only, `/home/vermont/tmp/gdn-census/{census.sh,section.py}` on
`src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu` (tree flags, gfx1102, `rc=0`, no GPU,
no lock). Which kernel is served: `launch()` at `recurrence.cu:528` sends `DV == 64` to
`chunk_recurrence_wmma_kernel` on HIP with the gate `rec_wmma_native()` **taken by default** (true when
`NINFER_GDN_REC_WMMANATIVE` is unset), and `value_tile()` returns 64 whenever
`value_heads * 16 > multiprocessor_count` (16 CUs on this box). So the served GDN recurrence here is the
native-WMMA kernel — `recurrence_wmma.cuh:214` calls
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` directly, with no PTX string involved.

**Script bug, recorded because it briefly produced a fake result.** The first version of `section.py`
counted LDS traffic under the `ds_read`/`ds_write` prefixes and printed `ds_load 0` for the served GDN
kernel. gfx1102 emits `ds_load_*`/`ds_store_*`. A raw mnemonic histogram of the same region (88
`ds_load_2addr_stride64_b32`, 64 `ds_load_u16`, 64 `ds_load_u16_d16_hi`, 51 `ds_load_b128`) exposed it;
the table below is the corrected run. The attention census above used the `ds_load_*` prefix, so its LDS
numbers were never affected.

Served-kernel comparison (per-kernel region split on `.Lfunc_endN`, verified against the `.size` lines
that name each region):

| | attention `bf16_kv_tiled_mma_kernel<64,32,1>` | GDN `chunk_recurrence_wmma_kernel<64>` |
|---|---|---|
| instructions | 14,202 | **3,549** |
| `v_wmma` | 64 (arithmetically minimal) | 38 |
| `ds_bpermute` | **812 = 12.7× WMMA** | **0** |
| `ds_load` / `ds_store` | 192 / 84 | 302 / 65 |
| scratch load/store | 201 / 135 | **0 / 0** |
| VGPR / spills | 256 / 19–20 (608 B private) | **237 / 0 (0 B private)** |
| `s_delay_alu` | 1,630 | 217 |
| `lgkmcnt(0)` drains | 94 | 0 |
| barriers | 4 | 2 |

The GDN kernel builds its WMMA fragments with `ds_load_u16`/`ds_load_u16_d16_hi` strided 16-bit LDS
loads (128 of them) and `v_perm_b32` 32-bit packing (144), and uses **zero** cross-lane permutes. That
is the same contraction family the served attention kernel handles with 812 permutes, run on the same
card in the same build. The GDN SIMT fallbacks — `chunk_recurrence_kernel<8/16/32/64>`, which the HIP
route does not launch at `DV == 64` — keep the PTX form and show its cost in the metadata:
**306–427 VGPR spills per lane, 1,100–1,616 B private, 992–1,248 `ds_bpermute`, 0 WMMA, 13.5k–17k
instructions.** So on this box the native-WMMA port is an in-tree, already-shipped technique whose
symptom-removal (spills, permutes) is visible in the compiler output; the served attention PV site is
the remaining place on the prefill route where a contraction goes through the PTX translator.

**Scope, settled by this census:** the lever applies to the served attention PV site only —
`attn_mixer` = 26.3% of prefill — not to `gdn_mixer` (23.1%), which already runs the native form.
Ceiling unchanged: attention-only cannot reach 300 t/s (~223 t/s if all its time vanished).

### The w32 fragment layouts, established empirically (no GPU run)

The served QK site *is* a layout proof: it executes on the served kernel and produces coherent text, so
its addressing rules are measured facts, not inferences. From `tiled_mma.cuh:54-55`
(`gid = lane >> 2`, `lid = lane & 3`), 168–169, 184–189:

- **A operand, per lane:** one M row, 16 contiguous K values (`query_row = warp_row0 + (lane & 15)`;
  lanes 16–31 mirror rows 0–15); loaded as 2 × `int4` = 32 B in one fragment.
- **B operand, per lane:** one N row, 16 contiguous K values
  (`key_row = nt16*16 + (lane < 16 ? lane/2 : 8 + (lane-17)/2)` → 2 lanes per N row).
- **C operand, per lane:** 8 contiguous N values at row `lane & 7`, chunk index `lane >> 3`; the four
  lanes `gid, gid+8, gid+16, gid+24` tile all 16 N values of row `gid`, and `j = 2*lid` indexes pairs
  inside the 8-value chunk. This is exactly what the 32 `__shfl` per `nt16` buy, and what the PV
  epilogue (386–387: `d0 = n*8 + 2*lid`, `qrow0 = q0 + warp_row0 + gid`) consumes directly, unshuffled.

With those rules the PV lever's open questions resolve to counted costs, not unknowns. PV wants
A = P (M = 16 query rows, K = 32 keys) and B = V (N = 128 dims, K = 32 keys):

- **A is buildable from QK's C with 2 gathers per fragment.** C at row `lane & 7` holds 8 contiguous
  keys (QK's N *is* keys); a full 16-key window for row `r` lives in lanes `r, r+8, r+16, r+24`. Per
  lane per key tile: 2 A fragments (K = 32) × 2 source lanes × 4 dwords = **16 `ds_bpermute`**, versus
  the tree's current 64 (`__shfl` packing) **plus** the translator's fragment shuffles.
- **B needs V transposed in LDS, and that is the lever's real cost.** Native B per lane = 16 contiguous
  keys at a fixed dim row; LDS holds V key-major, so a native B read is 8 strided 2-byte LDS loads
  (128 per lane per tile — worse than today). Staged Vᵀ (32 keys × 128 dims = 4,096 elements per CTA
  tile) costs ~32 scalar 2-byte LDS stores per lane per tile, replacing today's 16 `int4` stores, and
  buys 8 native B fragments × 2 × `int4` = 16 vector LDS loads with no permutes. GDN's use of
  `ds_load_u16` strided loads shows the 16-bit LDS access pattern is cheap on this hardware; what is
  unproven is whether paying it at staging time (which is synchronous — finding 1 above) beats the
  translator's per-tile repacking.
- Static budget per lane per tile, native form: **16 WMMA + ~16 permutes + 16 vector B loads + ~32
  transpose stores**, against a measured current form of 16 WMMA + 32 `ldmatrix_x4` + 64 PTX `mma.sync`
  calls + a large share of the 812 permutes + 1,630 `s_delay_alu` + 94 full drains.

This is a counted design, not a validated lever: no GPU time has been spent on it, and no speed claim is
made. The next step is a compile-only prototype behind a macro, then route qualification, per the
standing commit-scope rule (attention WMMA work lands only after build + route qualification).

## 2026-10-23 D=256 correction to the counted design, and the native-PV prototype's compile-only census: the permute storm goes, the register budget does not

**No GPU time was spent in this section.** Compile-only census (`/home/vermont/tmp/pv-census/`,
`census.sh` / `analyze.sh` / `census2.sh`, tree flags, `--offload-arch=gfx1102`, `rc=0`), plus a bank
model written as `mapping.py`. No kernel launched, no t/s claimed, and nothing below is a speed claim.

### Correction: the served instance is D = 256, and the measured 64 WMMA is one tile pass, not two

The arithmetic above was written at `D = 128`. The served instance is confirmed from the mangled name in
the generated `.s`: `bf16_kv_tiled_mma_kernel<CausalGeometry<256,24,4>, Bf16KvTiledMmaSchedule<64,32,1>,
PagedKVDirectMetadata>`, `.amdhsa_num_vgpr 256`. At `D = 256`, `Bc = 32`, per warp per key tile:

- QK = `(Bc/16) × (D/16)` = 2 × 16 = **32 WMMA**; PV = `(D/16) × (Bc/16)` = 16 × 2 = **32 WMMA**.
- **64 in a single instantiation**, which is the measured number. The reading "32 per warp per tile × 2
  tile variants (full + masked)" is superseded: `step<true>` and `step<false>` live in the same function
  body (labels `.LBB0_745` and `.LBB0_1429` of the 14,202-instruction region) and share one matrix body.
  The D=128 arithmetic reached 64 by 16 + 16 × 2 variants — a coincidence of D halving. The conclusion
  it supported ("no WMMA inflation") stands; the derivation was wrong.
- Consequential corrections to the bullet list above, all at D=256: native B per lane per key tile =
  `CTiles(16) × NT16(2) × 2 int4` = **64 LDS128 loads**, not 16; staged Vᵀ = 32 keys × **256** dims =
  8,192 elements = 16 KB per CTA key tile = **32 u32 stores + 8 int4 global loads + 32 PRMT per lane per
  tile**; N = 256 dims, not 128.

### The prototype and its qualification result

`NINFER_BF16_PV_NATIVE` (in `.../causal_cache/bf16/tiled_mma.cuh`, **default OFF** — the served build is
the control) swaps the PV region to direct `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`: scores stay in
the C-fragment layout, P reaches the A fragment through `__shfl` source-lane gathers, V is staged
transposed (`bf16_kv_stage_vt` in `.../bf16/tile_io.cuh`) so a B fragment is two `int4` LDS reads.
Served region (`G<256,24,4> S<64,32,1>`, the instantiation the server runs), **measured**:

| | control (off) | prototype (=1) | Δ |
|---|---|---|---|
| total instructions | 14,202 | 7,281 | **−48.7%** |
| `ds_bpermute` | 812 | **36** | **−95.6%** |
| `v_wmma` | 64 | 128 | 2 code copies (full + masked), 64 each |
| `ds_load` | 192 | 320 | +128 = 2 copies × 64 B-fragment reads — the predicted count |
| `ds_store` | 84 | 116 | +32 |
| `global_load` / `global_store` | 84 / 64 | 68 / 16 | −16 / −48 |
| scratch load / store | 201 / 135 | **576 / 514** | **+375 / +379** |
| `s_waitcnt` | 582 | 702 | +120 |
| `s_delay_alu` | 1,630 | 463 | −71.6% |
| `v_cmp` / `v_cndmask` | 1,581 / 491 | 366 / 253 | divergence machinery collapses |
| VGPR / **spills** / private | 256 / **19–20** / 608 B | 256 / **1,060–1,080** / 2,472–2,520 B | **fails** |

Control reproduces the recorded baseline exactly (14,202 instructions, 64 WMMA, 812 permutes, 201/135
scratch, 256 VGPR, 19–20 spills, 608 B) — machine-code evidence that the served path is unchanged by the
prototype's presence behind the default-OFF macro.

**The prototype fails the qualification rule** ("WMMA count must stay, permutes must fall, spills/priv
must not grow"): the permutes do fall (812 → 36) and the instruction count halves, but spills go up 53×
and private memory 4×. Root cause is liveness, visible in the source: the output accumulator is
`acc16[16][8]` = **128 fp32 live per lane** (the whole 16 rows × 256 dims, inherent to the C fragment),
plus both A fragments (32 floats), plus `cd`/`pw`, inside a fully unrolled 16 × 2 loop that presents 64
*independent* LDS128 loads to the scheduler at once. At 256 VGPR / 4 warps / 1 CTA per CU that does not
fit. The PTX-translated route runs the same 128-float accumulator at 19–20 spills today.

`NINFER_BF16_PV_NATIVE=2` reorders the same route (contraction group outer, output tile inner with
`#pragma unroll (1)`) so only one A fragment is live across the output tiles and the loads enter through
the WMMA dependency chain. That is a register-discipline hypothesis; its census is compile-only and
recorded by `census2.sh`. Whatever it measures, **no GPU cycle has been spent on this lever and no speed
claim is made for any of the three variants.**

### Derived: the Vᵀ layout's LDS behaviour, including the cost that the counted design understated

Bank model (derived arithmetic on the layout contract, not a measurement): a u32 store at element index
`e` occupies bank `(e*2/4) mod 32`; an `int4` (`LDS128`) access is 8 lanes/pass with each 16 B window
covering 4 banks; windows that are identical merge, windows that overlap do not.

- **Reads: conflict-free, as designed.** With `granule = (key>>3) ^ (dim&3)`, each 8-lane quarter of a
  B-fragment read covers 4 distinct dim rows with distinct granules; `mapping.py` measures 0 overlapping
  windows per pass, plus 4 identical-window merges per group for both key groups. The claim in
  `tile_io.cuh`'s comment checks out under this model.
- **Stores: 2-way conflicted on every wavefront, and structurally so.** For the key-pair u32 packing,
  `bank = 16*(dim&1) + 4*((pair>>2)^(dim&3)) + (pair&3)` — **independent of the dim chunk**. A 32-lane
  wavefront therefore covers 16 distinct banks at best, and a 32-key row has exactly 16 pairs, so each
  pair must repeat twice: every one of the 32 store wavefronts per warp per key tile is 2-way conflicted
  (64 LDS store passes against a 32 lower bound). Pigeonhole, not a mapping artifact — the search in
  `mapping.py` finds **no** lane→(pair, chunk) mapping with zero store conflicts (floor 2), because
  flipping the parity term needs an odd dim offset and every vector-loadable staging offset is even.
- Today's key-major `int4` staging is store-conflict-free (consecutive lanes, consecutive granules).
  **Vᵀ staging therefore costs +32 LDS store passes per warp per key tile** (32 → 64) on top of the
  +24 store instructions and the PRMTs: a real, derived cost that the bullet list above ("replacing
  today's 16 int4 stores") understated.

## 2026-10-23 native-PV value 2 closes the lever; the T2 GEMM activation-scale window measured; two of my own source-level readings corrected

**No GPU time was spent in this section.** Everything below is a compile-only census
(`/home/vermont/tmp/pv-census/census2.sh`, `/home/vermont/tmp/t2-census/census.sh`,
`/home/vermont/tmp/t2-census/census_vec.sh`, tree flags, `--offload-arch=gfx1102`, `rc=0`) and a
width census over the generated `.s`. No kernel launched, no t/s claimed, nothing below is a speed claim.

### Native-PV value 2 (`=ser`) fails, and its `ds_store` delta confirms the derived Vᵀ cost exactly

`census2.sh` compiles the three variants in one recipe (control = macro unset, `=1` output-tile
outer fully unrolled, `=2` contraction-group outer with `#pragma unroll (1)` on the inner loop),
served instance `CausalGeometry<256,24,4>` / `Bf16KvTiledMmaSchedule<64,32,1>`, served region
`...ScheduleILi64ELi32ELi1EEENS0_21paged:Direct`:

| served region | control (off) | `=1` | `=2` |
|---|---|---|---|
| total instructions | 14,202 | 7,281 | **7,202** |
| `v_wmma` | 64 | 128 | **68** |
| `ds_bpermute` | 812 | 36 | **36** |
| `ds_load` | 192 | 320 | **200** |
| `ds_store` | 84 | 116 | **116** |
| `global_load` / `global_store` | 84 / 64 | 68 / 16 | 68 / 16 |
| scratch load / store | 201 / 135 | 576 / 514 | **737 / 658** |
| `s_waitcnt` | 582 | 702 | **769** |
| VGPR / **spills** / private | 256 / **19–20** / 608 B | 256 / **1,060–1,080** / 2,472–2,520 B | 256 / **634–636** / **1,856 B** |

The register-discipline hypothesis is **refuted as a pass**: reordering fixed the permute storm
(812 → 36, the same as value 1) and halved the region (14,202 → 7,202), and it did cut spills 1,070
→ 635 — a 46% cut, which is exactly what "only one A fragment live across the output tiles" was
worth. It is nowhere near enough: 634–636 spills against the control's 19–20, private 1,856 B
against 608 B, scratch traffic **up** 737/658 against 201/135, `s_waitcnt` up to 769. The
qualification rule is "spills/priv must not grow"; it grows ~32×, so **the native-PV lever is closed
at all three values on static evidence and owes no GPU time.** The 128 live fp32 accumulator per
lane (the C fragment at D=256) is the wall, not the schedule around it — a 64-WMMA route cannot be
made to fit 128 accumulator floats + fragments at 256 VGPR / 4 warps / 1 CTA per CU by reordering.

`ds_store` 84 → 116 = **+32 on both native variants**, which is the derived Vᵀ store cost
(+32 LDS store passes per warp per key tile, from the 2-way pigeonhole conflict) to the instruction.
The bank model's prediction landing on the compiled count is the strongest confirmation in this
section: the layout's cost is understood, and it is not the reason the variant fails.

### The served T2 GEMM's activation-scale fold: written as a vector window, the win is registers, not loads

`serve_options.h:74 prefill_a8 = true` routes T2_G128_FP16 prefill linears through
`t2_prefill_i8_quantize_kernel` + `t2_prefill_wmma1_kernel<1,2,Panel=32,SplitStoreEpilogue>`
(cal41 panel raster; `NINFER_T2_A8_TILE` on, `NINFER_T2_A8_WMMANATIVE` on, `NINFER_T2_A8_WMMAPANEL=32`,
`NINFER_T2_A8_COLSPLIT` off per cal65/cal66). mlp buckets are 38.3% + 12.3% = 50.6% of prefill time.
The fold's per-group activation-scale reads were written as eight scalar `__half2float` reads per lane
per n (`t2_prefill_wmma.cuh`), which reads like 16 scalar 2-byte global loads per lane per group.
Changed to one explicit 16-byte window load per lane per n (values, order, and the `col < tokens`
predicate on the FMA untouched, so outputs are bit-identical by construction).

Compile-only census of the served GEMM, control vs changed (`census.sh` / `census_vec.sh`, same
recipe, quantise-kernel metadata identical in both as the control check):

| | control | xsv window | Δ |
|---|---|---|---|
| VGPR | 128 | **74** | −54 |
| **spills (vspill)** | **8** | **0** | −8 |
| private scratch | 36 B | **0 B** | −36 B |
| scratch load / store | 2 / 2 | **0 / 0** | −2 / −2 |
| `global_load` (all widths) | 24 | 21 | −3 |
| total instructions (region) | 1,452 | 1,445 | −7 |
| `ds_load` / `ds_store` | 128 / 32 | 128 / 32 | **unchanged** |
| `global_store_b16` | 16 | 16 | unchanged |
| cvt | 17 | 17 | unchanged |
| `s_waitcnt` | 111 | 115 | +4 |
| `s_delay_alu` | 126 | 132 | +6 |
| dual-issue | 32 | 29 | −3 |

**Two self-corrections, both from the machine code, both recorded because the source-level reading
said otherwise:**

1. **The eight scalar scale loads were already vector loads.** Width census over the control's served
   region: `global_load_b128 19`, `global_load_b32 1`, `global_load_b64 1`, `global_load_u16 2`,
   `global_load_d16_b16 1` — the backend coalesces the eight contiguous halfs into `b128` today. The
   change's measured effect is **register/spill relief** (128→74 VGPR, 8→0 spills, 36 B→0 private,
   scratch ops gone) plus **3 fewer load instructions**, **not** a traffic reduction. The comment in
   the source claimed "2 global loads per lane per group instead of 16"; corrected in place.
2. **There is no PRMT/perm table widening in this kernel.** The served region contains **zero**
   `v_prmt` and zero `v_perm`. The inherited note that `wmma_code_frag` widens codes through a PRMT
   table does not describe the shipped code path.

**Newly measured, not yet examined:** the weight-code widening in the served group body runs through
`v_bfe_u32` bit-extracts with **128 byte-granular LDS reads (`ds_load_u8`) and 32 LDS word stores
(`ds_store_b32`) per group body**, gated by `s_waitcnt lgkmcnt(N)`. That is the largest single LDS
traffic item in the served GEMM (the fold, the stores, and the WMMA operands are all smaller), it is
byte-granular, and nothing has been written about it. It is a candidate lever for wider LDS access;
no design has been counted for it and no GPU time has been spent.

Instruction-count is the metric this kernel has already taught us not to trust — "static instruction
count alone did not translate to speed" — and this change's count delta is −7 with `s_waitcnt` +4 and
`s_delay_alu` +6. Its favourable axis is register pressure, which has translated before (cal22's hoist
kept the tile off the spill heap; cal59's diet; cal61's spill geometry). So the change is a
**candidate that must earn its place by build + route qualification**, not a result, and it is not
committed as a win.

---

## 2026-10-23 (later) — the activation-scale window measured **−25% and was reverted**; the LDS code-widening
mechanism found and priced; the archive-swap bug that made two arms byte-identical

**GPU time: one ABBA set (4 server legs) inside a single `flock /tmp/ninfer-gpu.lock` hold. Card idle before
the set (card1 6% busy, 27 MB VRAM used), lock free, no server running. No GPU time for the widening design
(design 15) — see the verdict.**

### 15. The `xsv` 16-byte activation-scale window: qualified, lost, reverted.

The candidate described above (one `int4v` window load per lane per `n`, `__low2float`/`__high2float`,
values/order/`col < tokens` predicate unchanged) was built into two arms that differ by exactly one object and
run B,C,C,B. **It is a large loss. It is not committed; the source is reverted to HEAD.**

| leg | binary | prefill t/s | TTFT | decode | generated text |
|---|---|---|---|---|---|
| base1 | base | **178.8** | 23.6 s | 13.1 | sha1 `77f1779eb831`, 574 chars |
| cand1 | cand | **143.8** | 29.3 s | 13.0 | sha1 `77f1779eb831`, 574 chars |
| cand2 | cand | **121.9** | 34.6 s | 12.9 | sha1 `77f1779eb831`, 574 chars |
| base2 | base | **175.3** | 24.1 s | 12.9 | sha1 `77f1779eb831`, 574 chars |

Base mean **177.1** (off-leg spread 3.5 = 2.0%), cand mean **132.9**. **−44.2 t/s = −25%.** The legs do not
overlap: the best cand leg (143.8) is 31.5 t/s below the worst base leg (175.3) — nine times the base spread,
so the sign is settled by this set and does not need a repeat. What the set cannot settle is the magnitude
(cand's own spread is 21.9, six times base's).

- The generated text is **identical on all four legs** (same sha1, same 574 chars, same 8-gram probe), which is
  what the construction promises: the lever changes loads, not values. Coherence bar passed on every leg.
- **Decode is unchanged** (13.1/12.9 vs 13.0/12.9), consistent with a prefill-only route.
- **Drift is excluded by the ordering.** The slowest leg of the four is cand2; the leg that follows it, base2,
  returns to base level (175.3). A card-side trend would have dragged base2 down with it.
- The static profile was uniformly favourable: 128→74 VGPR, 8→0 spills, 36 B→0 private, scratch loads/stores
  2/2→0/0, 3 fewer `global_load`, −7 instructions in the served region. It lost by 25%. This is now the
  strongest instance on this kernel of the rule inherited from cal59/cal61: register relief and instruction
  count are not the axis this kernel runs on. The cand legs' 6× wider spread also suggests it runs in a worse
  regime, not merely a slower one.

**Provenance, because the first arm build was invalid.** `ar r` substitutes an archive member **by basename**.
The tree's member is `…/linear/t2/t2_a8.cu.o`; my staged object was named `t2_a8.o`, so `ar r` *appended* it and
the archive kept the tree's original. The linker took the original from both archives and both arms were the
tree: `ARMS IDENTICAL - no lever` (both `0ab1eec3ad`). Fixed in `/home/vermont/tmp/t2-ab/link-arms-t2.sh`,
which now gates on (i) exactly one `t2_a8.cu.o` member, (ii) no stray `t2_a8.o`, (iii) the archived member
byte-identical to the censused object, and (iv) the executable's embedded `.hip_fatbin` payload differing —
the decisive check for a HIP fat binary, since the device code lives in that 384,624,368-byte section.

Qualifying arms: `ninfer-serve.base` sha1 `a5222e435d…`, `ninfer-serve.cand` `072217e9b0…` (both 409,788,456 B),
anchor `4f6d892f91…` (b78f2c94, 178.4 t/s cold, coherent) carried forward as a lineage reference, not as the
base arm. Device payloads `fatbin.base` `88c06f0a7c` vs `fatbin.cand` `fd695f4a6e`, same size. At object level
exactly the **six `t2_prefill_wmma1` instantiations** differ; every other device section in the TU is identical
(72 vs 72 sections, 0 only-in-one, 6 content diffs), and the 54 `t2_prefill` symbol names are identical.
`link-run.log` records an earlier hash pair (`502264ab09`/`710d7fc2fc`) for the same two paths; three
re-reads of the files, the fatbin dumps, and the runner's own header all give `a5222e435d`/`072217e9b0`, so the
log's pair is the outlier (sha1 of a 409 MB file taken while the linker was still flushing) and the table above
names the binaries that ran. `asmcmp.py`'s pair labels are hard-coded, so the labels printed for this pair are
cosmetically wrong; the counts are the substance.

Flag-gap check, kept because it is what makes the census transferable: compiling the HEAD source with the
tree's **full** command (`-lineinfo -D__HIP_ROCclr__=1`) gives metadata identical to the pair-recipe build
(128 VGPR / 8 spills / 36 B private, 6/6 served GEMM sections byte-identical), and the pair-recipe build is
metadata-identical to the censused build. So the 128→74 verdict describes the arms that ran, and the earlier
"128 VGPR / 0 sgpr" for the pair-recipe candidate was a parse artifact of a truncated `.s` (the compile was
killed under memory pressure: swap 6.4/8 GiB).

### 15a. Design 15, priced. Why the LDS byte-staging is there, and why widening it in registers is not free.

The `ds_load_u8`/`ds_store_b32` traffic noted above is **LLVM emulating a variable-selector `PRMT`**, proven in
the served body (`t2_prefill_wmma1_kernel<1,2,32,SplitStoreEpilogue>`, 1386 instructions, 128 `ds_load_u8`,
32 `ds_store_b32`):

```
v_dual_mov_b32 v124, 0xfffe0100 :: v_dual_mov_b32 v123, 0      ; kTable, the sign-extend LUT
global_load_b128 v[0:3], v[92:93], off
ds_store_b32 v111, v124                                        ; table staged into LDS
s_waitcnt vmcnt(0)
v_and_b32_e32 v4, 3, v0                                        ; selector = a 2-bit code field
v_bfe_u32 v5, v0, 4, 2
   …  (four ds_load_u8 per site read table[selector] byte)
```

`wmma_code_frag` calls `__byte_perm(kTable, 0, data)`; RDNA3's `PRMT` selector is **immediate-encoded**, so a
data-derived selector cannot compile to `PRMT`, and LLVM puts the 4-byte table in LDS and indexes it per byte.
32 sites × (1 store + 4 byte loads) = 32 `ds_store_b32` + 128 `ds_load_u8`, exactly the census. **This
refines my own correction #2 in the previous section**: the statement "there is no PRMT/perm table widening
in this kernel" was right that no `v_prmt`/`v_perm` is emitted, and wrong in its conclusion — the table
widening *is* the shipped path, in LDS form. That is also the largest LDS item in the served GEMM.

The widening itself is `(x ^ 2) - 2` per 2-bit field, i.e. sign-extended signed values `0→0, 1→1, 2→−2, 3→−1`.
A register-only construction must be **per-byte**: a packed-lane subtract borrows across byte lanes, and my
first attempt (`s + 0x7C7C7C7C - ((s & 0x02020202) << 6) - 0x7C7C7C7C`) failed on 65528 of 65536 16-bit probes
for exactly that reason. The borrow-free form uses the per-byte GCN op (`__vsub4` = `v_sub_u8`):

```
s    = (b & 0x03) | ((b & 0x0C) << 6) | ((b & 0x30) << 12) | ((b & 0xC0) << 18)
bias = (s & 0x02020202) << 1          // 4 in lanes whose bit1 is set
y    = __vsub4(s, bias)               // 0→0, 1→1, 2→0xFE, 3→0xFF
```

Verified bit-exact against `__byte_perm` on **all 256** packed codes, and it reproduces `0, 1, −2, −1`.

> **Correction (2026-10-23, measured):** that verification is **CPU emulation of a form that does not compile here**. A
> builtin probe over this toolchain (clang shipped with ROCm, `gfx1102`, `-x hip`) finds **no `__vsub4`** and no
> `__builtin_amdgcn_vsub4`/`vadd4`/`vabsdiff4`/`permb16b32`/`v_perm`/`v_permlbrc`; the only byte-level intrinsics that
> compile are `__builtin_amdgcn_perm` (V_PERM_B32) and `__byte_perm`. So the "256/256 verified" widening above has
> never been through the assembler, and the 13-VALU-site cost below is a hand count of an unbuildable form. The
> register-only route was rebuilt from `__builtin_amdgcn_perm` and measured; see **§15b**.

**Cost, and that is the verdict.** Per site this replaces 5 LDS slots (1 store + 4 byte loads) with ~13 VALU
slots (4 and + 3 shift + 3 or + and + shift + `v_sub_u8`); the served body has 32 sites, so **−160 LDS slots
(11.6% of 1386) for +416 VALU slots (+30%)**, on a kernel that already reports 128 VGPR with 8 spills.
LDS is not the scarce pipe here (11.6% of issue slots) and VALU is the pipe this kernel is already dense in,
so the exchange is net-negative on issue capacity before any latency effect is counted, and it raises register
pressure on a kernel that is already spilling. **Design 15 therefore owes no GPU time: it is not a candidate
that needs qualification, it is priced out.** No GPU time was spent on it. The widening's correctness
proof (256/256) is CPU-only and is recorded above.

> **Correction (2026-10-23, measured): the pricing above understated what the shipped widening costs, and the
> verdict "priced out" was reached on an incomplete census.** The served body's widening band is **734 of 1384
> instruction slots (53%)**, not 160: `ds_load_u8` 128 + **`v_add_nc_u32_e32` 128 (LDS address math)** + width-2
> `v_bfe_u32` 112 (selector extraction) + `v_lshlrev_b32_e32` 63 + `v_lshl_or_b32` 32 + `v_or3_b32` 31 +
> `ds_store_b32` 32 + `s_waitcnt` 104. The −160/+416 comparison counted only the 160 DS slots against new VALU,
> and **omitted the ~366 selector/address ALU slots the shipped form already pays** (128+112+63+32+31) plus the 104
> in-band `s_waitcnt`. Band split of the 1384-slot body: widening **734**, epilogue store-address/predicate math
> **433** (its real math is only 16 `fma` + 17 `cvt`), prologue **213**. A register-only form removes DS **and**
> the address/selector ALU **and** the waits together, which is a different trade from the one priced above;
> §15b measures it.

### 15b. Register-only widening built from V_PERM_B32: exhaustive proof, and the served kernel's asm census.

Statics first, measurement second — this section now carries **both**, and the measurement is a served-route win.
GPU qualification: 4-leg ABBA (B,C,C,B in one `flock` hold), `tag perm2`, 21:43:46–21:47:04, log
`/home/vermont/tmp/t2-ab/abba-perm2.log`, full output in `abba-perm2.log`; 1024-token lighthouse prompt, `max_tokens
128`, `temperature 0`, `NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off`, card1 (`ROCR_VISIBLE_DEVICES=1`).

| leg | arm | prefill t/s | TTFT | decode t/s | generated text |
|---|---|---|---|---|---|
| base1 | shipped | 178.1 | 23.7 s | 13.0 | sha1 `77f1779eb831`, 574 chars |
| cand1 | permute | **203.4** | 20.7 s | 13.0 | identical |
| cand2 | permute | **202.5** | 20.8 s | 12.9 | identical |
| base2 | shipped | 175.2 | 24.1 s | 13.0 | identical |

Base legs spread 2.9 t/s (178.1, 175.2), candidate legs spread 0.9 t/s (203.4, 202.5): means **176.65 → 202.95
prefill t/s, +26.3 t/s = +14.9%**, with the candidate's *worst* leg 24.4 t/s above the base's *best* leg, so the
effect is far outside the off-leg spread and both candidate legs beat both base legs. Reference for the shipped
tree at b78f2c94 was 178.4 t/s cold; the base legs here (178.1, 175.2) reproduce it. **Coherence bar met**: all
four legs produced byte-identical text (sha1 `77f1779eb831`, 574 chars, 93 words / 59 distinct = 63%, most-frequent
word `'two'` ×8, max 8-gram repeat 1) — the same fingerprint as the prior band's verified output, which is the
expected result for a bit-exact change (proof-splay2: 0 mismatches over all 2^32 words). Decode rate is unchanged
(13.0 t/s), as it should be: the decode route does not use this kernel.

Confound recorded, not smoothed away: every leg after the first began with card1 reporting 76–81% `gpu_busy_percent`,
which is the lagging utilisation counter picking up the just-killed leg's server. That pattern struck **cand1 and
base2**, and base2 posted the *worst* base number of the set, so it did not systematically favour the candidate —
but the legs were not equally clean, so a confirm set (`tag perm2b`, `abba-perm2b.sh`) re-runs the same arms with a
30 s settle after each server exit and busy% sampled immediately before each request. Its outcome is appended below
this paragraph. A second, larger gap to watch: 202.95 t/s is still under the ~223 t/s attention-only ceiling
derived for this route, so the MLP buckets remain the binding constraint, and 300 t/s is not reached by this change.
**(Checked by arithmetic, and it holds, with one correction to how the ceiling must be read: it is a
*base-point* ceiling, not a fixed cap — see the reconciliation paragraph below the confirm set.)**

**Confirm set `tag perm2b` (`abba-perm2b.sh perm2b 128 30`, 21:48:33–21:54:12, one `flock` hold, 4 legs B,C,C,B):**

| leg | arm | prefill t/s | TTFT | decode | generated text |
|---|---|---|---|---|---|
| base1 | base | 176.9 | 23.8 s | 13.0 | sha1 `77f1779eb831`, 574 chars |
| cand1 | cand | **203.4** | 20.7 s | 12.9 | sha1 `77f1779eb831`, 574 chars |
| cand2 | cand | **203.2** | 20.7 s | 12.9 | sha1 `77f1779eb831`, 574 chars |
| base2 | base | 175.9 | 24.0 s | 12.9 | sha1 `77f1779eb831`, 574 chars |

Base mean **176.4** (legs 176.9, 175.9 — spread 1.0 t/s = 0.6%), candidate mean **203.3** (203.4, 203.2 — spread
0.2 t/s). **Δ = +26.9 t/s = +15.2%**, repeating `tag perm2`'s +14.9% to within 0.3 points, with a *tighter* spread
than the first set (base spread 2.9 → 1.0, candidate spread 0.9 → 0.2). The settle fixed the leg-cleanliness
problem: with 30 s after each server exit, card1 read **0% busy and 27,213,824 B VRAM before every server start**
(all four legs), and the busy% sampled immediately before each request was **51 / 48 / 53 / 51** across the four
legs — the lagging counter picks up each leg's own server load-in, and it hits the two arms equally.
Coherence bar met again on all four legs, byte-identical, same fingerprint as `tag perm2` and the prior band.
Decode 13.0 / 12.9 t/s both arms.

Pooled over both sets — 8 legs, 4 per arm, two independent flock holds: base **178.1, 175.2, 176.9, 175.9**
(mean **176.53**), candidate **203.4, 202.5, 203.4, 203.2** (mean **203.13**). **Δ = +26.6 t/s = +15.1%.** All four
candidate legs beat all four base legs; the candidate's worst leg (202.5) is 24.4 t/s above the base's best
(178.1), against a pooled base spread of 2.9 t/s. Every leg in both sets produced the same 574-character text
(sha1 `77f1779eb831`). The result is settled in sign and stable in magnitude across two sets.

Targeted test, run after the confirm set: `ctest --test-dir build-hip-test -R '^ninfer_linear_t2_a8_test$'` under one
`flock /tmp/ninfer-gpu.lock` hold, 21:57:13–21:57:26. **Passed, 12.88 s, exit 0, 1/1 tests passed**
(`gpu` label; log `/home/vermont/tmp/t2-ab/test-run.log`). The test is a real run, not a masked skip: its skip path
is `SKIP: no usable CUDA device` + **return 77** (`tests/ops/linear/test_t2_a8.cpp:48`), which CTest reports as a
skip via `SKIP_RETURN_CODE 77`, and this run reported "Passed" after 12.88 s of device work. Coverage is the
conformance sweep over widths 1…1024 — the small-T launches (one, two, four column tiles) *and* the prefill GEMM
that pads T to its tile width (`test_t2_a8.cpp:17`), i.e. exactly the kernel this change touches, at the served
chunk width 1,024 included. No other test was run: the change is confined to
`t2_prefill_wmma.cuh`, which the served a8 prefill route consumes, and the CPU suite is unaffected.

**Reconciliation of this win with the inherited bucket numbers — measured at the new operating point.**
GPU time: one leg, `profile-attrib.sh attrib1 cand 128 18991`, 21:54:54–21:55:42, inside one `flock` hold, card
idle at leg start. `NINFER_PREFILL_PROFILE=1` on the candidate arm, same prompt, same env as the ABBA sets. The
leg reported prefill 203.5 tok/s / TTFT 20.7 s with the usual text (sha1 `77f1779eb831`, 574 chars), so it is the
same operating point as the ABBA legs, not a perturbed one. Bucket sums are prefill calls only (the probe also
fires on decode calls; those are excluded — including them was what made the old 24.5 s vs 23.7 s overhead look
bigger; this leg's prefill call sum is 20.36 s against 20.7 s TTFT, +1.7%).

First, a correction to how the ~223 t/s ceiling must be read. It is `24.5 s − 5.676 s = 18.9 s`, i.e.
**everything except the `attn_mixer` bucket at the base operating point** — not "attention's time", and not a
fixed cap. Read at the candidate point, the same removal gives ~14.6 s ≈ **289 t/s**, and deleting *all* GEMM/MLP
time from the candidate point gives ~11.1 s ≈ **380 t/s**. The ceiling rises as the MLP buckets get cheaper; the
binding-constraint sentence above stands, the 223 number does not travel.

Bucket sums over the four 1,024-token chunks, base (from the table at §"the served T2 GEMM's activation-scale
fold") vs this leg:

| bucket | base (ms) | cand (ms) | Δ | Δ% |
|---|---|---|---|---|
| `gdn_mixer` | 4,986.5 | 4,298.6 | −687.9 | **−13.8%** |
| `gdn_mlp` | 8,262.1 | 6,693.2 | −1,568.9 | **−19.0%** |
| `attn_mixer` | 5,675.7 | 5,509.3 | −166.4 | −2.9% |
| `attn_mlp` | 2,645.2 | 2,139.3 | −505.9 | **−19.1%** |
| chunk total | 21,569.5 | 18,641.3 | −2,928.2 | −13.6% |

The Δ column is the win, and it lands where the kernel is used, not only in the MLP buckets: `gdn_mixer`'s GEMM
terms fell with the MLP buckets, and its non-GEMM terms did not move. Per sampled layer at T=1,024, base → cand:
`proj 13.43 → 10.69` (−20.4%), `out 4.91 → 3.91` (−20.4%), while `rec 4.74 → 4.78`, `nc 2.59 → 2.59`,
`norm 0.23 → 0.23` are flat. That is the mechanistic signature the change should produce — it edits the weight-code
widening inside the WMMA GEMM, so GEMM time moves and recurrent/elementwise time does not. The GDN bucket
(`proj+conv` was 52% of the GDN mixer, ≈14% of chunk time) is therefore *also* a consumer of this kernel, and the
share table below is computed on the cand leg.

Shares at the candidate point, all prefill calls (bucket total 19,320.7 ms): `gdn_mlp` 37.3%, `attn_mixer` 31.1%,
`gdn_mixer` 19.8%, `attn_mlp` 11.7%. GEMM/MLP = **49.0%** (was 50.6% at base — it shrank as a share because it got
faster, which is arithmetic, not a regression). Unbucketed layer time (embed/tail/elementwise) is 5.1% of the probe
sum. `attn_mixer` is now the second-largest bucket and keeps its quadratic growth (+556, +523, +539 ms per added
chunk on this leg), so it is the largest remaining single item and the one whose cost rises with context.

**What the instruction-count coincidence looks like after measurement:** the two `*_mlp` buckets fell **19.0%**
while this kernel's static instruction count fell **24.3%**. Count over-predicted the win by five points, so §15a's
lesson survives intact — instruction count is not a design metric here. (The earlier draft of this paragraph
computed −24.2% by attributing all of Δwall to the MLP buckets; the measured split puts 71% of the win in the MLP
buckets and 29% in the mixers, which is why that arithmetic was wrong.)

**What 300 t/s needs, priced from this leg.** 300 t/s on a 4,215-token prompt is 14.05 s of wall ≈ 13.82 s of
probe sum, i.e. **−6.54 s = −32.1%** off the current prefill. The whole GEMM/MLP budget is 9.47 s (49.0%):
removing **69% of every GEMM in the model** reaches 300 t/s with the mixers untouched. Equivalently, deleting the
entire `attn_mixer` bucket gives 289 t/s and the rest must come from `gdn_mlp`/`gdn_mixer`. There is no route to
300 t/s that leaves the GEMM time near where it is now, and none that is GEMM-only with the GEMMs left at
today's efficiency — this is the number to aim the next lever at, and it says the GDN and attention mixers are
now in scope, not exempt.

What the measured number is **not**: it is not the −24.3% instruction count speaking. Section 15a's own lesson was
that static instruction count alone has not translated to speed on this kernel; here 160 DS ops and 95 of 111 waits
leaving the body coincided with +14.9%. That is consistent with the widening being LDS-latency and dependency
stall-bound, which is what §15a's corrected accounting said to price — it is a hypothesis the data supports, not
something this measurement proves, and the 53% widening band did not become 53% faster (the MLP buckets' 50.6%
share of prefill time implies a whole-bucket win would be needed for 1.7×).

`__builtin_amdgcn_perm` is **V_PERM_B32**, and unlike `PRMT` its selector comes from a **register**, so the table
lookup the shipped path performs in LDS can be one VALU op per fragment register. Two forms were built and proved;
the first failed and is kept here because it is the interesting part.

**The mask form fails, exhaustively.** `perm(kTable, kTable, (w >> s) & 0x0F0F0F0F)` looks natural — V_PERM_B32
consumes one selector **byte** per output byte, and `& 0x0F0F0F0F` already isolates nibbles. It does not work:
nibble *j* of `(w >> s) & 0x33333333` is `code(2j) + 4·code(2j+1)`, so byte lanes carry **8..15**, and under the
ISA's strictest reading (index = `selector_byte & 0x07`, output byte forced to 0 when `selector_byte & 0xF8`) those
lanes select nothing and are zeroed. Measured over **all 2^32 code words × 4 fragment registers**
(`/home/vermont/tmp/t2-ab/proof-perm.c`): **4253024256 of 4294967296 words mismatch** the shipped widening, in all
four registers. Unfavourable result, preserved.

**The splay form works, and is proven immune to the ISA-reading ambiguity.** Each code pair is splayed into its own
byte lane's low two bits, so every selector byte is a code value 0..3 with bits 2..7 clear:

```
spread4(w) = (w & 0x00000003) | ((w << 4) & 0x00000300) | ((w << 8) & 0x00030000) | ((w << 12) & 0x03000000)
frag       = __builtin_amdgcn_perm(kTable, kTable, spread4(w >> s))   // s = 0, 2, 16, 18 for a, b, c, d
```

Selector bytes ≤ 3 make every reading of V_PERM_B32 agree at once — index mask `& 0x07` **and** `& 0x0F`, force-zero
mask `0xF8` **and** `0xF0`, and either operand as the low half — and the table is passed as **both** source operands,
so operand order cannot matter. Proven exhaustively (`proof-splay2.c`, log `proof-splay2.log`):
**`TOTAL mismatches = 0`, `words checked = 4294967296`**, across 4 fragment registers × 4 readings, plus the splay's
own layout claim (byte lane *j* == `code(2j + s/2)`, bits 2..7 clear). The proof is not vacuous: a third, independent
model of the shipped widening (`direct4`, plain sign-extend of each 2-bit field, no PRMT/nibble reasoning) is checked
against the shipped-side model in the same loop — **0 mismatches**, so both sides are anchored to the code semantics,
not to each other. Layout note that bit-proofed the shifts: nibble *j* sits at bit `4j + s` of the word, so a 16-bit
shift skips **8** codes, not 16 (an earlier harness used `2(2j+s)` and was wrong).

**Served-kernel asm census (compile only, no lock, no GPU).** Pair-recipe compile of the single TU, linked into two
arms (`link-arms-t2.sh`; gates: exactly one `t2_a8.cu.o` member per arm, archived bytes == the censused object,
identical 54-symbol `t2_prefill` symbol set, embedded `.hip_fatbin` payload differs — `88c06f0a7c` vs `54f806ed3f`,
384,624,368 → 384,579,312 bytes). Served kernel `t2_prefill_wmma1_kernel<1,2,Panel=32,SplitStoreEpilogue>`, body
terminated at its `.Lfunc_end` label, counted by one parser (`census-splay.py`, outputs `census-base.txt` /
`census-cand.txt`):

| served kernel body | shipped | permute form |
|---|---|---|
| instructions | 1365 | **1034** (−331, −24.3%) |
| `ds_load` (byte loads of the LDS-staged table) | 128 | **0** |
| `ds_store` | 32 | **0** |
| `s_waitcnt` | 111 | **16** |
| `v_add` total (incl. the LDS address math) | 209 | 81 |
| `v_bfe` (selector extraction) | 130 | 40 |
| `v_and` | 9 | 104 |
| `v_lshl` / `v_lshr` | 123 / 32 | 75 / 72 |
| `v_perm` (V_PERM_B32) | 0 | **32** |
| `v_wmma` / `global_load` / `global_store` / scratch ld+st | 16 / 24 / 16 / 2+2 | 16 / 24 / 16 / **2+2** |
| VGPR / sgpr / private segment | 128 / 43 / 36 B | **123** / 43 / 36 B |

All 160 DS ops in the body are gone, the LDS address math goes with them (adds −128), and 95 of the 111 waits
disappear. `v_perm` is **32, not the 64 a naive site count predicts** (16 `v_wmma` sites × 4 B-fragment registers):
LLVM common-subexpressions repeated permutes across the K-steps, which is a measured fact here, not a mechanism I
derived. LLVM did **not** fold the splay terms into `v_bfm` (it emits `v_and_b32_e32`/`v_lshrrev_b32_e32`/
`v_and_or_b32`), so the widening's mask+shift+permute ALU totals 323 slots (and 104, lshl 75, lshr 72, bfe 40, perm 32)
against the shipped form's 503 ALU + 160 DS = 663. The five sibling instantiations in the same TU move the same way
(1301 → 970, 1301 → 969, 1385 → 1051, 1385 → 1052, 1365 → 1035; all DS → 0, waits 111–143 → 16–48, VGPR 128 → 123).
Register pressure went **down** (128 → 123 VGPR, sgpr and private segment unchanged, both scratch accesses still 2+2 —
no new spills), which is the opposite of the §15a fear that a register-only widening must raise it.

A caution on my own earlier numbers: an ad-hoc diff script reported "2056 → 1325" body lines with 256/64 DS ops. That
was **a scope bug** — its start-label match swept past the kernel's own body (its DS counts were exactly 2× the
verified ones), so it was discarded rather than trusted, and the table above is the single-parser measurement. Census
scripts must terminate at `.Lfunc_end` and must not anchor on a bare label match: an earlier version of
`census-splay.py` also matched nothing at all, because GCN labels carry a trailing ` ; @mangled-name`.

**The lowering that this whole lever rests on is per-kernel, not a property of the idiom — measured.** The
shipped widening in the served kernel and the widening in the decode-route kernel are **the same source
idiom**: `__byte_perm(kTable, 0u, sel)` with `kTable = 0xFFFE0100u` and the `0x33333333` nibble mask
(`git show HEAD:src/ops/linear/t2/t2_prefill_wmma.cuh`, `wmma_code_frag` == `t2_prefill_i8.cuh`'s
`t2_prefill_decode`). Both live in the same translation unit (`t2_a8.cu`), and LLVM lowered them
**differently** (census of the base arm's own `.s`, `base/obj/t2_a8-hip-amdgcn-amd-amdhsa-gfx1102.s`):

| body | `ds_load` | `ds_store` | `v_perm` | `s_waitcnt` | instructions | VGPR | private seg |
|---|---|---|---|---|---|---|---|
| `t2_prefill_wmma1_kernel<1,2,32,SplitStoreEpilogue>` (prefill) | 128 | 32 | **0** | 111 | 1365 | 128 | 36 B |
| `t2_prefill_i8_kernel<5120/6144/17408,SplitStoreEpilogue>` (decode) | 19 | 9 | **16** | 160 | 7127 | 225 | 288 B |

So the claim that a data-derived-selector `__byte_perm` "cannot become PRMT **and therefore** lowers to an
LDS-staged table" — which is what the §15a diagnosis and my first version of this comment said — is only half
right: PRMT is indeed out (its selector is immediate), and the fallback LLVM picked *here* was the LDS table.
In the decode kernel, the same call became `v_perm_b32` with a register selector, 16 of them, no LDS table at
all. Consequences, both useful: (i) the fix in §15b is exactly "ask LLVM for the fallback the other kernel
already got", which is why it needs no new ISA trick; (ii) **the decode route needs no work** — its widening is
already register-selector, so the item "apply the same lever at `t2_prefill_i8.cuh:71-78`" is closed as
measured-not-needed, and decode staying at 13.0 t/s in the `tag perm2` legs is consistent with that (the
served decode rate did not move, as expected). Decode's own cost profile in that census is elsewhere — 7127
body instructions, 1172 `v_cmp`/`s_cmp`, 128 `scratch_load` and 288 B of private segment, i.e. spills, which
is a separate and larger thing than widening.

### Where the served prefill MLP route stands

Unchanged and unchanged by this section: `prefill_a8` sends the T2_G128_FP16 linears through
`t2_prefill_i8_quantize_kernel` + `t2_prefill_wmma1_kernel<1,2,Panel=32,SplitStoreEpilogue>`; mlp buckets are
50.6% of prefill time; attention-only ceiling ~223 t/s; the served reference point remains **178.4 t/s cold**
(b78f2c94). Two levers in this kernel are now closed by measurement (the `xsv` window, −25%) or by issue-slot
accounting (design 15, −160 LDS for +416 VALU — and see the §15a correction: that accounting omitted the shipped
form's own selector/address ALU, so it is not the last word). A third is **built, proven bit-exact, censused and
measured on the served route** (§15b: the served body loses all 160 DS ops, 128 address adds and 95 of its 111
waits, VGPR 128 → 123, no new spills; 4-leg ABBA `tag perm2` gave **176.65 → 202.95 prefill t/s, +14.9%**, with
byte-identical generated text on all four legs and decode unchanged at 13.0 t/s, and the pooled two-set result
**176.53 → 203.13 t/s, +15.1%**). It is now **committed** — `7042f064`, kernel file only, `ninfer_linear_t2_a8_test`
passed under the lock beforehand — so **the tree's served reference point is the candidate's 203 t/s**, and b78f2c94's
**178.4 t/s cold** is the base it beat, not the standing number. The 300 t/s target is not
reached — at the new operating point the buckets are 49.0% GEMM/MLP and 51.0% mixers (§15b attribution), the
attention-deletion ceiling has moved from 223 to ~289 t/s, and 300 t/s costs −32% of prefill time — but it is the
first change to this kernel that moved measured served prefill throughput **up**.
It is not the first to move it: the `xsv` window (§15) moved it **−25%** on the same route, and did so while
relieving register pressure far more than this change does (128→74 VGPR, 8→0 spills, 36 B→0 private, scratch
ops gone). Together the two results say register pressure does not predict the sign of a change to this kernel
— which is the same lesson §15a already drew for instruction counts, now learned twice on throughput.

## 2026-10-23 (later) — the attention edits build clean on the served route; **two more of my own lever hypotheses refuted statically**; the served bf16 occupancy roof becomes arithmetic

**No GPU time was spent in this section.** A full served compile, `grep` over already-compiled `.s`
files, and source reading. No kernel launched, no t/s claimed, nothing below is a speed claim.

### 16. Build qualification of the dirty attention files is met; route qualification is not, so they stay uncommitted

`tile_io.cuh` (+72) and `tiled_mma.cuh` (+254) — §3332's native-PV prototype and its value-2
schedule — are dirty in the tree. Full served build with them present:
`/home/vermont/tmp/pv-census/build-attn.log`, exit 0, ends `[100%] Built target ninfer-serve`. Two
diagnostics, both worth keeping:

```
tiled_mma.cuh:44:10: warning: loop not unrolled ...        (×4, one per template instantiation)
tiled_mma.cuh:408:13: warning: '#pragma unroll (1)' ... parenthesized  [-Wcuda-compat]
```

The second is the value-2 schedule's own pragma on the inner `for n` over `CTiles` (line 408); the
first is the compiler announcing it declined to unroll a loop in that region. Neither reaches the
served code path: the served binary is compiled with `NINFER_BF16_PV_NATIVE` unset, `kNativePV` is
false, and the served region is the PTX route unchanged. So handoff item 3's *build* precondition
for these files is satisfied. **Route qualification (a GPU ABBA) has not been run, and the lever is
closed on static evidence (§3417) — so the files stay uncommitted and no GPU time is owed.**

### 17. "The spills are only an unroll artifact" — refuted by the private-segment metadata

That `loop not unrolled` warning reads like a cause you can fix: an array indexed by a runtime loop
variable cannot be promoted to registers, so it lands in the private segment, which is precisely what
a 634-spill census looks like. If the loop spelling were the whole story, the fully unrolled arm
(`=1`) would come out register-clean.

Re-extracted straight from the metadata of the already-compiled census arms — `grep
.amdhsa_private_segment_fixed_size .amdhsa_next_free_vgpr .amdhsa_next_free_sgpr` over
`/home/vermont/tmp/pv-census/{off,on,ser}/launch-hip-amdgcn-amd-amdhsa-gfx1102.s`, all five
`tiled_mma` instantiations in each TU, CPU-only:

| arm | private segment, B/lane | `next_free_vgpr` | `next_free_sgpr` |
|---|---|---|---|
| `off` (served) | **608** (identical in all 5) | 256 | 55–57 |
| `on` (value 1, **fully unrolled**) | **2,472–2,520** (all 5) | 256 | 48–50 |
| `ser` (value 2, inner loop *not* unrolled) | **1,856** (all 5) | 256 | 49–51 |

The sign is the opposite of the hypothesis: the fully unrolled arm carries the **largest** private
footprint, ~4× the serial arm and 4× the served 608 B. §3417's census table already listed these
numbers; this is an independent extraction from the `.s` metadata section and it reproduces them,
which is the point of writing it down. The blowup is the 128 live fp32 accumulator floats per lane a
D=256 C fragment needs, not the schedule around it. **The native-PV lever stays closed at all three
values.** Recorded so no future session spends a build cycle on "just fix the unroll".

### 18. "The served attention loop stalls because it has no double buffering" — refuted by reading it

Checked `tiled_mma.cuh:182–315`. The served loop is already 1-deep software pipelined: `cp_wait<0>`
for K(kb) landed → issue V(kb) + commit → the QK MMA runs while V(kb) is in flight → `cp_wait<0>` for
V landed → prefetch K(kb+1) + commit → the PV MMA runs while K(kb+1) is in flight. Adding a second
buffer stage to a loop that already overlaps both loads with the opposite MMA is not a lever, and
the claim that it was one was mine, not inherited. **Refuted by source; no GPU time spent.**

### 19. The served bf16 LDS law, and the occupancy ceiling it implies

`schedule.cuh:69-70`: `bf16_kv_tiled_shared_bytes = (kQueryRows + 2·kKeyRows) · kHeadDim · 2`, with
`kWarps = kQueryTile/16`. At the served shape (`Bf16KvTiledMmaSchedule<64,32,1>`, D = 256):

| shape `(QueryTile, KeyTile)` | LDS = (Q + 2K)·256·2 | WG/CU (64 KB ceiling, `lds_fit.cuh`) | warps/CU | waves/SIMD (2 SIMDs/CU) |
|---|---|---|---|---|
| **(64, 32) served** | **65,536 B** | **1** | 4 | **2** |
| (64, 16) | 49,152 B | 1 | 4 | 2 |
| (32, 32) | 49,152 B | 1 | 2 | 1 |
| (32, 16) | 32,768 B | 2 | 2 × 2 = 4 | 2 |
| (16, 16) | 24,576 B | 2 | 1 × 2 = 2 | 1 |

Every shape the enum allows (`QueryTile ∈ {16,32,64,128}`, `KeyTile ∈ {16,32,64}`) tops out at
**2 waves/SIMD**; the served kernel occupies the *entire* CU LDS (65,536 B), so exactly 1 workgroup
per CU. The only shape that gets two workgroups per CU while keeping 4 warps — (32, 16) — reaches
the same 2 waves/SIMD, and pays for it by doubling K/V re-streaming per query row.
`Bf16KvTiledDescend` halves `KeyRows` while `Bc > 16` and only then halves `Br`, so both requested
instances (`instances.h:26-27`: `<64,64>` and `<128,32>`) land on the served `<64,32,1>`.

**Roof fact, arithmetic only:** for this kernel family at D = 256, raising occupancy by descending
the tile shape is not available. Any occupancy change must come from a *smaller live footprint*
(fewer accumulator registers, not a smaller LDS tile), which is exactly what §17 says the native route
cannot do. Nothing measured on the card here.

### 20. `vector_size` counts **bytes**: the prototype's wmma fragments are correctly spelled, and my "4× too wide" theory was wrong

I had read `Bf16WmmaAB = short vector_size(32)` / `Bf16WmmaC = float vector_size(32)` as 32 elements
each — 4× the width a w32 fragment needs — and taken that as the source of §3417's register blowup.
`__attribute__((vector_size(N)))` is **bytes**. The proven spellings in the shipped GDN recurrence
(`recurrence_wmma.cuh:195-220`) are `i16v16` = 16 shorts = 32 B for A/B and `f8v32` = 8 floats = 32 B
for C; t2 uses `int4v = int vector_size(16)` = 16 B, `int8v = int vector_size(32)` = 32 B. So
`float vector_size(32)` is exactly the 8-float C fragment, and the prototype's fragments match the
route already proven on this hardware. **The theory is withdrawn; the §3417 conclusion (structural
accumulator pressure) stands, unchanged and unrewarded.** No GPU time spent.

### 21. Bucket shares at the 203 t/s point: `proj` is the largest bucket never looked at on the kernel level — and its kernel mapping is **not finished**

From `/home/vermont/tmp/t2-ab/attrib1.log` (cand arm = the committed `7042f064`, `NINFER_PREFILL_PROFILE=1`,
a 1024-token prompt; the profiled leg is attribution, **not** a throughput claim — profiling perturbs
timing, the shares are the product):

| bucket | ms | share |
|---|---|---|
| gdn_mlp | 1662.6 (416.6/leg) | 37.3% |
| attn_mixer | 564.9 → 1120.0 → 1642.8 → 2181.6 | 31.1% (quadratic in prompt length) |
| gdn_mixer | 1072.3 | 28.0% |
| attn_mlp | 535.9 | 11.7% |
| total | 3835.9 | GEMM/MLP 49.0%, mixers 51.0% |

gdn-mixer fine split, per sampled layer (n=16), 1024 tokens: `nc=2.587 proj=10.691 rec=4.780
norm=0.232 out=3.911` ms, sum ≈22.20 ms. **`proj` is 48% of `gdn_mixer`, ≈9.5% of all prefill time —
the biggest single named part with no kernel-level census to my knowledge.**

Stated as an open lead, not a finding: the probe's five-part window for the GDN mixer has **not been
mapped to kernel names**. The instrumentation lives in `src/models/qwen3_5/execution/text.cpp`
(`fine_bounds` declared :289, cleared :338, sample start :368-369, sample cap guard :374-376, bucket
loop :401-416, fine sum :420-424, prints :437-444, step-probe twin :258-259), and the call site that
pushes the individual part boundaries is not yet located. `proj` sits between `nc` and `rec`, so the
strong prior is that it is the mixer's input projection GEMM plus the short conv and gate — i.e.
mostly the GEMM lane's i8 route, where §1419-1448 prices the bit-exact headroom at ≤7% of the 300 t/s
budget — but **prior, not verified.** Next step is to find that call site and pull those kernels'
`.amdhsa_*` metadata, CPU-only.

### 22. Compiled metadata of the committed served T2 TU (`7042f064`), for reference by the next session

From the offload `.s` of the served T2 kernel objects (`cand/obj/...-gfx1102.s`):

| kernel | VGPR | vspill | SGPR | private | group |
|---|---|---|---|---|---|
| `t2_prefill_i8_quantize_kernel<5120/6144/17408>` | 25 | 0 | — | 0 | 0 |
| `rowsplit_a8::quantize_activations` | 42 | 0 | — | 0 | 0 |
| `t2_prefill_wmma2_kernel<2,1,{8,16,32},SplitStoreEpilogue>` | 78 | 0 | 46 | 0 | — |
| **served `t2_prefill_wmma1_kernel<1,2,32|0,SplitStoreEpilogue>`** | **123** | **8** | 45 | **36 B** | — |
| decode `t2_prefill_i8_kernel<5120,SplitStoreEpilogue>` | **225** | 0 | 34 | **288 B** | 32,000 (2 CTA/CU) |

The served GEMM is the only prefill kernel in this TU that still spills, and decode is the only one
with a real private segment (288 B/lane, 128 `scratch_load`, 7127 body instructions, 1172 compares —
§3855). Decode's spill profile is a bigger item than widening; it is named here so it is not lost,
with no claim attached.

### 23. §21's `proj` lead is closed; the served GEMM's wmma stream has **no slack at all**; and 300 t/s turns out to have two price tags, one of them not a GEMM problem

CPU-only work. **No GPU time spent, no lock taken.** Everything numeric below is either (a) read
out of a dump/log that already exists, or (b) arithmetic on such numbers, labelled as arithmetic.
Nothing here is a measurement of a new quantity, and no throughput is claimed for any unmeasured
change.

**a. `proj` is now mapped to kernel names (closes §21's open lead).** The probe's own boundary
comment states the window contents — `src/models/qwen3_5/execution/text.cpp:357-360`: boundaries per
sample are *start, after norm_control, after projection+conv, after recurrence, after gated norm,
after output projection*; sampling lands on layers 0 mod 4 (the GDN layers, `:364-366`) and the marks
are pushed at `:1373-1374` and `:1434`. The call sites inside the prefill window are `gdn_projection`
at `:1426` and `ops::causal_conv1d_silu_split` at `:1431` (wrapper
`src/ops/wrapper/causal_conv1d_silu.cpp:277`). The GEMM route behind `gdn_projection` for the served
T2 model is `t2_two_parent_project` (`src/ops/wrapper/gdn_input_proj.cpp:783`, called `:885`/`:1254`),
which quantises `x` once (`t2_prefill_i8_quantize_kernel<5120>`) and then issues **two sequential**
`project()` calls — `kQkRows = 4096` and `kValueRows + kZRows = 12288` rows, `k = 5120` — each landing
on `t2_prefill_wmma1_kernel<1,2,32,SplitStoreEpilogue>` (metadata in §22: 123 VGPR, 8 `vspill`,
36 B private, one warp per CTA). At `columns = 1024` the panel grids are (32, 32, 8) and (32, 32, 24),
i.e. 8192 + 24576 = 32768 CTAs per layer.

**b. Geometry verified** from the conversion json and the `require_*` constants: hidden 5120;
**64 layers = 48 GDN + 16 full-attention** (3 linear : 1 full); heads 24, kv-heads 4, head_dim 256;
`intermediate_size` 17408; `kQkRows 4096`, `kValueRows = kZRows = 6144`. Arithmetic on that: per-token
linear MACs = **2.31e10**, which confirms the ≈2.2e10 the roof section quotes (so that figure is
*true* MACs, not the doubled convention — see d). Bucket rates from `attrib1.log` leg 1, computed as
`MACs × 1024 tokens ÷ bucket_ms`: proj **8.03**, gdn out-proj **8.24**, gdn_mlp **7.90**, attn_mlp
**8.17** TMAC/s — all four agree with §1419's table, which validates both my geometry arithmetic and
that table's convention.

**c. The hot loop's census: the wmma stream is already at the theoretical minimum.** Parsing the
offload `.s` of the served kernel (script `loopcount.py`, dumps under `/home/vermont/tmp/t2-ab/`), the
`kGroupK = 128` loop (`kGroups = k_cols/128 = 40`) is:

| | cand (shipped `7042f064`) | base (`b78f2c94`) |
|---|---|---|
| instructions per k-group | **522** | **682** |
| `v_wmma` per k-group | **16** | **16** |
| `global_load` per k-group | 24 (19 `b128`) | 24 |

16 wmma × 40 groups = **640 `v_wmma` per CTA**, which is exactly the minimum for a 16×32 output tile
at `K = 5120` (one 16×16×16 instruction per 16 K-slots, 2 accumulators). **So §3153's "issued ≈ 2×
useful, therefore at the issue wall" does not hold for the instruction stream: there is no wmma slack
to recover.** The 2× that the source comment describes is in activation *load bytes* — the 16 per-lane
`b128` fragment loads deliver 16 B/lane where 8 B/lane are read by the instruction ("the odd and 16
activation slots duplicate rows the instruction never reads"), i.e. a DRAM/L2 traffic lever, not an
instruction-count lever.

Mix of the 522: `v_and` 104, `s_delay` 76, `v_lshrrev` 55, `v_lshlrev` 48, `v_perm` 32, `v_or3` 32,
`v_bfe` 24, `global_load` 24, `v_cvt` 17, `v_wmma` 16, `v_fma` 15, `v_cndmask` 15, `v_mul` 14,
`s_waitcnt` 14, dual-source 12, `v_add` 12 — ≈295 instructions per group are fragment-building and
scale-folding for 16 wmma. Arithmetic on that mix: at 32 CU × 4 SIMD × 2.557 GHz = 327.3 G
warp-instructions/s, serving 8.04 TMAC/s at 522/16 = 32.6 instructions per wmma consumes
**64.0 G instr/s = 19.6% of issue**. The kernel is **not** at the issue wall; the calibrated model
`MAC/s = 4.173e15 / I` (fitted to I=522 → 7.99 vs 8.04 measured) puts this mix's ceiling at
≈11.0 TMAC/s, of which the served kernel attains ≈73%.

**d. Units: the peak-probe numbers are in two conventions and the docs mix them.** Verified against
each probe's own shapes and times: cal66's 12.7 / 12.5 / 12.8 "TMAC/s" are **2×** the true MAC rate
(true: gate_up 6.35, down 6.25, o_proj 6.41) because it counts 8192 per 16×16×16 instruction; cal67 /
cal70 / cal71 count 4096 (cal71 source line 235 `macs = (16*MB)*(400*8*C)*4096.0` reproduces its
16.55e12 headline exactly). Consequence: "the engine runs at 98% of cal67's 12.95 peak" compares a
true-MAC numerator against a 4096-convention denominator; the true ratio is ≈**49%**. Hardware
derivation for reference: 128 int8 MAC/cycle/SIMD → 41.9 TMAC/s at 2.557 GHz; cal42's barrier-free
peak clock-scaled is 21.5 TMAC/s. The served 8.0–8.4 is ≈39% of cal42's measured peak.

**e. The device is 32 CUs, and two probe papers sized their grids for 16.** `rocminfo`: gfx1102 RX
7600M XT = **32 Compute Units** (the gfx1035 iGPU is the 16-CU part). cal70 hardcodes `G = 112 // 16
CU × 7` and cal71 uses `G = 16 × MB_target`, so **their thr/CU and blocks/CU labels are 2× inflated**
(cal71's headline 16 blocks/CU is really 8). The shipped tile's real occupancy — 15 blocks/CU at
123 VGPR, launch-bound 8 — was never probed at 32 CU. Any occupancy conclusion from cal70/cal71 needs
a re-run with `G = 32 × MB` before it can be reused.

**f. Unfavourable, and it changes the plan: at long prompts the 300 t/s gap is mostly attention, not
GEMM.** The four 1024-token legs of `attrib1.log` (the 4215-token request prefills in 1024-token
chunks against a growing cache) are:

| leg | total ms | linear (gdn_mixer+gdn_mlp+attn_mlp) | attn_mixer ms | attn share | t/s for that leg |
|---|---|---|---|---|---|
| 1 (0–1024 keys) | 3835.9 | 3270.8 | 564.9 | 14.7% | 267.0 |
| 2 | 4396.0 | 3275.7 | 1120.0 | 25.5% | 232.9 |
| 3 | 4930.6 | 3287.6 | 1642.8 | 33.3% | 207.7 |
| 4 | 5478.8 | 3297.0 | 2181.6 | 39.8% | 186.9 |

All inter-leg growth is attention: leg-to-leg total grows +560/+535/+548 ms of which attention is
+555/+523/+539 and linear is +4.9/+11.9/+9.4 ms. Across the four legs: **total 18641.3 ms, linear
13132.0 ms (70.4%), attention 5509.3 ms (29.6%)**, average linear rate 7.21 TMAC/s. Arithmetic:
a 300 t/s budget for those 4096 tokens is 13653 ms = **−26.8% total time**. If attention is left
untouched, linear must drop 38.0% (7.21 → 11.62 TMAC/s), the calibrated model wants
I = 522 → 361 (−31%), and at the measured conversion rate (below) a 31% instruction cut buys ≈20%
throughput → ≈**256 t/s**, not 300. Conversely: the attention layers' own qkv/o GEMMs are
≈1.03e12 MACs per leg (16 layers × 6.29e7/token), i.e. 129 ms/leg at 8 TMAC/s, so the *pure*
attention term (QK + softmax + PV) is ≈4994 ms of the 18641; **deleting it entirely lands at
296–302 t/s** (band taken over 6–10 TMAC/s for those GEMMs). So 300 t/s on this benchmark is
reachable from the attention mixer alone, or from a mix — and **not** from GEMM instruction cutting
alone. I record that as the unfavourable result for the GEMM-centric plan.

**g. Conversion rate of instruction removal, measured, and a withdrawal of mine.** Base→cand
(§16-§22, pooled two-set ABBA): instructions per k-group 682 → 522 (**−23.5%**) for **+15.1%**
prefill throughput ⇒ **0.64×**, sub-proportional, consistent with a latency-bound kernel. Two
consequences: instruction-count reduction should not be extrapolated linearly, and **my working note
this session that "300 t/s ⇒ I ≤ ~232" is withdrawn** — it priced 300 t/s against leg 1 (267 t/s for
a 1024-token empty-context chunk), not against the served 203.5 t/s long-prompt average. Both
targets are named explicitly here so the next session cannot confuse them:
leg-1 shape 267 → 300 t/s needs −11.0% time, I ≈ 425 (−18.6%), plausible by instruction cutting;
served long-prompt 203.5 → 300 t/s needs −26.8% time, which per f. is out of reach for instruction
cutting alone.

**h. What the census leaves as levers (all A/B-gated, none measured).** (i) the ≈295 non-wmma
instructions per group — `v_and` 104, shifts 103, `v_perm` 32, `v_or3` 32, `v_bfe` 24 — which at
0.64× conversion are worth at most ≈20% prefill throughput; (ii) the 2× activation load amplification
as a *traffic* lever (19 `b128` per group, 16 of them oversized by 2×); (iii) `s_delay` 76 +
`s_waitcnt` 14 per group = 17% of the loop is scheduler padding, a packing/latency lever rather than
a work lever; (iv) occupancy at real 32-CU sizing (e); (v) per f., the attention mixer, which is now
the largest single lever at long prompts and whose served route (§1151, §3332, §3417: BF16 KV, LDS
65,536 B ⇒ 1 WG/CU, native-PV closed at all three `NINFER_BF16_PV_NATIVE_ON` values) is exactly where
previous sessions ran out of room. (i)–(iv) must be A/B'd under `flock /tmp/ninfer-gpu.lock`; the
sign of a change is not predictable from any of the counts above (twice demonstrated in §16-§22).

## §24 — Served attention mixer, static census at the served operating point; two of my own claims retracted

Compile-only work. **No GPU was touched, no lock taken, nothing measured, no t/s claimed.** Dumps used:
`/home/vermont/tmp/pv-census/{off,on,ser}/launch-hip-amdgcn-amd-amdhsa-gfx1102.s`, produced by
`census2.sh` from `src/ops/softmax_attention/dense/causal_cache/bf16/launch.cu` with the tree's exact
flags, three times: macro unset / `NINFER_BF16_PV_NATIVE=1` / `=2`.

**a. Naming trap, and my error.** The dump directory named `ser` is the **macro value 2 arm**, not "the
served build". The served build compiles with the macro **unset** — the `off` arm — because
`tiled_mma.cuh` defaults `kNativePV` off (§3357). I spent the first part of this session censusing
`ser` and reported the served attention kernel as 5,455 instructions / 68 `v_wmma` / 0 LDS ops.
**Retracted.** The served numbers are below, and they agree with what §3169 and §3272 already recorded
(64 WMMA, 812 `ds_bpermute`), so the docs were right and my working note was wrong. A directory called
`ser` is a hazard: the three arms should be named by macro value in every future census.

**b. Two defects in my census scripts (both fixed in `/home/vermont/tmp/isacensus/`).** (i)
`loopcount.py`'s label regex required `^(\.L[\w.]+):\s*$`, so label lines carrying a trailing
`; %Flow…` comment never registered; it reported "total backward branches: 0" for a kernel with 671.
(ii) `inspect.py`, `loopcount.py` and `opcode.py` filtered instructions with `^[vms]_|^global_|^buffer_`,
which **silently drops every `ds_*`, `flat*` and `accperm` mnemonic**. On RDNA3 the LDS ops print as
`ds_load_bNN` / `ds_store_bNN` / `ds_bpermute_b32`, so "ds = 0" was an artefact of the filter, not a
property of the kernel — and every body total those scripts printed was short by the LDS ops. The
attention totals below are the ds-aware ones. (iii) The **same scripts were blind to spills**: the
accepted prefixes did not include `scratch_` / `flatscratch`, so every `scratch_load_*` /
`scratch_store_*` was invisible. That defect is what produced this section's "0 scratch ops in the
body" claim (since retracted below, §25a) and it shorted every body total by the spill count; with
`scratch_`/`flatscratch` accepted, the served body total lands on exactly the control value
`census2.sh`'s header predicts. Both filters are fixed in `/home/vermont/tmp/isacensus/`. The GEMM figures in §16-§23 (522 / 682 per k-group,
`census-base.txt` / `census-cand.txt`) came from `t2-ab/census-splay.py`, whose patterns count `ds_load`
/ `ds_store` and whose label regex already tolerated trailing comments — **re-checked, those figures are
unaffected**.

**c. Served route, re-verified statically (no GPU).** HIP defines `NINFER_SM8X_COMPAT` (CMakeLists.txt:49-51);
the served profile is BF16 KV, so `bf16_kv_append_attention` / `bf16_kv_cached_attention`
(`causal_softmax_attention.cpp:692`, `:762`) → `launch_bf16_kv_tiled_mma`
(`bf16/template_launch.cuh:97`) → `Bf16KvTiledMmaSchedule<64,32,1>`, the LDS-fit choice
(`bf16/lds_fit.cuh:21-42`). Kernel: `bf16_kv_tiled_mma_kernel<CausalGeometry<256,24,4>,
Bf16KvTiledMmaSchedule<64,32,1>, PagedKVDirectMetadata>`. **That schedule is the only tiled variant in
the fatbin**: all three dumps carry 62 kernels, of which 6 tiled_mma instantiations, all `<64,32,1>`
(the 3 metadata forms × 2 geometries). `Tiled64 <64,64>` and `Tiled128 <128,32>` in `instances.h:17-41`
are **not instantiated here** — so "try a bigger tile shape" is not a knob at this operating point; it
needs new instantiations, and the `<64,32,1>` tile is already pinned at LDS 65,536 B ⇒ 1 WG/CU (§1151).

**d. Served body census (macro unset, ds- and scratch-aware).** **14,202** instructions in the served
symbol's body — the exact control value `census2.sh`'s header says the served arm must reproduce, so
this census and the recorded one now agree line for line.
`v_wmma_f32_16x16x16_bf16` **64**; `ds_bpermute_b32` **812** (12.7× the WMMA count); `ds_load` 192 +
`ds_store` 84 = 276 LDS data ops; **`scratch` 336** (116 `scratch_store_b128`, 88 `scratch_load_b128`,
64 `scratch_load_b64`, 49 `scratch_load_b32`, 19 `scratch_store_b32` = §3173's 201 loads / 135 stores,
confirmed); `global_load` 84, `global_store` 64; fp32 fma family 620 (518 `v_fmac_f32_e32` + 102
`v_fma_f32`) plus 366 `v_dual_fmac_f32`; `v_cmp` 1,581
(1,208 of them `v_cmp_ne_u32_e32`); `s_delay_alu` 1,630 (11.5%); `s_waitcnt` 575-582; `v_mov_b32_e32`
1,338; `v_cndmask` 491; `v_lshl` 777; 131 distinct opcodes.
Metadata: `next_free_vgpr 256`, `next_free_sgpr 55`, `private_segment_fixed_size 608 B/lane`,
`group_segment_fixed_size 0` (the 65,536 B tile is dynamic), `wavefront_size32 1`, `uses_dynamic_stack 0`.

**e. The loops (fixed `loopcount.py`, 671 backward branches).**

| loop | instr/iter | wmma | ds_bpermute | ds ld/st | cmp | fma | waitcnt |
|---|---|---|---|---|---|---|---|
| `[10239..19690]` via `s_cbranch_scc0` (main, contains the next row) | **5,747** | **32** | **404** | 128 | 726 | 276 | 248 |
| `[3173..10190]` via `s_cbranch_vccz` | 3,852 | 0 | 320 | 0 | 575 | 317 | 120 |
| `[12693..19648]` via `s_cbranch_vccz` (nested in the main loop) | 3,790 | 0 | 320 | 0 | 575 | 257 | 109 |
| epilogue: nested `s_cbranch_execz` cascade, innermost `[19930..20610]` | 617 | 0 | 0 | 0 | — | — | — |
| … growing **+24 instructions per nesting level** … | | | | | | | |
| epilogue outermost `[20110..21870]` via `execnz` | **1,697** | 0 | 0 | 0 | 126 | 0 (add=252) | 63 scratch, 63 `global_store` |

**179.6 instructions per WMMA in the main loop** (5,747 / 32) against the optimised GEMM's 32.6
(522 / 16). Of the main loop, 404 `ds_bpermute` + 128 LDS data + 248 `s_waitcnt` = 780 of 5,747
(13.6%) is LDS-pipe and its bookkeeping; the two 320-permute reduction loops carry **zero WMMA** and
574 `v_fma`-family + 1,150 `v_cmp` between them. The epilogue cascade is `v_bfe`/`v_or`/`v_add`/shifts/
`v_cndmask` — bf16 packing, 617 instructions at the innermost level growing by 24 per level to 1,697 at
the outermost.

**f. Disposition of the uncommitted attention candidate (handoff item 3).** `git diff --stat`:
`tiled_mma.cuh` +254, `tile_io.cuh` +72, both **behind `NINFER_BF16_PV_NATIVE`, default OFF**. The
served arm's census (d) reproduces the recorded served control (64 WMMA, 812 `ds_bpermute`, 608 B
private), i.e. **the candidate changes nothing in the served binary**. Its active forms are macro=1
(**7,281** instr, 128 WMMA, `ds_bpermute` 812→36, private **2,472 B/lane**; recorded FAIL at 1,060-1,080
spills) and macro=2 (**7,202** instr, 68 WMMA, 36 permutes, private **1,856 B/lane**; measured closed,
§3389-§3435). Precision note, corrected: inside the served symbol's own body the scratch count is
**336 / 1,090 / 1,395** for macro unset / 1 / 2 — the earlier "0 in all three arms" was the
scratch-blind filter of §24b(iii) — so the gate arms spend **3.2× and 4.2×** the served route's spill
traffic, which is the cost side of the 812→36 permute saving; whole-TU scratch ops run
30,714 / 35,312 / 37,065. Consequence under item 3's gate: build and
route qualification are done and the outcome is "no served-route change", so a served ABBA would
compare identical machine code, and **there is nothing to commit as a perf change and no GPU time is
warranted for it**. The two files stay uncommitted.

**g. What (d)-(e) leave as attention levers, none measured.** The mixer's instruction bulk is not
tensor-core work: 13.6% LDS-pipe + bookkeeping, 11.4% compares, 11.8% `s_delay_alu`, and 179.6
instructions per WMMA. §25 explains why, and it is a stronger statement than this paragraph made. The `ds_bpermute` count is the signature: on RDNA3 every wave32 `__shfl` is an
LDS-pipe instruction (§3217), and the served kernel has 812 of them for 64 WMMA. The gate arms show a
route that removes 95.6% of it (812→36) at the cost of 2× WMMA and +1,864/+3,064 B/lane private, which
is what the closed arms bought. Levers that do **not** exist at this point: bigger tile shapes (c), and
the PV-native gate as shipped (f).

## §25 — Why the served mixer spends 179.6 instructions per WMMA: its PV contractions are software-emulated

Compile-only again, and it answers the question §24 left open (the served PV path issues 64
`mma_bf16` calls **per warp per block step** — `PVNt × PVKs = 32 × 2`, so 256 per block step across its
4 warps, and 128 in the body because the peeled first block and the main loop each hold a copy — yet
the body holds only 64 `v_wmma`, which are QK-only). **No GPU was touched, no lock taken, no t/s
claimed.** Same three dumps as §24; new scripts
`/home/vermont/tmp/isacensus/{asmreg,rangehist,peelscaffold}.py`, `/home/vermont/tmp/isacensus/lc-{off,on,ser}.txt`,
`/home/vermont/tmp/asmprobe/`. Geometry check: `tiled_mma.cuh:52-66` gives `D=256`,
`Br=kQueryRows=64`, **`Bc=kKeyRows=32`** (source comments saying "64 key cols" are stale), `Threads=128`,
`kWarps=4` ⇒ `QKNt=4`, `QKKs=16`, `PVNt=32`, `PVKs=2`, `CTiles=16`, `NT16=2`; native PV is
`CTiles × NT16 = 32` wmma per warp per step.

**a. `mma_bf16` has two definitions, split by `#if defined(__HIPCC__)` (line 12) / `#else` (line 453)
in `src/ops/common/mma.cuh`.** The CUDA form (`:482`) is
`asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 …")`. The **HIP form (`:268`) is a
SIMT emulation**: four `gather8` (each eight `bcast_u32` lane broadcasts, `:42`) plus four `dot_bf8`
(`:126`), and `dot_bf8` is eight iterations of `unpack_bf2w` on both sides and two products and two
fp32 adds — 16 unpacks, 16 products, 16 adds per `dot_bf8`. So on gfx1102, per `mma_bf16` call:
**32 lane broadcasts (LDS-pipe on RDNA3, §3217) + 16 unpacks × 2 sides ≈ 32 word unpacks (each a
`load_vec` + `__bfloat1622float2`, ≈ 2 ISA ops ≈ 64 ops) + 4 × `dot_bf8` × (8 mul + 8 fma) ≈ 64
fma-family ops** — i.e. **~128 VALU + 32 shuffles for 2,048 MACs**. (§25a's original "~64 VALU ops"
was an undercount: it counted the products only.) `pack_bf16x2`'s HIP arm (`math.cuh:55-57`) is a
`load_vec<uint32_t>`, not the `cvt.rn.bf16x2.f32` asm — that asm is in the `#else` CUDA arm at `:62`.
That is what the served route's PV runs on. Proof in the dump: the head of loop `[12693..19648]` is
`.LBB0_850: ; %_ZN6ninfer3ops8mma_bf16ERfS1_S1_S1_jjjjjj.exit542.i.3 ; in Loop: Header=BB0_851
Depth=2`, and its instructions are the `v_and_b32 0xffff0000` / `v_lshlrev_b32 16` /
`v_{fma,fmac,dual_fmac}_f32` idiom of `dot_bf8`, not any `v_wmma`.

**b. All 64 `v_wmma` in the served body belong to QK, not PV, and they are the best-shaped code in
the kernel.** They occur in exactly two runs of 32 — body lines 1757-1907 (the peeled first block) and
11202-11352 (inside the main loop `[10239..19690]`) — each run **151 instructions**: 88 `ds_load_b128`,
32 `v_wmma_f32_16x16x16_bf16`, 28 `s_waitcnt`, 1 `s_delay_alu`, 1 `s_barrier`, 1 `buffer_gl0_inv`,
with **byte-identical inter-WMMA spacing** between the two runs (4 8 2 6 4 6 6 1 6 6 4 6 4 6 4 10 2 6 4
…). That is **4.7 instructions per WMMA, entirely LDS-fed** — 7× tighter than the optimised GEMM's 32.6
per WMMA (§16-§22). The tensor-core part of the main loop is 151 of its 5,747 instructions = **2.6%**.

**c. Therefore 179.6 instructions per WMMA is not the MMA issue being expensive** — it is 97.4%
non-tensor-core work around 151 instructions of tensor-core work. The dominant block is the nested
`vcc` loop `[12693..19648]`, 3,790 instructions = **66% of the main loop**, with only 35 distinct
opcodes: 575 `v_cmp_ne_u32_e32` (15.2%), 570 `v_mov_b32_e32` (15.0%), 528 `s_delay_alu` (14.0%), 320
`ds_bpermute_b32` (8.4%), **287 `s_cbranch_vccz` + 287 `s_cbranch_vccnz`**, 225 `v_fmac_f32_e32` + 207
`v_dual_fmac_f32` + 32 `v_fma_f32`, 206 `v_lshlrev_b32_e32`, 170 `v_and_b32_e32`, 109 `s_waitcnt`,
70/38/32 `v_add_co_u32`/`v_add_co_ci_u32_e32`/`v_add_co_ci_u32_e64`, 32 `flat_load_u16`. Its peeled
twin `[3173..10190]` is the same shape (3,852 instr, 575 cmp, 320 permute, the same 287+287 branch
pairs, 32 `flat_load_u16`).
**Correction (measured, `peelscaffold.py` + `/home/vermont/tmp/asmprobe/probe.s`):** I first read
those 575 compares / 574 branch pairs as "per-element control flow for the causal mask and the softmax
reductions". **That reading is refuted.** Every one of the 575 compares is
`v_cmp_ne_u32_e32 vcc_lo, 0, <reg>`, and for every one of them the nearest preceding write to that
register in program order is `v_mov_b32 <reg>, 0` or `1`; 574 of them are followed by an
`s_cbranch_vcc{z,nz}` within three instructions (skipping `s_delay_alu`/`s_nop`/ASM markers). The
comparands are **loop-peel / divergence flags written by the compiler, not mask or softmax data** — no
masked-element comparison, no reduction comparison appears in that loop's compares. See (h).

**d. The structural reading (reading, not a measurement).** Two costs compound in the served mixer:
the emulated PV (`mma.cuh:268`: 32 lane broadcasts + ~128 VALU ops per contraction call, 64 calls per
warp per block step) and the compiler's peel/divergence scaffolding around those call sites (574 branch
pairs + 570 movs per iteration — scaffolding, not mask logic, per the (c) correction). The tensor-core
equivalent of the same PV contractions is 32 WMMA plus LDS staging —
exactly the 151-instruction shape that part (b) shows is achievable on this kernel, at 4.7
instructions per WMMA. The `NINFER_BF16_PV_NATIVE=1` dump is that conversion already performed (128
WMMA where the served arm has 64, `ds_bpermute` 812→36, body 14,202→7,281), and its recorded failure
is on the register side, not the tensor-core side: 1,090 in-body scratch ops and 2,472 B/lane private
(§24f, corrected). What the arms did **not** change is the mask/softmax control-flow shape — both
gate arms and the served arm keep 36 permutes, and macro 2 is the contraction-group-outer form.
**No throughput claim is made for any of this.** Instruction counts in this class have twice failed to
predict the sign of a throughput change here (§16-§22), and the serving point is 203 t/s (§7042f064)
until an ABBA says otherwise.

**e. `v_perm_b32` appears 0 times in the served attention body.** The shipped `V_PERM_B32`
code-fragment widening (7042f064, pooled ABBA 176.53→203.13) lives in the GEMM code-fragment path, not
in this kernel. So there is no `v_perm` lever standing in attention today, and the +15.1% is a GEMM
result.

**f. Attribution/tool notes.** (i) §3169's "`ds_bpermute` … 498 inside `;;#ASMSTART`/`;;#ASMEND`
regions" is **not reproducible in the current dump**: the served body has 1,208 inline-asm regions with
**0** class instructions inside them, and the whole file has 73,176 `ds_bpermute` with **0** inside ASM
regions (`asmreg.py`). Served shuffles are compiler-lowered `__shfl`/`bcast_u32` builtins, so the lever
is source-level layout, not asm editing; §3169's count presumably belongs to a pre-b78f2c94 dump.
(ii) Source sites of the served route's shuffles: `tiled_mma.cuh:253-256` (four
`__shfl(dots[e], gid + 8k)` over 8 `e` = **32 per 16-key tile**, 64 per block step — the QK
C-fragment→score-register redistribution; this is the "2×(exactly 64)" run pair of §3169), `:344` and
`:370` (native max/sum butterflies, one each), `:386-387` (native A-fragment gathers, 16 per tile),
and `mma.cuh:42` `gather8` (32 per emulated contraction call). (iii) Source-line attribution via line
tables does not work on this toolchain: rebuilding the served TU with `-gline-tables-only`
(`/home/vermont/tmp/isacensus/locbuild.sh`) produced a 313 MB dump containing **0** `.loc` directives.
`;;#ASMSTART` markers and LLVM's `; %Flow…` / `; in Loop: Header=… Depth=…` label comments are what
carry attribution.

**g. The lever (c), (d) and §24f point at, stated for the next static pass.** Register demand, not
permutes, is what closes the wmma-PV route: the native path keeps `acc16[CTiles][8]` live across PV
(`CTiles = 16` ⇒ 128 fp32 accumulators per lane at 256 VGPR, with the 608 B/lane served baseline
already at the spill boundary). Macro 2 attacked it by moving the contraction group outer (one big
loop of 2,799 instructions, 649 scratch ops, 1,856 B/lane) and lost. What has **not** been tried
statically: shrinking the live accumulator set itself — splitting the PV output-tile loop so at most
half of `acc16` is live at a time, or draining each output tile's accumulator to its destination
before the next tile's contractions begin — combined with the wmma-PV form. Any A/B for this needs the
card, the lock, and a coherence check; nothing here is committed.

**h. What the 575 compares actually are: compiler scaffolding at the emulation call sites (measured,
compile-only).** Census of the served body (`/home/vermont/tmp/isacensus/served_body.s`, 1-indexed
lines; region boundaries as in (c)):

| region | instructions | `;;#ASMSTART`/`END` pairs (all **empty**) | `v_cmp_ne_u32_e32` | full `mov(0\|1)→cmp 0→vcc` triples | `s_cbranch_vcc*` | `.peel*` labels | `Child Loop … Depth 3` comments |
|---|---|---|---|---|---|---|---|
| nested `[12694..19649]` | 3,790 | **575** | **575** (all `cmp 0,reg`) | **574** | 574 | 287 | 288 |
| peeled twin `[3174..10191]` | 3,852 | 575 | 575 | 574 | 574 | 287 | 288 |
| main loop's own `[10240..12694]` | 1,927 | 24 | 24 (all `cmp 0,reg`) | 24 | 28 | 12 | 301 |
| whole body | 14,202 | 1,208 (0 non-empty) | — | — | — | 604 | 1,178 |

The counts line up 1 : 1 : 1 — one empty inline-asm region, one `cmp 0, flag`, one `s_cbranch_vcc` — and
that triple reproduces in isolation (`/home/vermont/tmp/asmprobe/probe.cu`, same clang/flags/`gfx1102`
as `build-hip/compile_commands.json` for this TU): a kernel making **64 `mma_bf16` calls emits exactly
64 ASM region pairs + 64 `v_cmp_ne_u32_e32` + 64 `s_cbranch_vcc`** (452 instructions, 69 VGPR, private
segment 0); a kernel making 32 `bcast_u32` calls, folded by LLVM into 8 unrolled copies, emits **8 of
each**. So the triple is emitted **per unrolled copy of a lane-broadcast/contraction helper call site**,
and the 575 in the served loop are that pattern, not mask arithmetic. Source-side scan of all 908 files
in `src/` found 70 `asm volatile` sites and **none** in the served route's HIP-visible arms
(`mma.cuh:457-563` ldmatrix/`mma.sync` are in the `#else` CUDA branch after `mma.cuh:453`;
`math.cuh`'s `tanh.approx`/`ex2.approx`/`cvt.rn.bf16x2.f32` are likewise `#else`; `src/hipshim/`
contains **zero** `asm`), which is why all 1,208 markers in the served HIP body are empty.

**Attribution is now complete (measured, probe IR).** `-emit-llvm` on the probe shows each site as
`%r = tail call i32 asm sideeffect "", "=v,0"(i32 range(i32 0, 2) 1) #5` with `attributes #5 = {
convergent nounwind }`, followed by `icmp ne %r, 0` → `llvm.amdgcn.ballot.i32` → `icmp eq …, 0` →
`br`, i.e. ROCm's **convergence guard around a convergent shuffle — a ballot spin**. Lowered for
`gfx1102` it is exactly the served shape, verbatim (`served_body.s` 13123-13138, at
`.LBB0_853: ; %_ZN6ninfer3ops10mma_detail9bcast_ptrEmj.exit` — the guard sits at a `bcast_ptr` exit):

```
v_mov_b32_e32 v0, 1          ;;#ASMSTART ;;#ASMEND      (empty: asm sideeffect "" → no instructions)
s_delay_alu instid0(VALU_DEP_1)
v_cmp_ne_u32_e32 vcc_lo, 0, v0
s_cbranch_vccz .LBB0_853
.LBB0_852:                 ; %.peel.next.i.i.i.i282.i383  ← the spin header
v_mov_b32_e32 v0, 0        ;;#ASMSTART ;;#ASMEND
s_delay_alu instid0(VALU_DEP_1)
v_cmp_ne_u32_e32 vcc_lo, 0, v0
s_cbranch_vccnz .LBB0_852  ; while (ballot != 0) spin
```

The predicate is a **compile-time constant** (`v0 = 1`, then `v0 = 0`), so the spin's branches are
provably degenerate — they survive because the marker asm is `sideeffect` **and** `convergent`, which
makes it immortal to LLVM's folding. Source of the convergent ops: `mma.cuh:30` `bcast_u32` and `:34`
`bcast_ptr`, both `__shfl_sync(0xFFFFFFFFu, …)`, reached from `gather8` (`:42-51`, 32 per emulated
contraction call) and from the ldmatrix reference `bcast_pair_t` (`:66-75`). Served census of the shape
in the nested loop: **320 `ds_bpermute`, 287 guards (287 `s_cbranch_vccz` + 287 `s_cbranch_vccnz`), 287
`.peel*` labels**; the main loop's own region has 84 bpermutes / 14+14 guards.

Why it matters (and what it is *not*): the marker carries **no `memory` clobber** —
`attributes #5 = { convergent nounwind }` — so it is not a memory barrier and does not block load/store
hoisting. Its cost is its own body: each guard is ≥ `v_mov` + `v_cmp` + `s_cbranch_vcc` +
`s_delay_alu`, and the served loop has **574 guard branches (287 spins) ≈ 1,720 instructions = 45% of
the 3,790-instruction nested loop, 30% of the 5,747-instruction main loop**, for a convergence condition
the compiler itself has already proven constant. The 151-instruction QK wmma runs of (b) contain **none**
of this shape (0 guards in 151 instructions). So the instruction budget at the emulation sites is paid
twice: ~128 VALU + 32 convergent shuffles inside each contraction, plus a ballot-spin guard per unrolled
shuffle copy.

**The lever this names — measured, refuted, kept for the record in §27.** The guard exists only because
the broadcast is a *convergent wavefront shuffle*. `v_perm_b32` is not convergent and not LDS-pipe traffic — it is a per-lane byte select from
two registers — so a broadcast staged as "LDS128 into this lane's registers, then `v_perm_b32` byte
selects" carries **no guard, no ballot, no peel** and no `ds_bpermute`. That is precisely the shipped
GEMM code-fragment shape that produced the +15.1% (7042f064; memory
`v-perm-b32-packed-byte-lookup`), and `gather8`'s 8 outputs come from exactly 2 source registers, which
is a `v_perm_b32` table. Cost sketch: per `gather8`, 8 `ds_bpermute` + 8 guards → ~1 `ds_load_b128` +
8 `v_perm_b32` (the register-table cost is live registers, which is what §26 says is tight). Reducing
the **number of unrolled shuffle copies** therefore pays in shuffles, guards and VALU at once. **No
throughput claim is made** — instruction counts have twice failed to predict the sign of a throughput
change in this class (§16-§22), and the serving point stays 203 t/s (7042f064) until an ABBA says
otherwise. Nothing here is committed; any A/B needs the card idle-checked, `flock /tmp/ninfer-gpu.lock`
taken and released inside one invocation, and a coherence check.

## §26 — Register lifetime at the served tile: the occupancy lever is closed, and the accumulator is not the difference

Static only — `tiled_mma.cuh`, `tile_io.cuh`, the `.amdhsa_*` metadata from the three dumps. No GPU, no
lock, no measurement, no t/s claim.

**a. Occupancy is pinned by LDS and cannot move.** The served schedule `Bf16KvTiledMmaSchedule<64,32,1>`
declares `q_s` 32,768 B + `k_s` 16,384 B + `v_s` 16,384 B = **65,536 B** (`tiled_mma.cuh:70-73`), which
equals the measured dynamic LDS, i.e. the whole gfx1102 per-CU budget ⇒ **exactly 1 workgroup per CU**.
Two workgroups would need ≤32 KB; the Q tile alone is 32,768 B. Static LDS is 0. `lds_fit.cuh` selects
this schedule and it is the only instantiated one, so tile shape is not a runtime knob. There is no
occupancy lever left inside this kernel's LDS budget.

**b. The accumulator is the same size on both routes, so it is not what makes wmma-PV fail.** Served:
`float acc[PVNt][4]` = 32 × 4 = **128 fp32 live per lane** (`:152`). Native: `Bf16WmmaC acc16[CTiles]`
= 16 × 8 = **128 fp32 live per lane** (`:161`; the source comment at `:159-160` says exactly that).
What the native route adds on top is `cd[NT16][8]` = 16, `pw[NT16][4]` = 8, `afrag[NT16]` = 8, plus the
`int4` LDS128 pairs hoisted to feed each WMMA (`:391-438`) and the alpha-rescale temporaries. Against
256 VGPR with the served arm already spilling 608 B/lane (§24f), that is the difference between the
served arm and the two gate arms (macro 1: 1,090 in-body scratch, 2,472 B/lane; macro 2: 1,395, 1,856).

**c. The one static shape that halves the live set without duplicating the dominant work: split D across
warp pairs.** `D = 256` and the 128 live accumulators are the D-wide output tile held per lane. Let
warp 0-1 own columns 0-127 and warp 2-3 own 128-255: accumulators drop **128 → 64 fp32/lane** and PV
wmma per warp per step drop **32 → 16**. QK, the causal mask, the softmax reductions and the P repack
must then be shared, since each warp pair only computes half of the K-side scores per row; sharing the
P tile through a ~4 KB LDS buffer costs **4 `ds_load_b128` per lane per step** and one more barrier, and
keeps the QK/softmax/P VALU **un-duplicated**. It needs ~4 KB under the 65,536 B cap; single-staging K
frees 16 KB.

**d. The QK-duplicating variant is rejected statically.** Having each warp pair recompute QK for its
half of D doubles the mask/softmax/P VALU — which is the dominant share of the main loop (the emulated
contractions plus, per §25h, the scaffolding at their call sites) — to save LDS it cannot use, since
occupancy is pinned at 1 WG/CU by (a). Doubling the dominant VALU to halve a register set that is
already at 256 VGPR is not a trade worth an A/B.

**e. Standing, un-exercised levers, ranked by what the statics say.** (i) §25h's site-count lever
(fewer unrolled emulation copies ⇒ fewer scaffolds **and** fewer VALU/shuffle bodies at once). (ii) (c)'s
warp-pair D-split, the only shape that lets wmma-PV fit registers. (iii) GEMM-side: the ~295 integer
fragment-build ops per group, the 2× activation-load amplification, `s_delay_alu` 11.5% / `s_waitcnt`.
(iv) re-run the cal71 occupancy sweep with `G = 32*MB` — cal70/cal71 hardcode 112/16 CU, so their
occupancy labels are ~2× inflated on this 32-CU card. Every one of these is A/B-gated: card idle check
(`/sys/class/drm/card1/device/gpu_busy_percent`), `flock /tmp/ninfer-gpu.lock` taken and released inside
**one** invocation, ABBA pairs inside a single hold, unfavourable results preserved here. Nothing in §26
is committed. **(i) has since been measured and is worse — see §27.**

## §27 — The §25h lever, measured: removing the ballot guards costs more than the guards

Compile-only, no GPU, no lock. `/home/vermont/tmp/asmprobe/probe2.cu`, built with the project's exact
HIP flags for gfx1102 (`/usr/lib/rocm/llvm/bin/clang++ -x hip -DNINFER_BACKEND_HIP -DNINFER_SM8X_COMPAT
-O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx1102`). Two kernels with **identical arithmetic** and the
served PV shape (64 emulated `mma_bf16` contractions, A = live `p_frag[2][4]`, B = 32 pipelined
`ldmatrix_x4_t` loads, 32 accumulator tiles = 128 fp32/lane):

| | A: as shipped (`__shfl_sync`) | B: `__builtin_amdgcn_ds_bpermute` |
|---|---|---|
| body instructions | **3,752** | **6,006 (+60%)** |
| `;;#ASMSTART` pairs | 552 | **0** |
| `v_cmp_ne_u32_e32` | 552 | **0** |
| `s_cbranch_vccz` + `s_cbranch_vccnz` | 276 + 276 | **0 + 0** |
| `ds_bpermute_b32` | 296 | **800 (+170%)** |
| `s_delay_alu` | 486 | 756 |
| `s_waitcnt` | 126 | **608 (+383%)** |
| scratch ops | 86 | 3 |
| `num_vgpr` | 90 | **159** |
| `private_seg_size` | 560 B/lane | 48 B/lane |

B does exactly what §25h predicted for the guard: **every** ASM pair, compare and `vcc` branch is gone.
It is also **60% larger**, with **2.7× the shuffles** and **4.8× the `s_waitcnt`**.

**Mechanism (this is the part worth keeping).** `__shfl_sync` is `llvm.nvvm.shfl.sync.idx.i32`
(`/usr/lib/rocm/llvm/lib/clang/20/include/__clang_cuda_intrinsics.h:173`, `__MAKE_SYNC_SHUFFLES`), a
**pure, CSE-able** call in LLVM: identical `(value, source-lane)` pairs collapse, and the fetches whose
address is a compile-time-known LDS address fold to address math with no shuffle at all. The served PV
needs 32 gathered words × 64 contractions = **2,048 shuffles per warp-step of raw demand; A emits 296.**
The convergence guard is the *price of that property*: a `convergent` + `sideeffect` marker is what
stops LLVM from speculating the fetch across control flow, and the folds it permits are worth far more
than the scaffolding they cost. B keeps the fetches as opaque per-lane builtins: no guard, far less
spilling (48 vs 560 B/lane) — and the folds largely disappear, so shuffles, `v_mov`s and latency waits
come back.

**Consequences for the plan, all unfavourable and all preserved here.**
1. The §26(e)(i) lever ("fewer unrolled shuffle copies ⇒ shuffles + guards + VALU at once") **is not
   available by changing the intrinsic.** Do not spend GPU time on the builtin swap.
2. §25h's arithmetic — "574 guard branches ≈ 1,720 instructions = 45% of the nested loop" — is *true as
   a census* and **misleading as a lever**: that scaffolding is load-bearing. Removing it does not shrink
   the loop; measured, it grows it.
3. This also supplies the missing mechanism for §16-§22's instruction-count→throughput failures (base →
   candidate −23.5% instructions for +15.1% throughput, 0.64× conversion): the served `.s` is not
   work-plus-overhead. Scaffolding, shuffles and waits are the visible residue of a fold structure that
   is removing ~85% of the mathematically required shuffles, so counting them cannot predict cycles.
4. Spill behaviour is the mirror image: A spills more (560 B/lane, 86 scratch ops) and issues less; B
   spills almost nothing and issues 60% more. Register pressure is not a proxy for speed in this class,
   in either direction.

**What remains open, unmeasured.** The `v_perm_b32` idea from §25h only removes *intra-lane* byte
selection: `v_perm_b32` reads two of **this lane's** registers, so it can replace a table lookup (what
the GEMM path used it for, 7042f064) but **cannot replace a cross-lane fetch** — the group's other
lanes' words are not in this lane's registers. A cross-lane path needs LDS staging, and LDS is at the
65,536 B cap (§26a), so it needs freed bytes before it can be tried at all. The served point stays
**203 t/s (7042f064)**; nothing in §27 is committed, and no throughput claim is made for A or B.

## §28 — Every tile shape the schedule family admits is pegged at 4 warps/CU; the register wall moves in exactly one shape

Compile-only, no GPU, no lock. `/home/vermont/tmp/asmprobe/probe3.cu` instantiates
`bf16_kv_tiled_mma_kernel<256,24,4, Bf16KvTiledMmaSchedule<Br,Bc,MinBlocks>, PagedKVDirectMetadata>`
for nine (Br,Bc,MinBlocks) combinations and is built with the project's exact HIP command recovered
verbatim from `build-hip/compile_commands.json` (`/usr/lib/rocm/llvm/bin/clang++ … -O3 -DNDEBUG
-std=gnu++20 --offload-arch=gfx1102 -x hip -c --save-temps=obj`), giving
`/home/vermont/tmp/asmprobe/out3/probe3-hip-amdgcn-amd-amdhsa-gfx1102.{s,o}`. Census script
`/home/vermont/tmp/asmprobe/shapecensus.py` (run `python3 -I shapecensus.py <path.s>` from
`/home/vermont/tmp/asmprobe`); spills from `/usr/lib/rocm/llvm/bin/llvm-readelf -n <path.o>` (GNU
`readelf` on this machine does not decode the AMDGPU notes and prints nothing).
**Index 0 is the served shape as a control and it reproduces the independently measured served
kernel exactly — 256 VGPR, 608 B/lane private, 55 SGPR, 19 VGPR spills, 14,202 body instructions —
so the other eight rows are trustworthy.**

| shape (minblk) | thr | LDS B | blocks/CU | **warps/CU** | VGPR | priv B/lane | vgpr spills | body instr | bperm | `v_wmma` | scratch | guard-class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **64×32 mb1 (served)** | 128 | 65,536 | 1 | **4** | 256 | 608 | 19 | **14,202** | 812 | 64 | 336 | 17.1% |
| 64×16 mb1 | 128 | 49,152 | 1 | **4** | 222 | 528 | **0** | **12,522 (−11.8%)** | 748 | 32 | 256 | 19.4% |
| 32×16 mb2 | 64 | 32,768 | 2 | **4** | 256 | 976 | 215 | 20,964 (+48%) | 1,356 | **0** | 863 | 22.9% |
| 32×16 mb1 | 64 | 32,768 | 1 | 2 | 256 | 976 | 218 | 21,023 (+48%) | 1,356 | **0** | 853 | 22.8% |
| 32×32 mb1 | 64 | 49,152 | 1 | 2 | 256 | 976 | 223 | 20,169 (+42%) | 1,340 | **0** | 796 | 23.6% |
| 16×16 mb2 | 32 | 24,576 | 2 | 2 | 256 | 880 | 177 | 19,783 (+39%) | 1,356 | **0** | 703 | 24.2% |
| 64×32 mb2 | 128 | 65,536 | 1 | **4** | 256 | 608 | 19 | 14,202 | 812 | 64 | 336 | 17.1% |
| 128×32 mb1 (Tiled128, undescended) | 256 | 98,304 | **0 — over ceiling** | – | 256 | 1,040 | 239 | 20,657 | 1,340 | **0** | 844 | 23.1% |
| 64×64 mb1 (Tiled64, undescended) | 128 | 98,304 | **0 — over ceiling** | – | 256 | 800 | 87 | 14,620 | 940 | 128 | 450 | 16.6% |

`sgpr_spill_count` is 0 in all nine. LDS is `(Br + 2·Bc)·512` B (`bf16_kv_tiled_shared_bytes`), the
launch ceiling is 64 KiB (`kBf16CausalLdsCeilingBytes`), warps/block = `Br/16` = `kWarps` (codified by
`static_assert(Br == Schedule::kWarps * 16)`, `tiled_mma.cuh:67`), and 256 VGPR × 32 lanes = 8,192 of
the CU's 64K registers leaves 8 waves/CU, so **LDS is the binding limit in every admissible shape**.

**1. The occupancy ceiling is 4 warps/CU = 1 wave per SIMD, and no shape in the family reaches 2.**
`max warps/CU = floor(65536 / ((Br+2Bc)·512)) · (Br/16) ≤ 4` for every shape: halving `Br` to 32 drops
to 2 warps/block, so doubling blocks/CU is a wash (32×16 mb2: 4 warps/CU, exactly the served
figure). RDNA3 gives 8 wave slots per SIMD, so the served kernel runs **12.5% of the machine's wave
capacity** and the template knobs cannot change that. The 2-wave point that §26c wanted requires
**new code, not a template argument**: the accumulator is shape-independent — `PVNt = D/8 = 32` ⇒
`acc[32][4]` = 128 fp32/lane in *all nine* shapes — so only a warp-pair split of D (two warp pairs
each owning half of D, 32 query rows per block ⇒ LDS (32+2·16)·512 = 32,768 B ⇒ 2 blocks/CU ⇒ 8
warps/CU) halves accumulator liveness *and* doubles waves. Nothing existing does that.

**2. `min_blocks` is inert.** 64×32 mb2 is **census-identical** to 64×32 mb1 (same VGPR, same
private, same 19 spills, same 14,202 instructions, same bpermute/scratch/guard counts). The compiler
does not honour the occupancy request when LDS cannot satisfy it, and `.amdhsa_min_blocks` is not
emitted in this output at all. The "retune `__launch_bounds__` / min-blocks for occupancy" lever is
**closed by measurement**, not by argument.

**3. The QK tensor-core path compiles only for `QueryRows == 64`.** `tiled_mma.cuh:227` gates the
native site — `if constexpr (Schedule::kQueryRows == 64) { … __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
(…) }` — and everything else falls to the emulated `ldmatrix` + `dot_bf8` path at line 265. The census
confirms it: Br=32 and Br=128 shapes emit **zero** `v_wmma` instructions. So the tile shape is **not a
neutral knob** — changing it off 64 silently changes the *implementation class* of the QK site, which
is a large part of why the small-tile bodies grow 39–48% while doing less work per block. (Side note
for anyone reading a `.s` dump: the `_w32` intrinsic lowers to `v_wmma_f32_16x16x16_bf16`, **no `_w32`
suffix** — 288 copies across this TU; the served kernel has 64, of which 32 are in the main loop.)

**4. (64,16,1) is the only shape that relieves register pressure — and it buys no occupancy.** It is
the table's **only zero-spill** row (222 VGPR, 0 spills, 528 B/lane private) and the smallest body
(12,522, −11.8% vs served), it keeps the wmma QK path (32 `v_wmma` in the main loop), and its 49,152 B
LDS leaves **16,384 B free** — enough for the ~4 KB P-sharing that §26c said the 65,536 B cap blocked.
Costs, measured: occupancy is unchanged (128 threads, 1 block/CU, 4 warps/CU), and K/V tile fills
**double** (Bc 32→16). The private-segment arithmetic is consistent with spills + address-taken
residue: served 608 = 528 + 19·4 (76); the 528 B appears in both shapes and is not spill traffic.
**No throughput claim**: §27 item 3 established that instruction count does not convert (−23.5% body
for +15.1% throughput, 0.64×), so a −11.8% body with 2× tile fills is a *question for an A/B*, not a
prediction. It has not been measured on the card.

**5. Plan choice is inert on this card.** `Tiled64 = ⟨64,64⟩` and `Tiled128 = ⟨128,32⟩` both need
98,304 B > the 64 KiB ceiling, and `Bf16KvTiledLdsFit` (`template_launch.cuh:84`) descends both —
halving KeyRows to 16 first, then QueryRows — to the **same ⟨64,32⟩ kernel** as the served shape.
Undescended, they spill 87 and 239 and 128×32 loses wmma entirely. On gfx1102 there is exactly **one**
attention tile plan, not two.

**6. Spill census across the served attention family** (served build,
`/tmp/attn-asm/cand/…gfx1102.o`): tiled-MMA prefill 19/20 spills, grouped decode `⟨16,32,2,2,1⟩` 98,
Grouped32 `⟨32,32,1,2,0⟩` 164, merge kernels 0. Prefill is the *least* pressured kernel in the family —
consistent with §27's conclusion that spills are not the throughput lever.

**Served main-loop census, for reference** (5,747 instructions, lines 10239–19690 of
`/home/vermont/tmp/isacensus/served_body.s`): guard-class 1,831 = **31.9%**, `ds_bpermute_b32` 404,
`s_waitcnt` 247, scratch 104, `v_wmma` 32 ⇒ **179.6 instructions per WMMA**. That ratio, not the
absolute count, is the number a real lever has to move.

The served point stays **203 t/s (7042f064)**. Nothing in §28 is committed, no shape here is
measured on the card, and no throughput claim is made for any of them.

## §29 — A tile shape moves the step count as well as the step size: every shape that fits in LDS costs more executed work than served, and every shape that costs less does not fit

Compile-only, no GPU, no lock. Same assembly as §28
(`/home/vermont/tmp/asmprobe/out3/probe3-hip-amdgcn-amd-amdhsa-gfx1102.s`); tool
`/home/vermont/tmp/asmprobe/loopsplit2.py`, run as `python3 -I loopsplit2.py <path.s>` from
`/home/vermont/tmp/asmprobe`.

**Why §28's `body` column cannot be read as throughput.** A body is per-step code plus
once-per-kernel code. A tile shape changes the size of the per-step code *and* how many times it runs:
the key loop trips `n_block_max = max_query_abs / Bc + 1` times (`tiled_mma.cuh:167`). ⟨64,16⟩'s
−11.8 % body (§28) is per-step code that then runs twice as often. Executed work is
`A + peel + (steps−1)·S`: once-per-kernel code, plus one peeled copy of the diagonal step, plus the
steady loop. All three are separated below by fingerprint, not assumed.

**Loop identification, and one measurement lesson.** A first version paired loop headers with
backward `s_cbranch*` edges only. That **missed the key loop in every peeled shape**, because LLVM
rotates those loops: the back edge is an unconditional `s_branch` and the exit test branches to a
label outside the range. Its "NO barrier loop found" rows for ⟨64,16⟩, ⟨32,×⟩ and ⟨128,32⟩ were an
artifact of that regex, not a property of the shapes. The current version accepts
`s_cbranch* | s_branch | s_setpc_b64` back edges, merges nested ranges, and identifies the key loop by
`≥2 s_barrier` (the step does `__syncthreads()` twice) and `size ≥ 1000`. WMMA is deliberately **not**
a gate: the `QueryRows != 64` shapes compile the QK product through the emulated ldmatrix + `mma_bf16`
path and contain zero `v_wmma` (§28), so a wmma-gated filter silently drops their key loop. Where WMMA
exists it confirms the split exactly — per-step `v_wmma` = `NT16·QKKs = (Bc/16)·(D/16)` = 32, 16, 64 for
⟨64,32⟩, ⟨64,16⟩, ⟨64,64⟩, unroll factor ×1 in all nine shapes. Every kernel body contains exactly
4 `s_barrier`s: 2 in the steady loop, 2 in the peeled copy. LDS traffic on gfx1102 appears as
`flat_read_*`/`flat_write_*` (`ds_read_*` count is 0 in every shape).

**Table A — one query block over a dense 1024-key context** (per-step code `S`, peeled copy `peel`,
once-per-kernel `A`, `steps = 1024/Bc`):

| shape | body | S per step | bar/wmma/exp2 | peel | A once | steps | exec@1024 | ratio |
|---|---|---|---|---|---|---|---|---|
| **⟨64,32⟩ mb1 (served)** | 14,202 | **5,747** | 2/32/18 | 5,385 | 3,070 | 32 | **186,612** | **1.000** |
| ⟨64,32⟩ mb2 | 14,202 | 5,747 | 2/32/18 | 5,385 | 3,070 | 32 | 186,612 | 1.000 |
| ⟨64,16⟩ mb1 | 12,522 | 4,837 | 2/16/10 | 4,776 | 2,909 | 64 | 312,416 | 1.674 |
| ⟨64,64⟩ mb1 | 14,620 | 6,263 | 2/64/34 | 5,671 | 2,686 | 16 | 102,302 | 0.548 |
| ⟨32,32⟩ mb1 | 20,169 | 8,825 | 2/0/18 | 8,688 | 2,656 | 32 | 284,919 | 1.527 |
| ⟨32,16⟩ mb1 | 21,023 | 8,982 | 2/0/10 | 8,852 | 3,189 | 64 | 577,907 | 3.097 |
| ⟨32,16⟩ mb2 | 20,964 | 8,915 | 2/0/10 | 8,858 | 3,191 | 64 | 573,694 | 3.074 |
| ⟨128,32⟩ mb1 | 20,657 | 8,967 | 2/0/18 | 8,813 | 2,877 | 32 | 289,667 | 1.552 |
| ⟨16,16⟩ mb2 | 19,783 | *no key-loop fingerprint — see fact 5* | | | | 64 | not derived | — |

**Table B — a whole 1024-token causal prefill** (a prefill is `1024/Br` query blocks; block *j* trips
`floor((Br·(j+1)−1)/Bc) + 1` times, so the step total is `Σ_j` over blocks, and the once-per-kernel code
runs once per block). `instr/row-key` = executed instructions ÷ query-rows × keys-the-kernel-actually-sweeps:

| shape | blocks | Σ steps | exec-causal | ratio | row×keys | instr/row-key | LDS `(Br+2Bc)·512` | fits 64 KiB |
|---|---|---|---|---|---|---|---|---|
| **⟨64,32⟩ served** | 16 | 272 | **1,606,512** | **1.000** | 557,056 | **2.884** | 65,536 | yes |
| ⟨64,16⟩ mb1 | 16 | 544 | 2,676,896 | 1.666 | 557,056 | 4.805 | 49,152 | yes |
| ⟨32,32⟩ mb1 | 32 | 528 | 4,740,208 | 2.951 | 540,672 | 8.767 | 49,152 | yes |
| ⟨32,16⟩ mb2 | 32 | 1,056 | 9,514,528 | 5.922 | 540,672 | 17.598 | 32,768 | yes |
| ⟨128,32⟩ mb1 | 8 | 144 | 1,313,032 | 0.817 | 589,824 | 2.226 | 98,304 | **no** |
| ⟨64,64⟩ mb1 | 16 | 136 | 885,272 | 0.551 | 557,056 | 1.589 | 98,304 | **no** |

The two columns disagree in exactly one place, and the disagreement is the point: the dense column
counts one query block, the causal column counts all of them. A shape with `Br = 32` runs twice the
blocks (32 per 1024 tokens), so its once-per-kernel code runs twice and its step count is four times
served's; a shape with `Br = 128` runs half the blocks. Table B is the one to read against a
1024-token prefill measurement.

**1. `min_blocks` is inert in the split too.** ⟨64,32⟩ mb2 is census-identical to mb1 in §28 and
identical here down to the loop boundaries (same 5,747 / 5,385 / 3,070, same 1,606,512). Nothing to
pursue.

**2. ⟨64,16⟩ — the §28 "cheapest kernel" — executes 1.666× served's work.** Its per-step code is
4,837, i.e. 84.2 % of served's 5,747, while the step count doubles. Its probe advantages are real
(0 spills vs 19, 222 VGPR vs 256, 49,152 B LDS) and they buy nothing at this density: 4.805 vs 2.884
instructions per query-row per key, **+66.6 %**. The planned ⟨64,16⟩ A/B is a qualification of this
prediction, not a test of a hoped-for win.

**3. Per-step cost has a large Bc-independent floor** (steady-loop census, Bc = 16 → 32 → 64):
`ds_bpermute` 372 → 404 → 468; scratch 80 → 110 → 190; `s_waitcnt` 199 → 248 → 259; `v_exp` 10 → 18 → 34;
`flat_*` (LDS) 40 → 40 → 40. Instructions per WMMA per step: **302.3 → 179.6 → 97.9**. Halving the key
tile removes 16 % of a step and doubles the step count; doubling it costs +9 % per step and halves the
count. §28 ended with "that ratio, not the absolute count, is the number a real lever has to move" —
measured here, that ratio moves in exactly one direction, and that direction is the one the LDS ceiling
blocks.

**4. The only two shapes that beat served on executed work are the two that do not fit.** ⟨64,64⟩ at
0.551× (1.589 instr/row-key, −44.9 %) and ⟨128,32⟩ at 0.817× (2.226, −22.8 %; it also sweeps 5.9 % more
row×keys than served, because a 128-row block's causal window is quantized to 128 keys). Both need
`(Br + 2·Bc)·512 = 98,304` B against the 65,536 B per-block ceiling — a deficit of exactly 32,768 B,
one full tile.

**5. ⟨16,16⟩ mb2: unverified, no claim.** No barrier-bearing range ≥1,000 instructions exists in it;
its two largest loops (8,623 and 8,570 instructions) contain 0 barriers. The split does not attribute a
key loop for it, so it gets no ratio here.

**Q in registers is not the route** (arithmetic, not measurement). A `wmma_f32_16x16x16_bf16_w32`
A-fragment is a 16×16 bf16 tile spread over 32 lanes = 8 elements = 4 VGPR/lane. `QueryRows = 64`
needs 4 row-tiles × 16 K-chunks = 64 fragments = **256 VGPR/lane** of resident Q, on top of the 128
fp32 accumulator (§28) and a served kernel that already reports 256 VGPR with 19 spills. At ≥512
VGPR/lane the CU admits 2 waves, and a ⟨64,64⟩ block is 4 warps — it cannot be scheduled.

**The deficit is one tile, and splitting K's `D` covers it** (arithmetic, not measurement).
Q 64·512 = 32,768 + K 32,768 + V 32,768 = 98,304. Stage K in two 128-wide `D` halves (K LDS 16,384, the
QK product accumulated across the two passes) and V in two 32-key chunks (V LDS 16,384):
32,768 + 16,384 + 16,384 = **65,536 = the ceiling exactly**. Global traffic per key is **unchanged** —
K 2×16,384 + V 2×16,384 = 65,536 B per 64-key step = 1,024 B/key, and served moves 32,768 B per
32-key step = **the same 1,024 B/key**. What it costs is a doubled live range on the score accumulator (`NT16·4` = 16
fp32/lane per 16-row tile × 4 tiles = 64 VGPR/lane) and a second K pass per step. At ~320 VGPR/lane the
CU admits 6 waves ≥ the 4 a ⟨64,64⟩ block needs. This is a plan, not a result: nothing in it is built.

**Predictions, labelled as predictions.** Using the one exponent this project has measured (−23.5 %
executed instructions → +15.1 % throughput, i.e. time ∝ instr^−0.525, from the pooled
176.53 → 203.13 ABBA pair at 7042f064): ⟨64,64⟩ at 0.551× predicts attention time ×0.731, total
×0.921 → ~221 t/s. ⟨128,32⟩ at 0.817× predicts ×0.899 → ~209 t/s. ⟨64,16⟩ at 1.666× predicts
**−8 %** → ~187 t/s. That is one data point defining the exponent, two of the three shapes are not
launchable as compiled, and none of the three is measured. Read it as a ranking, not a forecast.

**Arms recipe — measured fact about the control gate.** `/home/vermont/tmp/tile16-ab/build-arm.sh`
stopped itself: compiling the unpatched tree sources through the recorded
`build-hip/compile_commands.json` command reproduces `plan.cpp.o` byte-for-byte (sha1 prefix
`00579a45e3`) and does **not** reproduce `launch.cu.o` (tree `b940466764`, control `aeba939408`). The
stored object predates the §27 probe edits to `tiled_mma.cuh`/`tile_io.cuh`, and the recorded CU
command carries `-lineinfo`, so added source lines change object bytes even when the emitted code is
inert. Byte-exact equality with the tree is not reachable from a snapshot build either
(`-ffile-prefix-map=/home/vermont/git/ninfer-3090=.` embeds the build path), so a snapshot-based arms
build must gate on the kernel metadata census — base 256 VGPR / 608 B private / 19 spills vs candidate
222 VGPR / 0 spills / 49,152 B LDS — and on base ≠ candidate, not on byte equality with the tree.

**2026-10-24, that recipe sentence is half wrong.** Measured from `build-snap.sh`: a compile of the
**snapshot's** unpatched `plan.cpp` through the recorded command produced `/home/vermont/tmp/tile16-ab/obj/ctrl-plan.cpp.o`
with sha1 `00579a45e3b0dc337cd74b843983b7e197f06ef3`, byte-identical to the tree's
`build-hip/src/ops/CMakeFiles/ninfer_ops.dir/.../plan.cpp.o`. So the parenthetical reason given above is
false: `-ffile-prefix-map` (which the driver splits into `-fmacro-prefix-map` and `-fdebug-prefix-map`,
visible in the `-cc1` line) removes the build path from the object entirely, and a snapshot build
**can** reproduce tree bytes. The `launch.cu.o` mismatch in that recipe has a different cause, which the
recipe already named: the stored tree object predates the §27 probe edits to `tiled_mma.cuh`/`tile_io.cuh`
and the command carries `-lineinfo`, so added source lines move object bytes even when the emitted code
is inert. Gate A on `plan.cpp.o` is therefore a real pipeline proof; on `launch.cu.o` it is not, and the
metadata census stays the gate for the kernel.

**Two arms-harness bugs found by running it, both silent** (recorded because both were one step from
reporting a fake result):
1. `run()` stripped `${cmd% -c *}` to drop the source path, which deletes the `-c` flag too; clang++ then
   compiles **and links**, and for the HIP file the link dies as `ld.lld: error: unable to find library
   -lineinfo` — lld reads `-lineinfo` as `-l ineinfo`. The fix drops only the trailing path and requires
   the rewritten command to end in `-c`.
2. The gates hashed `<tag>-launch.o` while the recorded `-o` path yields `<tag>-launch.cu.o`, so the
   comparisons ran against missing files. `cmp -s missing other` exits **2**, which reads as "not
   identical" — that combination prints a **GATE C PASS with no objects present**. The gates now require
   the objects to exist before comparing.

**Measured fact about the archive: `ar r` cannot target the served attention stub.** Re-measured 2026-10-24
on `build-hip/src/ops/libninfer_ops.a` (412,715,246 B, `ar -t` lists **367** members): the member name
`launch.cu.o` occurs **6** times exactly (ar list indices 122, 329, 332, 335, 337, 339), 8 names end in
`launch.cu.o`, 29 end in `plan.cpp.o`, and CMake stores plain basenames — `ar -t` shows no directory
prefixes, so nothing in the member table says which `launch.cu.o` is the bf16 causal_cache one.
`ar r lib.a launch.cu.o` replaces the member whose **basename** matches, which is a choice among six op
directories; a wrong hit is silent, and it is exactly the kind of arms bug that shows up as a clean
measurement of the wrong kernel. The arms therefore never edit the archive: the candidate's
`launch.cu.o` + `plan.cpp.o` go on the **link line before** `libninfer_ops.a`, so `ld` takes their
definitions first and a duplicate from the archive fails the link loudly. Which definitions actually won
is then read off the linked binary (`nm -a` schedule-symbol census), not assumed. Hand-parsing the
archive is also unsound here: member 0 is the `/` symbol table whose declared size does not match the
byte gap to the next header.

The served point stays **203 t/s (7042f064)**. Nothing in §29 is committed, nothing in §29 is measured
on the card, and every throughput figure in it is labelled a prediction.

### §29 correction (2026-10-24): served's per-key traffic was overcounted, so the split-⟨64,64⟩ plan
### buys nothing on bandwidth — it is traffic-neutral

Compile-only, no GPU, no lock. §29's plan paragraph claimed the split route cuts global traffic per key
from served's 1,536 B to 1,024 B. That 1,536 was a per-step miscount: it divided the *whole* LDS footprint
`(Br + 2·Bc)·512 = 49,152` B by the 32 keys in a step, which charges Q's staging to every step. Q is
staged **once per kernel**, not once per step — "Stage Q into smem once via cp.async (overlaps with the
K(0) prologue load below); it stays resident for the key loop" (`tiled_mma.cuh:126-128`), and the loop
census puts Q staging in the once-per-kernel bucket (`A` = 3,070 in served, Table A), where it is 512 B
per query *row*, not per key.

Measured, from the served steady loop's opcode census (`out3/probe3-…-gfx1102.s`, range 10239-19690):
32 `global_load_b128` + 1 `global_load_b32` + 40 `flat_load_u16`. A `b128` cp.async is 16 B/lane, so the
*compiled* loop body carries 32 × 16 B × 128 threads = 65,536 B of staging capacity per step. The tile
only needs 32,768 B (K `Bc·D·2` = 16,384 + V = 16,384), and the reason the census sees 2× is
`tile_io.cuh:35-55`: `bf16_kv_stage_tile` compiles **both** staging paths into the loop body — the
`full_tile` fast path (8 `cp_async<16>` per tile) and the predicated causal-tail path (8 `cp_async<16>`
plus 8 zero-fill `store_vec`s) — and one step takes one of them. Executed staging on a full-tile step is
16 `b128` loads = 32,768 B = 1,024 B/key, the same density the split-⟨64,64⟩ plan would move.

Consequences, stated as the arithmetic that follows, not as opinions:

1. The split-⟨64,64⟩ plan's **only** quantified benefit is the instruction-count density (1.589 vs
   2.884 instr/row-key, §29 Table B). It does not reduce global traffic, LDS read traffic, or bytes per
   key, and the earlier "1,536 → 1,024 B/key" sentence is deleted from the plan.
2. That makes its cost side heavier than §29 presented: a doubled live range on the score accumulator
   (~64 VGPR/lane), a second K pass per step, and a re-partitioned swizzle, bought for a predicted
   instruction-count gain whose throughput translation rests on a single measured exponent (−0.525).
   Its priority is **below** what §29 implied.
3. Bounded, unfavourable side note on the census itself: the compiled per-step counts in §29's Tables
   A/B include the untaken predicated staging path, so they overstate *executed* work by the amount of
   that path — ~24 of served's 5,747 (0.4 %), ~12 of ⟨64,16⟩'s 4,837 (0.2 %). Both shapes carry the same
   dual-path structure, so the ratios between shapes are unaffected at this precision; the density
   column should be read as compiled-loop density, and it is not a bandwidth measurement.

## §30 — The ⟨64,16⟩ arms exist and the route is qualified by symbol census; two of the four gates
### behave as §29 predicted, and `--save-temps=obj` is not object-neutral

Compile and link only — no GPU touched, no lock taken. Script: `/home/vermont/tmp/tile16-ab/build-snap.sh`
(HEAD snapshot via `git archive`, tree never patched; snapshot-only `instances.h` patch recorded in
`instances.patch.diff`). Both target lines move to `Bf16KvTiledMmaSchedule<64, 16>`:

```
-NINFER_BF16_KV_INSTANCE(Tiled64,  false, Bf16KvTiledMmaSchedule<64, 64>);
-NINFER_BF16_KV_INSTANCE(Tiled128, false, Bf16KvTiledMmaSchedule<128, 32>);
+NINFER_BF16_KV_INSTANCE(Tiled64,  false, Bf16KvTiledMmaSchedule<64, 16>);
+NINFER_BF16_KV_INSTANCE(Tiled128, false, Bf16KvTiledMmaSchedule<64, 16>);
```

Arms: `bin/ninfer-serve-base` 409,894,120 B sha1 `904bbc8c9017` (the tree's own `build-hip/apps/ninfer-serve`,
copied aside — the served 203 t/s point at 7042f064); `bin/ninfer-serve-cand` 409,818,584 B sha1
`c6bd4b871f04` (same link line, candidate `launch.cu.o` + `plan.cpp.o` placed **before** `libninfer_ops.a`).

**Gate A — plan passes, launch fails, exactly as §29's dated correction says.** Control compile of the
unpatched snapshot reproduces the tree's `plan.cpp.o` byte-for-byte (`00579a45e3` = tree `00579a45e3`);
`launch.cu.o` does not reproduce (control `ede30efe87` vs tree `b940466764`) because the stored object
predates the §27 probe edits and the recorded command carries `-lineinfo`. Byte equality is the wrong
gate for a file whose header inputs are dirty-in-tree by design; the metadata census is the gate.

**Gate C — PASS**: cand launch object `bb591b8858` ≠ control `ede30efe87`. The patch had an effect.

**Gate B — kernel metadata census, and the parser bug that hid its most important column.** The served
kernel's record from the census-base device object (six ⟨64,32⟩ mb1 variants, 62 kernels total):

| shape | VGPR | SGPR | private | **vgpr_spill** | LDS `(Br+2Bc)·512` |
|---|---|---|---|---|---|
| ⟨64,32⟩ mb1 (base, 6 variants) | 256 | 55 / 57 | 608 B | **19 / 20** | 65,536 |
| ⟨64,16⟩ mb1 (cand, 6 variants) | 221 / 222 | 35 / 37 | 528 B | **0** | 49,152 |

The candidate reproduces §28's recorded row exactly, including the spill drop 19 → 0 and the SGPR drop
55 → 35. `max_flat_scratch` and `amdhsa_group_segment_fixed_size` are absent/0 in both — dynamic-shared
kernels report 0, so LDS is computed, not read (the §28 method).

The first run printed `-` in the spill column for every kernel. Cause, recorded because the failure was
silent: `census.py` merged `llvm-readelf -n` output by anchoring on a `Kernel : NAME` line, **which this
ROCm's reader never prints**. Its AMDGPU metadata is YAML: one entry per kernel starting at `  - .args:`,
with `.vgpr_count:` / `.vgpr_spill_count:` / `.private_segment_fixed_size:` / `.name:` as sibling keys —
and the scalar keys come **before** `.name:` inside the entry, which is the same "anchor on the name,
read forward" trap recorded at lines 3088-3097: it attaches the *next* kernel's numbers to the current
one. Fixed by collecting each entry's keys, taking its name from within it, and merging after the entry
closes (`cens/{base,cand}/census.txt`, regenerated 2026-10-24). A census whose key column is empty
reads as "no spills recorded", not as "the census found nothing" — that is how a gate passes on a
column nobody looked at.

**Gate D — PASS, and decisive.** Schedule symbols present in each linked binary
(`nm -a | grep -o 'Bf16KvTiledMmaScheduleILi…E' | sort | uniq -c`):

```
base:  36 × ⟨64,32,1⟩   8 × ⟨64,64,1⟩   8 × ⟨128,32,1⟩   0 × ⟨64,16,1⟩
cand:  20 × ⟨64,16,1⟩   (and nothing else in this family)
```

So the link-order insertion did what §29's archive measurement said it should: the archive's six
same-named `launch.cu.o` members never won, the candidate's definitions won, the served symbols are
gone from the candidate, and no duplicate-definition error fired. The route is qualified — this is a
served-route A/B, not a probe build.

**Measured: `--save-temps=obj` is NOT object-neutral.** `censbase` `56099e72bb` ≠ `ctrl` `ede30efe87`,
and `censcand` `a37112c94e` ≠ linked `cand` `bb591b8858`. Measured the divergence on the base pair:
section *names and sizes* identical, **349,578 of 5,732,056 bytes** differ, spread over 115 of 1,400
16 KiB pages, and the census object is larger (5,753,104 B vs 5,732,056 B). So the census arm is a
sibling compile, not the linked byte-identical arm, and §29's instruction to gate on the census needs
that qualification: the census is admissible because the census-base compile reproduces served's
*recorded* metadata exactly (256 VGPR / 55-57 SGPR / 608 B private / 65,536 LDS / 19 spills — the
served row §28 and §24 obtained from the same `-save-temps=obj` pipeline, and now matched by an
independent arm), not because the census object is the object that was linked.

**The per-section proof, finished (2026-10-24, CPU only, no lock).** `objcopy --dump-section` is the
wrong tool here and that is why the first attempt printed nothing: it takes **the first section
matching a name**, and a HIP host object has hundreds of sections named `.group`. Bytes are read at
the `(fileoff, size)` records `objdump -h` prints instead, and sections are keyed by
`(name, size, sha1)` — `/home/vermont/tmp/tile16-ab/secdiff.py`.

Host objects, census vs plain (current artifacts in `/home/vermont/tmp/tile16-ab/obj/`; the compiles
were re-run since the byte-diff above, so absolute sizes moved — the **delta did not**: 21,048 B in
both runs):

```
base  ctrl 5,794,504 B → censbase 5,815,552 B   (516 sections in both)
cand  cand 5,633,040 B → censcand 5,654,088 B   (437 sections in both)
```

Section *count* is unchanged; the **only section whose size grows is `.hip_fatbin`**, by exactly
21,048 B in both arms (5,413,728 → 5,434,776 base; 5,354,592 → 5,375,640 cand). Host *content*
differences beyond the fatbin: the aggregate `.text` (11,970 B base / 11,634 B cand, same size, bytes
differ) in both arms, and in the **base arm only** six `configure_cuda_device_once` lambdas (553 B
each) and one libstdc++ `_M_rehash` helper (307 B).

Inside `.hip_fatbin` (magic `__CLANG_OFFLOAD_BUNDLE__`, unpacked with
`clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1102` — the type is `o`, `O` is
rejected):

```
device payload   ctrl 5,409,632 B → censbase 5,430,680 B   (+21,048, +1 section)
                 cand 5,350,496 B → censcand 5,371,544 B   (+21,048, +1 section)

.text   5,058,764 B  byte-identical in the base pair      (cand pair: 4,999,628 B, identical)
.note      98,972 B  byte-identical in the base pair      (cand pair: 98,964 B, identical)
.comment / .dynamic / .bss                               identical
.AMDGPU.csdata   0 B  present in the census arm, absent in the plain arm
.dynsym / .dynstr / .hash / .gnu.hash / .rodata / .relro_padding   same sizes, bytes differ
```

So the whole 21,048 B is **layout, not code**: the census arm's device object carries one extra empty
`.AMDGPU.csdata` section, whose insertion shifts addresses and forces the padding and link-edit
tables (`.dynsym`, `.dynstr`, `.hash`, `.gnu.hash`, `.relro_padding`, `.rodata`) to be rewritten. No
device section grows. The device **ISA bytes (`.text`) and the kernel metadata notes (`.note`) are
byte-identical between the plain compile and the `--save-temps=obj` compile, in both arms** — which
is a stronger admissibility argument for Gate B than the metadata-agreement argument alone: the
census arm is ISA-neutral on the device side, and it is the *host*-side perturbation (aggregate
`.text`, the `configure_cuda_device_once` lambdas) that keeps the census object from being the object
we link. Gate C's arms use the plain compiles.

**The ABBA measurement was launched after this build** (`abba-t16.sh`, 4 legs B-C-C-B in one flock
hold, ports 18971-18974, 1024-token lighthouse prompt, `max_tokens` 128, temperature 0; expected
**unfavourable**, §29's density prediction ~187 t/s vs served 203). Its result goes below when the
legs land: prefill t/s per leg, decode rate, completion-text sha1 against the `77f1779eb831` baseline,
pooled delta against the off-leg spread. Card state at launch: lock free, `/sys/class/kfd/kfd/proc/`
empty, no `ninfer-serve` running. When the set queued, another session took the lock and ran
`llama-bench` on the card (`-ngl 999 -fa 1`, therock build) — our set waits behind it, which is the
protocol, not an incident.

**Set t16, legs landed 00:28:45–00:33:20 — contaminated by foreign card occupancy, so the pooled
number is an artifact and is recorded as nothing.** Measured, in ABBA order:

```
leg      binary      TTFT    prefill t/s    decode t/s (server's last 3)   completion sha1
base1    904bbc8c90   20.8 s      203.1     8.4  / 12.8 / 12.9             77f1779eb831
cand1    c6bd4b871f   21.3 s      198.3     8.2  / 13.0 / 12.9             77f1779eb831
cand2    c6bd4b871f   21.3 s      198.4     8.8  / 11.8 /  7.1             77f1779eb831
base2    904bbc8c90   31.0 s      135.9     9.2  /  9.4 /  9.7             77f1779eb831
```

Coherence passes: all four legs print the baseline sha1 `77f1779eb831`, 574 chars, 93 words, 59 distinct
(63%), max 8-gram repeat 1 — the tile shape changes no output byte.

The set's statistics do not survive. Off-leg spread is 67.2 t/s in the **base** arm (203.1, 135.9) and
0.1 t/s in the cand arm (198.3, 198.4); pooled, cand 198.35 vs base 169.50 = **+28.85 t/s (+17.0%) on
the candidate**, i.e. the sign every static analysis in §28–§30 predicted to be the other way. That
sign is produced by the contamination, not by the tile shape.

**Cause, found after the fact.** At 00:35, ten minutes into our lock window's tail, `/proc/*/fd` and
`fuser /dev/dri/renderD129` show a foreign process on **our compute card**:

```
PID 1250943 (seen at 00:35, gone)  .../llama.cpp-prism-tune/build-therock/bin/test-backend-ops -o CPY -b ROCm0
PID 1290090 (started 00:35:13)     same binary, next op
/sys/class/drm/card1/device/gpu_busy_percent: 60% at 00:34:24, 27/28/28/12/6% over the next 5 s
```

It does not take `/tmp/ninfer-gpu.lock` — CLAUDE.md's rule that the user's own processes are unlocked,
here for the first time with a measurement it ruined. The same binary ran TTFT 20.8 s in leg 1 and 31.0 s
in leg 4. The degradation begins in cand2 (its decode falls 8.8 → 11.8 → **7.1** while its prefill was
already measured) and is full by base2. In ABBA order B-C-C-B, contention arriving in leg 3 and
deepening in leg 4 hits the *second base leg*, so it inflates the candidate by exactly the amount the
contamination takes from base2. The pooled delta is that mechanism; **discard it.**

**What does survive** is the uncontaminated prefix — the three legs whose TTFT is 20.8–21.3 s: base1
203.1 vs cand 198.3 / 198.4 (cand spread 0.1) = **−4.8 t/s (−2.4%) on the candidate**, same direction as
§29's prediction (unfavourable) at about **a third of its predicted magnitude** (−16 t/s predicted,
−4.8 measured). This is a lead, not a result: the base arm has one clean leg, and "a delta inside the
off-leg spread is not a result" cannot be evaluated against a contaminated spread.

**Re-run, and the protocol lesson written into the rules.** `abba-t16b.sh` re-runs the same two binaries
(arms unchanged, base `904bbc8c90`, cand `c6bd4b871f`) with two things set t16 lacked: (1) it **waits for
the card to be actually quiet before taking the lock** — no foreign opener of `/dev/dri/renderD129` and
`gpu_busy_percent ≤ 3` in every sample over a sustained 25 s window, aborting after 3 h rather than
measuring on a busy card; (2) it samples `gpu_busy_percent` plus the list of foreign render-node openers
every 2 s through the whole lock hold (`gpuwatch.py`, artifacts in `/home/vermont/tmp/tile16-ab/watch/`)
and prints per-leg windows, so admissibility is proved per leg instead of assumed. The rule this
measurement earns: **holding `/tmp/ninfer-gpu.lock` proves no other *lock user* is on the card; it proves
nothing about the user's own processes, and a leg is admissible only with a clean sampler window.**

**Set t16b, legs landed 00:46:06–00:49:34 — clean, and the tile shape is confirmed unfavourable.** The
quiet-gate held for the whole set: `card QUIET 00:45:19`, lock acquired the same second, released
00:49:48, and **zero foreign openers of `/dev/dri/renderD129` in every leg window** (sampler at 2 s,
129 samples, leg windows base1 1-23, cand1 24-67, cand2 68-95, base2 96-123; logs
`/home/vermont/tmp/tile16-ab/abba-t16b.log`, samples `watch/t16b-samples.tsv`). The busy percentages in
those windows (max 100%, means 57–80%) are **this set's own servers** — a prefill+decode leg is exactly
what a busy card looks like from `gpu_busy_percent`; the admissibility column is the opener list, and it
is empty in all four windows.

```
leg      binary      TTFT    prefill t/s    decode (server's last line)   completion sha1
base1    904bbc8c90   20.7 s      203.7     12.9                          77f1779eb831
cand1    c6bd4b871f   21.1 s      199.6      6.6  ← anomaly, see note     77f1779eb831
cand2    c6bd4b871f   21.2 s      198.6     12.9                          77f1779eb831
base2    904bbc8c90   20.8 s      202.8     12.9                          77f1779eb831
```

Prefill, pooled ABBA: cand 199.10 vs base 203.25 = **−4.15 t/s (−2.0%) on the candidate ⟨64,16⟩**.
Off-leg spread: base 0.9 t/s (203.7, 202.8), cand 1.0 t/s (199.6, 198.6); widest 1.0. The delta is four
times the widest off-leg spread, so this set **is** a result, in the direction §28–§30 predicted. Its
magnitude is a quarter of §29's predicted −16 t/s, and its sign matches the clean prefix t16 left
behind (−4.8). The contaminated +28.85 artifact is superseded: the clean set lands on the conclusion the
artifact had hidden.

Coherence: all four legs print the baseline sha1 `77f1779eb831`, 574 chars, 93 words, 59 distinct (63%),
max 8-gram repeat 1 — byte-identical to every leg of both sets.

**What this settles.** Gate B's census had the candidate *better* on every static column it prints: 0
spills against 19/20, 221 VGPR against 256, 528 B private against 608, 49,152 B LDS against 65,536. It
measured 4.15 t/s slower. That is the handoff's rule demonstrated on a second lever, after qfrag:
removing all spills and a third of the registers **did not translate**, because §29's executed-work term
is what moves the clock — ⟨64,16⟩ doubles the key-block steps per query tile, and the saved scratch
traffic is not paid back. The tile-shape lever stays closed, and now it is closed by measurement, not
prediction. Nothing here is committable as a perf change; `tile_io.cuh` and `tiled_mma.cuh` stay
uncommitted.

**Decode note, recorded as an anomaly, not a result.** cand1 decoded 2.4 / 7.0 / 12.8 / 6.6 tok/s and
its total was 40.3 s against 30.6–31.0 s in the other three legs, with prefill and TTFT in line
(199.6 t/s, 21.1 s) and no foreign opener in its window. One leg, cause unidentified, prefill
unaffected: not admissible as evidence about the candidate's decode rate, and not a reason to touch the
prefill verdict.

## §31 — The T2 GEMM prefill group, censused: 295 integer fragment-build ops per 16 WMMA, and the
### activation-scale stream is 16× redundant in issued bytes

Static only — one device-object census, no GPU, no lock, **no throughput claim anywhere in this
section**. Artifact: `/home/vermont/tmp/gemmcensus/` (device `.s`/`.o` from a `-save-temps=obj` compile
of `src/ops/linear/t2/t2_a8.cu` through its recorded `build-hip/compile_commands.json` command; the
`.s` has 73 kernels). Census script `/home/vermont/tmp/gemmcensus/census.py` (read-only, `python3 -I`).

**Which kernel.** The T2 A8 prefill GEMM is `t2_prefill_wmma1_kernel<1, 2, P, Epilogue>`, launched at
`src/ops/linear/t2/t2_a8.cu:393` as `<<<wgrid, 32, 0, stream>>>` — **32 threads, one wavefront per
block**, so every "per group" figure below is per warp. `P = wmma_panel()`, which defaults to 32
(`t2_a8.cu:92-98`, env `NINFER_T2_A8_WMMAPANEL=0` disables) and applies when `(w.n / 16) % 32 == 0`
(`t2_a8.cu:388`); the source comment at `t2_a8.cu:88-90` states the engine shapes satisfy it (gate_up
2176, mlp_down 320, gdn 768 tiles). Census target:
`_ZN6ninfer3ops6detail23t2_prefill_wmma1_kernelILi1ELi2ELi32ENS1_12_GLOBAL__N_118SplitStoreEpilogueEEE…`.

**All six instantiations are register-identical** (measured from the `llvm-readelf -n` YAML note):

| variant | VGPR | vgpr_spill | SGPR | private | max workgroup | LDS |
|---|---|---|---|---|---|---|
| P=32 Store / ResidualAdd / SplitStore | 123 | **8** | 40 / 40 / 45 | 36 B | 32 | 0 |
| P=0 Store / ResidualAdd / SplitStore | 123 | **8** | 40 / 40 / 45 | 36 B | 32 | 0 |

Paneling costs zero registers, as the shipping comment claims. The comment's *number* does not
reproduce: it says "128 VGPR, 36 B private segment" for the shipped kernel; this object reports
**123** VGPR with **8** spills (the 36 B private matches, and is consistent with an 8-VGPR spill).
Recorded as a docs-vs-object discrepancy, not as a claim about which number shipped.

**The steady group.** Two back edges close the same loop: `s_cbranch_execz` at body-index 643 → 123,
and an **unconditional `s_branch` at 651 → 123**. Range 123..651 = **524 instructions**; the
`s_cbranch`-only census (the pattern at `/home/vermont/tmp/isacensus/loopcount.py:30`) stops at 643 and
reports 517 — the 7-instruction gap is the second latch, not a counting choice. Whole kernel body:
1,119 lines / 1,033 instructions. Per group:

| class | count | | class | count |
|---|---|---|---|---|
| `v_wmma_i32_16x16x16_iu8` | **16** | | `v_perm_b32` | 32 |
| integer fragment-build, total | **295** | | `v_or3_b32` | 32 |
| — `v_and_b32_e32` + `v_and_or_b32` | 104 (72 + 32) | | `v_bfe_u32` | 24 |
| — `v_lshrrev_b32_e32` | 55 | | `global_load` | 24 (19 `b128`) |
| — `v_lshlrev_b32_e32` | 48 | | `scratch_load_b128` | 2 |
| | | | `v_cvt` / `v_fma` | 17 / **16** |

295 build ops to 16 WMMA = **18.4 integer shuffles per tensor-core instruction**, 56 % of the group.
This is the GEMM-side analogue of §25's attention finding (179.6 instructions per WMMA) on a much
smaller scale: the WMMA issue rate is not what this kernel is bounded by, statically.

**Issued vs address-unique bytes per warp-group.** 32 lanes × width:

| stream | issued | address-unique | ratio |
|---|---|---|---|
| activation codes (16 × `global_load_b128`) | 8,192 B | 4,096 B | **2.00×** |
| activation scales `xs` (the narrow stream) | 1,024 B | **64 B** | **16.0×** |
| weight codes (2 × `global_load_b128`) | 1,024 B | 512 B | 2.00× |
| weight scales | 64 B | 32 B | 2.00× |
| **global total** | **10,304 B** | **4,704 B** | **2.19×** |

Mechanism, from source. `t2_prefill_wmma.cuh:160-161` —

```
const int colWin = lane >> 1; // Awin window column within the warp tile
const int rowWin = lane & 15; // Bwin slot row within the warp tile
```

`lane >> 1` makes lanes `2i` and `2i+1` compute **identical addresses** (`pB = x + (col0 + colWin) *
k_cols`, `:181`), so each activation `int4v` load issues 32 lanes for 16 lane-distinct rows: 2.00×
issued/unique per instruction, 8,192 B issued for 4,096 B of codes the group actually reads. `lane & 15`
folds the weight side the same way in pairs (`pA[m]` at `:189`), so the weight stream carries the same
2.00×. The xs scale stream is the extreme case, from `:237-241`:

```
const int cb = col0 + n * 16 + (lane >= 16 ? 8 : 0);
for (int e = 0; e < 8; ++e) { xsv[n][e] = __half2float(xs[g * t_pad + cb + e]); }
```

`cb` takes **two values per warp** (`lane < 16` and `lane >= 16`), so each of the 16 hoisted scalar
scale fetches has 2 distinct addresses replicated over 16 lanes apiece; per group the stream issues
1,024 B to move the 32 scale halfs (64 B) the fold consumes — **16.0×**.

Correcting two earlier wordings on the way here: the loads are **not** "oversized per lane" — 16 B/lane
(`int4v`) is exactly one A fragment, and 32 B/lane (`int8v`) is exactly one weight group line — and the
redundancy is not activation-only.

**Load-site vs instruction reconciliation, stated as a bound.** The source has ~34 load sites per group
(16 `av[n]` `int4v` + 2 `line[m]` `int8v` + 16 xs scalar), the loop census sees 24 global-load
instructions (19 `b128` + `b64` + `b32` + 2 `u16` + `d16_b16`): the compiler merges and broadcasts some
of them, so the **per-stream** split of the 19 `b128` (16 activation-code + 2 weight-code + 1 xs) is an
attribution consistent with the width census and with the `colWin`/`rowWin`/`cb` address patterns above,
not an instruction-by-instruction trace. The issued-bytes total (10,304) is read directly off widths × 32
lanes; the 4,704 address-unique figure is computed from those address patterns.

Whether any of this costs time is **unmeasured**: 10 % of issued bytes is the xs stream, on a kernel
whose per-group cost is dominated by the 295 build ops, and §31 claims nothing about throughput.

**Provenance caveats.** (1) The census is of compiled-in defaults: the engine route (panel on,
`SplitStoreEpilogue`, `wgrid` from `t2_a8.cu:388-391`) was read off the launch site, not traced in a
serving process, so a shape that misses the divisibility runs the P=0 instantiation (register-identical
per the table above, group census not taken separately). (2) The env overrides
`NINFER_T2_A8_WMMANATIVE` / `_TILE` / `_WMMAPANEL` / `_COLSPLIT` / `_MIN` were **not** inspected inside
the serving process — this section describes the kernel the defaults select. (3) `NINFER_T2_I8_SMALL=off`
in the served A/B environment keeps the small-T route out of scope. (4) A per-section byte hash of
census vs linked objects is not established here (the §30 `objcopy` attempt was inconclusive).

## §32 — The packed 16-bit WMMA accumulator does **not** halve the accumulator: same 128 VGPR at the
### served tile count, so the native-PV wall of §26/§28 stands

Static only — a handful of compile-only probes and one metadata read. **No GPU, no lock, no throughput
claim in this section.** Artifacts: `/home/vermont/tmp/packedacc/probe13.cpp` (accepted call forms +
disassembly), `probe14.cpp` (the folded-away version, kept as the trap record), `probe15.cpp` +
`p15.dev.o` (the census), `p13.dev.o`.

**Why this was owed.** §25 established that the served attention mixer's PV contractions are
software-emulated (`mma_bf16` at `src/ops/common/mma.cuh:268` = 4 × `gather8` + 4 × `dot_bf8`; QK is
native), and §26/§28 closed the occupancy lever with the finding that the accumulator is the register
wall: 16 tiles × 8 fp32 accumulator values per lane = 128 VGPR/lane, the point at which the served
shape goes 1 WG/CU. The hypothesis carried forward from earlier sessions was: *"a packed 16-bit
accumulator cuts the PV accumulator to 64 VGPR/lane."* That hypothesis is **refuted**, and it is refuted
by the definition of the fragment, not by a compiler quirk.

**What the builtins actually look like on this target** (probe13, ROCm clang 20, `--offload-arch=gfx1102`):

```
__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(short v32 A, short v32 B, short v32 C, const bool)
__builtin_amdgcn_wmma_f16_16x16x16_f16_w32   (__fp16 v32 A, __fp16 v32 B, __fp16 v32 C, const bool)
```

Four arguments, the 4th a **compile-time constant bool** — a runtime `bool` is rejected ("must be a
constant integer"). A, B and C are 32-byte vectors; C is the accumulator, in place. HIP's
`__nv_bfloat16` cannot be a vector element type, so the packed form is reachable only through the
builtin spelled with `short` (bf16) or `__fp16` (f16).

**Emitted ISA** (`p13.dev.o`, `llvm-objdump -d --triple=amdgcn-amd-amdhsa`):

```
v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7],  v[8:15],  v[16:23] op_sel:[0,0,1]   // 4th arg = true
v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7],  v[8:15],  v[16:23]                  // 4th arg = false
v_wmma_f16_16x16x16_f16   v[16:23], v[0:7],  v[8:15],  v[16:23] op_sel:[0,0,1]
v_wmma_f16_16x16x16_f16   v[16:23], v[0:7],  v[8:15],  v[16:23]
```

The accumulator is an **in-place 8-register block** (`v[16:23]` in both the source and destination
slots), exactly as in the served integer form (§33's ISA line): 8 registers = 16 half-precision values
per lane per tile.

**The census that kills the hypothesis.** probe15 is a 16-tile chain with **per-tile distinct operands**
(`a[i]`, `b[i]`), all 16 accumulators live across a `__syncthreads()`, all stored — so no folding is
possible. probe14, the same shape with shared operands and a summed store, compiled down to nothing
useful and is kept only as the trap record: identical-operand WMMA chains get folded away, so a register
census taken on them measures nothing. Metadata from `llvm-readelf -n p15.dev.o`:

| kernel | VGPR | vgpr_spill | SGPR | private |
|---|---|---|---|---|
| `packed16` (16 × packed-accumulator builtin) | **128** | **16** | 24 | **68 B** |
| `fp32_16` (16 × served 3-arg fp32-accumulator builtin) | **128** | **32** | 32 | **132 B** |

The register counts are **identical**, and the reason is arithmetic, not scheduling: a wave32 16×16×16
D-fragment is 8 accumulator values per lane, and 8 fp32 values occupy exactly 8 registers; the packed
D-fragment is also 8 registers, holding **16** half-precision values. Packed doubles the accumulator
*values* per tile at the same register footprint — it does not halve the footprint. At the served tile
count (16 tiles/lane) both spellings land on 128 VGPR/lane, which is the wall §28 found and §26 closed.
The claim "a packed 16-bit accumulator cuts the PV accumulator to 64 VGPR/lane" is **retracted**.

What packed does do here, measured, is cut the **spill traffic** — 68 B private / 16 spills against
132 B / 32 spills for the fp32 chain at the same 128 VGPR — i.e. at equal register count the packed
chain sits under less pressure, consistent with carrying 16 live 16-bit values where the fp32 chain
carries 8 floats plus a conversion. That is a real margin and not a measured speed, and it is not a
lever at the served shape: 128 VGPR/lane is the wall whoever produces it, and every shape that drops
below 16 tiles/lane was measured (§28, §29) to cost more executed work than served. **Native PV stays
walled.** Nothing here was run on the card and nothing here predicts a throughput number.

## §33 — The §31 fragment-build stream is at its **ISA floor** on gfx1102: 4 perms per 16-byte fragment,
### and the bit-scatter instructions that would have compressed the selector are not implemented on this GPU

Static only — served-assembly reads and compile-only inline-asm probes. **No GPU, no lock, no throughput
claim in this section.** Artifacts: served device assembly
`/home/vermont/tmp/t2-census/base/t2_a8-hip-amdgcn-amd-amdhsa-gfx1102.s` (460 `v_perm_b32` in the TU),
probes `/home/vermont/tmp/perm8/probe20.cpp` and `/home/vermont/tmp/perm8/probe22.cu` plus the
single-instruction probe files generated from it.

**Operand widths, read off the served ISA.** The served WMMA sites are:

```
v_wmma_i32_16x16x16_iu8 v[8:15], v[30:33], v[16:19], v[0:7] neg_lo:[1,1,0] clamp
```

so per instruction **D = 8 registers, A = 4 registers (16 B/lane), B = 4 registers (16 B/lane), C = the
same 8 registers in place**. The builtin's sign arguments print as those modifiers: the served call
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, acc, true)`
(`src/ops/linear/t2/t2_prefill_wmma.cuh:227-228`, inside the m/n loops at `:224-230`) is
`neg_lo:[1,1,0] clamp` — the two `true` PSGN
arguments negate A and B, the 6th argument is the clamp. This closes §31's fragment-layout question from
the other side: the `int4v` (16 B/lane) activation load is exactly one A fragment and the `int8v`
(32 B/lane) weight line is exactly two B fragment rows, so the loads are correctly sized and the build
stream is pure byte-shuffling, not layout repair.

**Where the 295 build ops go, and why the perm half is a floor.** Per steady group there are 8
`wmma_detail::wmma_code_frag` calls (`t2_prefill_wmma.cuh:218-222`, the call on `:221`, run for
`s = 0..7` with `MF = 1`) serving the 16 WMMA, and each call is 4 × `spread4` + 4 ×
`__builtin_amdgcn_perm` (`:113-124`) = 32 `v_perm_b32` per group, matching §31's
table exactly. The four code windows per call read bits (0..7), (2..9), (16..23) and (18..25) of the
weight word — deliberately overlapping, because the ternary code stream packs 2 bits per weight.

`v_perm_b32` takes a 32-bit selector and produces **4 output bytes = 1 output register**, and there is
no b64 perm. A B-fragment is 4 registers = 16 output bytes, so a fragment costs **4 perms minimum**;
8 calls × 4 = 32 perms per group is exactly that minimum. The perm half of the build stream cannot go
below what served already spends.

**The remaining half — the selector build — was the lever, and the ISA closes it, not LLVM.** `spread4`
is `(w & 3) | ((w<<4) & 0x300) | ((w<<8) & 0x30000) | ((w<<12) & 0x3000000)`, which LLVM already fuses
into the `v_bfe_u32` / `v_and_or_b32` / `v_lshl_or_b32` forms in §31's histogram (~7 ops each, ~224 of
the 295). The plan was to replace it with the GCN bit-scatter pair — `v_bfmper_b32` to extract four
contiguous 2-bit fields, `v_bfmper3_b32` to expand them to byte stride — 2 ops for the same spread. That
plan is **dead**: probed by inline asm at `--offload-arch=gfx1102`:

| spelling | result on gfx1102 |
|---|---|
| `v_bfmper_b32` | **invalid instruction** |
| `v_bfmper3_b32` | **invalid instruction** |
| `v_permbate_b32` | **invalid instruction** |
| `v_pk_bf16_permbate_b32` | **invalid instruction** |
| `v_permlane16_swap_b32` | **instruction not supported on this GPU** |
| `v_bfe_u32`, register-only form | assembles |
| `v_perm_b32` with immediate selector | assembles (the compiler emits `v_perm_b32 v5, v5, v6, 0x6050400`) |
| immediate selector in **inline asm** | **invalid operand for instruction** |

Those last two rows are separate facts and both matter to anyone repeating this: the *compiler* emits
immediates freely (`v_bfe_u32 v1, v1, v4, 16`, `v_perm_b32 …, 0x6050400`), but AMDGPU inline asm parses
bare digits as **numbered registers**, so an immediate written in an asm string does not assemble — the
working spelling is all-register (`asm("v_perm_b32 $0, $1, $1" …)`). Separately, ROCm clang 20 exposes
**no** `__builtin_amdgcn_bfe_u32`, `__builtin_amdgcn_bfmper` or `__builtin_amdgcn_bfmper3` at all
(probe20's `spread4_bfe` / `spread4_bfmper` fail as undeclared identifiers); the `v_bfe_u32`
instructions in served code come from InstCombine recognizing a shl+and pair, not from a builtin.

**Consequence.** The served weight-side builder (`wmma_code_frag`, the register-table perm form shipped
in `7042f064`) is at the hardware floor on this GPU in **both** halves: 4 perms per fragment set by
instruction width, and ~7 ops per selector build because the 2-op bfmper-class forms do not exist on
gfx1102. §31's 18.4 integer shuffles per WMMA is therefore **not** reducible by instruction selection,
so a "fewer shuffles" candidate is not owed any GPU time. The bfmper selector-compression idea is
retracted. This says nothing about time — it says the op-count lever §31 opened is closed at the static
level.

**What this does not close** — where §31's linear lead still lives: the **register** side of the same
kernel, i.e. the diet that fits `(2,2)@8` into 64 VGPR with y-fastest panel geometry, which is the lever
the cal57/cal58 record above leaves open (`(2,2,8)` measured −4 % bit-exact at gate_up standalone at
44 B spill; every wider shape spill-bound at 432–1272 B scratch) — and §31's 16.0× `xs` activation-scale
stream. Both are register/traffic levers, not shuffle-count levers, and both stay behind the same gate:
qualify by compile census (spills 0, private ≈ 0, instruction count at or below served) before any GPU
leg, because §15 and cal59/cal61 both measured register relief and instruction reduction predicting
neither magnitude nor sign.

## §34 — The `(2,2)` register diet is **not needed**: plain `(2,2)` is 120 VGPR / 0 spills at the served
### `__launch_bounds__(32,8)`, in both a verbatim-copy probe TU and the served TU with the real epilogue; the LDS-diet and xs-de-hoist variants are refuted by census, and cal58's 44 B spill for `(2,2,8)` does not reproduce

Compile-only — device-metadata reads off `llvm-readelf -n` on `--save-temps=obj` gfx1102 objects, plus an
instruction census over the `.s`. **No GPU, no lock, no throughput claim in this section.** Artifacts:
`/home/vermont/tmp/t2diet/cal72.cu` (verbatim copies of the served body at `t2_prefill_wmma.cuh:143-278`
plus diet variants, probe epilogue `OutStore{ __half* }`), `/home/vermont/tmp/t2diet/cal73.cu` (the **real**
header `ops/linear/t2/t2_prefill_wmma.cuh` with the **real** served epilogue `SplitStoreEpilogue`, verbatim
from `src/ops/linear/t2/t2_a8.cu:238-257`), `/home/vermont/tmp/t2diet/census72.py`, and the save-temps
`.s`/`.o` beside them. Census parser notes: in this ROCm's `.s` each kernel body runs from its column-0
label `NAME: ; @NAME` to `.Lfunc_end`; LDS ops spell `ds_load_*`/`ds_store_*` (not `ds_read`/`ds_write`);
`__syncthreads()` in a 32-thread block elides entirely (0 `s_barrier` in the whole file) — the LDS variants
paid no barrier, they paid liveness.

**Metadata** (`VGPR / SGPR / private / LDS / spills`, MINB=8 as served):

| kernel | TU | VGPR | priv | LDS | spill |
|---|---|---|---|---|---|
| `<1,2,32>` control | cal72 probe | 123 | 36 B | 0 | 8 |
| `<2,2,32>` plain | cal72 probe | **120** | **0** | 0 | **0** |
| `<2,2,33>` y-fastest (col0/row0 swapped) | cal72 probe | 120 | 0 | 0 | 0 |
| `<2,2,32>` LDS per-m staging (`k_pmlds`) | cal72 probe | **240** | 0 | 8192 B | 0 (occ-4 warning) |
| `<2,2,32>` LDS staging, xs **un-hoisted** (`k_pmlds_unh`) | cal72 probe | **248** | 0 | 8192 B | 0 (occ-4 warning) |
| `<2,2,32>` LDS staging, half-warp copy (`k_pmlds_hw`) | cal72 probe | 128 | **424 B** | 4096 B | **108** |
| `<1,2,32>` control, real epilogue | cal73 served TU | 123 (SGPR 45) | 36 B | 0 | 8 |
| `<2,2,32>` real epilogue | cal73 served TU | **120** | **0** | 0 | **0** |
| `<2,2,0>` real epilogue (shipped raster) | cal73 served TU | 120 | 0 | 0 | 0 |
| `<1,2,0>` real epilogue | cal73 served TU | 123 | 36 B | 0 | 8 |

The cal72 control reproduces the served served-shape metadata (123 VGPR / 8 spills / 36 B private; SGPR
37 probe vs 45 served TU vs the 43 in the earlier census — SGPR is TU-context-dependent, VGPR/spill are not).

**Instruction census** (whole function, cal72 `.s`; per-tile = total / (MF·NF·16) tiles per block):

| shape | total | WMMA | perm | bfe | global_load | instr/tile |
|---|---|---|---|---|---|---|
| (1,2) panel32 (served shape) | 852 | 16 | 32 | 24 | 24 | **53.25** |
| (2,2) panel32 | 1455 | 32 | 64 | 48 | 24 | **45.47** |
| (2,2) panel33 (y-fastest) | 1455 | 32 | 64 | 48 | 24 | 45.47 |
| (2,2) LDS staging | 1593 | 32 | 64 | 48 | 32 | 49.78 |
| (2,2) LDS + xs un-hoisted | 1733 | 32 | 64 | 48 | 54 | 54.16 |
| (2,2) LDS half-warp copy | 1737 | 32 | 64 | 48 | 32 | 54.28 |

**Findings.**

1. **No diet is needed for `(2,2)` at the served launch bounds.** Plain `(2,2)` allocates 120 VGPR with
   zero spills and zero private memory at `__launch_bounds__(32,8)` (128-VGPR cap), in the probe TU **and**
   in the served TU with the real bf16 dual-destination epilogue. The cal57/cal58 premise — fit `(2,2)@8`
   into 64 VGPR (MINB=16) — is moot for the served MINB=8 launch.
2. **cal58's 44 B spill for `(2,2,8)` does not reproduce in served TU context, and my own hypothesis for
   it (epilogue-induced) is refuted.** cal73's `<2,2,32>` with `SplitStoreEpilogue` verbatim is 0 spills.
   Whatever produced cal58's 44 B is specific to the cal58 probe TU, not to the served kernel+epilogue.
   The (2,2) integration leg therefore runs at strictly better allocation than the 44 B-spill leg measured
   at gate_up standalone; cal58's −4 % time result stays valid for what it measured.
3. **Every diet variant is refuted** (rows preserved per the standing rule): LDS-staged per-m split raises
   VGPR to 240 and drops occupancy to 4 waves; removing the xs hoist on top of it is worse (248, and
   +278 instructions, `global_load` 24→54, `s_waitcnt` 32→114 — the hoist is what keeps the fold's scale
   stream out of the steady loop, matching cal22's finding); the half-warp copy meets occupancy 8 only by
   spilling 108 times / 424 B private (84 scratch ops in the body) — 13.5× the served spill count.
   Register pressure in this kernel is not spent where a diet assumed; staging spends it.
4. **y-fastest geometry is census-neutral**: swapping the `col0`/`row0` block-index expressions (probe
   `Panel==33`) changes neither registers (120/0/0) nor any instruction count (1455, 32 WMMA, 64 perms,
   48 bfe, 24 loads). In the **real** header the `Panel == 0` raster already carries a row/col swap in the
   runtime `rows_fast` flag (`t2_prefill_wmma.cuh:153-158`), and cal73 measures `<2,2,0>` at 120/0/0 —
   y-fastest for the shipped raster needs only host-side grid mapping. Panel-mode y-fastest
   (`Panel > 0`) needs the geometry branch in the kernel template.
5. **Gate status for the §33 law** (spills 0, private ≈ 0, instruction count at or below served):
   spills **0** ✓, private **0** ✓, instructions: 1.71× served **per launch**, **0.85× served per tile**
   (fragment-build cost — 4 perms + 6 bfe per fragment, 2 tiles per `wmma_code_frag` amortization —
   amortizes across the doubled M; the per-tile figure is the comparison the §23f bucket math consumes).

**Next leg (not run, gated):** GPU ABBA of `(2,2)` + y-fastest vs served `(1,2)` + x-panel, in a clean
sampler window under `flock /tmp/ninfer-gpu.lock`, per the cal58 traffic model (in-flight x set ≤ L2 ⇒
x-once ≈ 5.24 GB + weights-once 1.43 GB ⇒ DRAM ≈ 13.5 ms + request ≈ 13.1 ms vs the 29–30 ms measured
legs; a model prediction, **not** a measurement — §15/cal59/cal61 all showed register/traffic levers
predicting neither magnitude nor sign). Integration is host-side: template shape `(1,2,32)` → `(2,2,32)`
with grid recomputation at the launch site (`t2_a8.cu:393`), geometry branch for panel y-fastest, bit-exact
completion check against the served sha1 as always.

## §35 — The `(2,2)` + y-fastest leg is **measured parity**: 202.9 / 202.7 vs 203.5 t/s, inside the
### off-leg spread, bit-exact on all four legs. The −14.6 % per-tile instruction reduction did **not** translate
### ABBA run `mf2`, 2026-10-24, one `flock /tmp/ninfer-gpu.lock` hold, one binary, arms differ only in env

GPU measurement, favourable and unfavourable rows both preserved. Card: RX 7600M XT (gfx1102, 32 CUs),
`card1`, model `bonsai2_27b_text_swift.ninfer` (T2_G128_FP16). One binary
`build-hip/apps/ninfer-serve` sha1 `58efa02ba2`; the arms are **environment only**:

- **BASE** = served defaults: `t2_prefill_wmma1_kernel<1,2,32>`, x-panel raster (the 203 t/s operating point, §16).
- **CAND** = `NINFER_T2_A8_MF=2 NINFER_T2_A8_WMMAPANEL=0 NINFER_T2_A8_RASTER=rows` →
  `t2_prefill_wmma1_kernel<2,2,0>` with the `rows_fast` swap: 32×32 one-warp blocks, y-fastest.

Protocol (`/home/vermont/tmp/t2diet/abba-mf2.sh`): 4 legs B-C-C-B inside one lock hold (acquired 01:41:53,
released 01:48:27), 30 s settle after each server exit, `gpu_busy_percent` + VRAM sampled before server start
**and** before each request, `ROCR_VISIBLE_DEVICES=1`, `NINFER_T2_DECODE=simt NINFER_T2_I8_SMALL=off` (served
decode route), temperature 0, 1024-word lighthouse prompt → **prompt 4,215 tokens**, `max_tokens 128`.

| leg | arm | prefill tok/s | TTFT | decode tok/s | completion sha1 |
|---|---|---|---|---|---|
| base1 | base | 39.3 | 1 m 47.2 s | 13.0 | `77f1779eb831` |
| cand1 | cand | 202.9 | 20.8 s | 7.9 | `77f1779eb831` |
| cand2 | cand | 202.7 | 20.8 s | 12.9 | `77f1779eb831` |
| base2 | base | **203.5** | 20.7 s | 12.9 | `77f1779eb831` |

Coherence: all four legs produced the **same 574-character completion** — 93 words, 59 distinct (63 %),
most-frequent word `two` ×8, max 8-gram repeat 1, byte-identical to the served control text. The candidate
route is bit-exact end-to-end.

**Verdict: no result.** The usable comparison is `base2` 203.5 vs the candidate pair 202.9 / 202.7 (mean
202.8): **−0.35 %**, inside the off-leg spread the served record itself carries (203.4 / 202.5 = 0.9 t/s
≈ 0.45 %). Per the standing rule, a delta inside the off-leg spread is not a result. `base1` (39.3 t/s,
TTFT 1 m 47.2 s) is the cold first leg — first server start after idle; it is excluded from the throughput
comparison exactly as §16's pooled-first-leg caveat excludes it, and recorded here because the protocol
preserves every leg.

Confound recorded: a **CPU-only** compile of the test tree (12 jobs, no lock, no card access) ran
concurrently. It cannot perturb kernel time, but it loads CPU/DRAM during each leg's model-load phase;
the B-C-C-B layout and the identical 30 s settle absorb it, and both cand legs and base2 sit in the served
band, which bounds the effect.

**Findings.**

1. **The instruction-count lever is measured-inert for tile doubling.** §34 census: `(2,2)` is 45.47 instr/tile
   vs served 53.25 (**−14.6 %**) at 120 VGPR / 0 private / 0 spills, and 1.71× the instructions per launch
   with **half** the launches per tile. Measured effect on served prefill: **−0.35 % (inside spread)**. The
   §31→§16 conversion factor measured on the `v_perm_b32` change (−23.5 % instructions → +15.1 % speed,
   factor ≈ 0.64) would have predicted ≈ +9 % ≈ 221 t/s. It over-predicted by ≈ 9 points **in the wrong
   direction**. The §34 traffic model (DRAM ≈ 13.5 ms + request ≈ 13.1 ms vs the 29–30 ms measured legs)
   over-predicted by > 2×. This joins §15/cal59/cal61: static instruction counts and traffic models
   predict **neither magnitude nor sign** at this operating point.
2. **Launch-count halving per tile bought nothing either** — the served leg is not launch-overhead-bound,
   and at 203 t/s it is not instruction-issue-bound. Doubling one warp's tile work (2× WMMA, 2× fragment
   builds per block) is throughput-neutral at this occupancy, exactly as cal58's standalone gate_up −4 %
   at a 44 B-spill `(2,2,8)` leg already hinted.
3. **The `(2,2)` family is qualified as a route, not promoted as a lever.** Default (unset) keeps the served
   `(1,2,32)` geometry and grid exactly and measures 203.5 t/s with the served sha1 on the served route
   (base2); the knob's y-fastest arm is bit-exact on served shapes. The **panel** arm (`NINFER_T2_A8_MF=2`
   alone, `<2,2,32>`, 120/0/0) is **unmeasured** — queued, not claimed.
4. **What the target needs.** §23f: 300 t/s needs −26.8 % of total time; the linear bucket is 70.4 % of it.
   Two levers on that bucket are now measured inert (fragment-build stream at ISA floor §31/§33; tile-shape
   doubling §35). The remaining bucket-movers change **bytes moved**, not instructions issued: 4-bit
   activations (`iu4`) and the gate_up integer-activation route — both **user-gated**, neither measured,
   nothing claimed here.

**Next leg (queued, gated):** route qualification battery `ninfer_linear_t2_a8_test`
(`tests/ops/linear/test_t2_a8.cpp`: widths 1…1024 over `{n,k}` = `{1024,5120} {4096,5120} {7168,5120}
{34816,5120} {5120,6144} {5120,17408}` — `{34816,5120}` is the served gate_up shape, `{5120,17408}` is the
served mlp_down shape; Full/SampledRows vs the CPU reference) in three env legs — served defaults,
`MF=2` panel, `MF=2 WMMAPANEL=0 RASTER=rows` — one lock hold per test
(`/home/vermont/tmp/t2diet/qual-mf2.sh`). At those shapes the `MF=2` law
(`w.n % 32 == 0`, panel law `(w.n/32) % 32 == 0`) puts n = 1024/4096/34816/5120 on the panel path and
n = 7168 (`224 % 32 = 7`) on the `<2,2,0>` fallback, so both new code paths are covered. Then the
`MF=2` **panel** ABBA arm.
