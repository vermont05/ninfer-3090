# Decode 30 t/s plan (gfx1102 eGPU, swift bonsai2 t2)

Status: DRAFT 2026-10-21. Owner: overnight loop. All numbers below are measured on this box,
not borrowed from the 3090 profile.

## Budget (measured)

| item | now | 30 t/s needs |
|---|---|---|
| step time | 68.5 ms (14.6 t/s) | 33.3 ms |
| gdn_mlp | 29.7 ms | ~14 ms |
| gdn_mixer | 18.3 ms | ~9 ms |
| attn_mlp | 9.9 ms | ~5 ms |
| attn_mixer | 9.9 ms | ~5 ms |
| lm_head | 2.7 ms | ~1.3 ms |

Envelope: the 7.2 GB of weights stream once per token; 254 GB/s memcpy-class ⇒ ~28 ms ⇒
~35 t/s ceiling. 30 t/s is ~86 % of that envelope, so it needs GEMV *and* everything around
it at load-class efficiency. cal7 already showed the load side is cheap (staging floor
169 µs vs 344 µs shipped on gate_up), and partial compute (dequant + dot) is the whole gap.

## Facts this plan stands on (calibrated, this box)

- word-GEMV partial is issue-bound: bits ≈ 90-139 instr/word, cndmask worse; per-activation
  dequant is ~8 VALU and irreducible in the cndmask/bit-trick class (cal3-8 exhausted it).
- cal7 T=16 gate_up: floor 169 µs, +x-loads 197 µs, bits 331 µs → dequant/dot is ~50 % of time.
- **NEW: `v_wmma_f32_16x16x16_bf16` assembles AND executes correctly on gfx1102**
  (llvm-mc + runtime probe), and in a peak harness it lands at ~35 TFLOP/s vs ~4.6 TFLOP/s
  for the same-harness VALU fp32 FMA loop — ~8x the VALU rate. This is RDNA3's matrix path.
- ISA survey (llvm-mc, gfx1102): OK `v_wmma_f32_16x16x16_bf16`, `v_perm_b32`,
  `v_bfe_u32`, `v_alignbit_b32`, `v_lshl_or_b32`, `v_cndmask_b32_e64` (src3 = SGPR only);
  REJECTED: `v_mfma_*`, `v_wmma_i32_*_i8` (byte/DP4A-class wmma), `v_prmt`, `v_byteperm`,
  `v_permlane_swap/shift`, `v_dot_i32_u8_u8_i32`, `v_gteac_u32`, `v_bfmask_b32_b64`.

## Phase 1 — WMMA GEMV spike (`/tmp/gemv_cal10.cu`, standalone, no repo touch)

Goal: for T=1, can a wave-wide `v_wmma_f32_16x16x16_bf16` beat `t2_gemv_word_partial_bits`
on gate_up (34816 x 5120)?

Design:
- A-operand = weight tile: stage one 16-k group's codes in LDS, dequant 4 codes/lane-slot
  straight into the 4-double (`v4double`) A fragment (the probe's operand form), sign via
  the bit-trick — no float round-trip, no LDS scatter.
- B-operand = x broadcast: x[k] for the 16-k group is wave-uniform per k; build the fragment
  once per k-group, reuse across m-rows if the fragment layout is column-uniform.
- C accumulates across k-groups; final group-scope cross-lane fold via ds_permute/ds_bpermute
  into the row-sum.
- First sub-step: dump the fragment layout — compile the probe kernel with
  `llvm-objdump -d --mcpu=gfx1102` on a variant that writes per-lane A/B/C offsets and read
  LLVM's wmma lowering (AMDGPU trackOperandRegUnits layout) so the spike places elements
  correctly, not optimistically.

Gate (measure against cal7's bits number, same harness):
- GO  if ≥ 1.5x bits on gate_up with correct sums → Phase 2.
- NO-GO → Phase 3 fallback.

### Phase 1 result (2026-10-20): NO-GO

Classic rocwmma (`#include <rocwmma/rocwmma.hpp>`, therock headers + therock clang,
single `--offload-arch=gfx1102`) compiles and **computes correctly** on gfx1102:
16x16 gemm matches the CPU reference (cal10 step-1, cal10i end-to-end probe).

But the whole classic-API load path is ~10x too slow for GEMV. cal10/cal10g on
34816x5120 (same shape as the cal7 bits bar):

| variant | time | vs 331 µs bar |
|---|---|---|
| cal10 (stack tmp store) | 3.85 ms | 0.09x |
| launch sweep best (cal10b, bs=128 nb=64) | 3.46 ms | 0.10x |
| cal10g (shuffle extract, unrolled k) | 1.97 ms | 0.17x |

`load_matrix_sync` on gfx11 is not a vectorized per-lane load: it costs far more than a
dwordx4 per lane and the k-loop can't amortize it. Fragment layout (measured via
accessors, cal10j — authoritative): for both A (row-major) and B (col-major) tag,
lane L holds ONE contiguous 8-half chunk: lanes 0..15 = cols 0..7 of rows 0..15,
lanes 16..31 = cols 8..15 of the same rows; C: lane L<16 = row L even cols
{0,2,..14}, lane 16..31 = row (L-16) odd cols. An LDS-staged loader could close some
of the gap but plausibly not the 6x needed to beat the bar, let alone the 1.5x gate
(≤220 µs). No MFMA on gfx1102 (ISA rejects it), and the builtin-asm path can't be
filled by hand because fragment storage layout ≠ accessor view (raw dword pokes
disagree with operator[]). **WMMA line closed; Phase 3 is the way.**

## Phase 2 — production integration (if spike passes)

1. New kernel family `t2_gemv_wmma` in `src/ops/linear/t2/` (schedule table extended,
   row-per-warp → row-quad-per-wave), env `NINFER_T2_WMMA=1` for A/B.
2. Dispatch: route the four decode shapes (gate_up 34816x5120, down 5120x17408,
   attn 6144x5120, and the 34816-row MLP pair) through it; keep bits for anything it loses.
3. Numerics: ternary A entries are exact bf16 (±1/0); wmma f32-accum keeps every term exact —
   same per-term accuracy as the SIMT path, only summation order moves. Verify with the
   existing linear test vectors + a 3-prompt logit diff vs the bits build.
4. Full step probe + server t/s; goal gate ≥ 20 t/s end-to-end, stretch 25+.

## Phase 3 — integer-dot DP4A path (result 2026-10-21: WINNER)

Phase 1 closed WMMA; the fallback lever list is superseded by a bigger find —
**hardware byte-dot works on gfx1102 after all**:

- `__builtin_amdgcn_sudot4(true, a, true, b, c, false)` assembles to `v_dot4_i32_iu8`
  and executes correctly (plain `sdot4` is rejected: needs `dot1-insts`).
- `__builtin_amdgcn_perm` (=`v_perm_b32`) compiles and executes; with a 4-entry
  symbol table it expands one code byte → four sign bytes in 2-3 VALU + 1 perm.
- Prism precedent: llama.cpp-prism ships `build-rocm-dp4a` for gfx1102 with
  `-DGGML_HIP_PTQ1_DP4A=1`; q2_0 symbols use the same perm trick (vecdotq.cuh).

Calibration (`/tmp/gemv_cal11.cu` + `/tmp/gemv_cal12.cu`, idle GPU, cal7 harness):

| shape | floor | bits (current bar) | dp4a-int8 best | speedup vs bits |
|---|---|---|---|---|
| gate_up 34816x5120 | 171 | 346-365 (T16/T32) | 246.7 (T16) | 1.40x |
| down 5120x17408 | 58.6 | 182 (T16) | 95.4 (T32) | 1.91x |
| attn 6144x5120 | 45.8 | 70 (T32) | 39.1 (T32) | 1.79x |

Correctness: `/tmp/cal11c.cu` — sym4+dp4a vs scalar reference over 8192 random
words (x8 16 codes, t2_g128 table 0→0,1→+1,2→0,3→−1): 0/8192 mismatch. The
16-code dot probe matched too (`/tmp/cal11v.cu`).

Semantics note: `v_perm_b32`'s selector is a 4-bit-per-byte source-byte index
(16 = byte 0 of src1, etc. — probed via src0=0x33../src1=0x77.. ladder); my earlier
identity-model of it was wrong, which is why the standalone table probe looked
broken while the composed dot is right. Trust the end-to-end probe.

Next: production kernel family on this route (int8 x quantized per 128-elem group,
scale per group like weight scales; quant kernel once per GEMV call, not per row),
A/B vs current route on the four decode shapes, FP32-oracle + logit-diff qualification,
then make it the dispatch default (env-gated divergence is the 4.3x trap).

**SHIPPED (2026-10-21, commit 3f13fb32):** `src/ops/linear/t2/t2_gemv_dp4a.{cuh,cu}`
plus the `t2_dispatch` hook (env `NINFER_T2_DP4A=off` to fall back). Server A/B on
the eGPU, same 256-token decode, same request:

| leg | step | wall |
|---|---|---|
| bits (NINFER_T2_DP4A=off) | 70.26 ms (gdn_mlp 29.6, attn_mlp 9.90) | 24.4 s |
| dp4a default | 56.65 ms (gdn_mlp 21.6, attn_mlp 7.06) | 17.8 s |

1.24x step, 1.37x wall — steady-state decode ~14.2 -> ~17.5 t/s. The first
integration silently vanished because the dispatch hook was guarded `__HIPCC__`
in a host TU (no `-x hip`); guards are `NINFER_BACKEND_HIP` now (the convention
CMakeLists.txt:55 documents).



Qualification data (2026-10-21, this tick): cal13q proves the dp4a kernel matches the
float64 reference for the *quantized pipeline* (bad=0 at k=5120/17408), and the single
sample /tmp/cal13q harness covers the kernel. Production end-to-end, the route is an
int8-activation approximation of the exact-BF16 bits route, and the difference is not
cosmetic: a 10-prompt greedy suite (temperature 0, 64 tokens, deterministic server --
verified by running the bits leg twice with identical output) gives 2/10 fully
identical outputs and 7/10 first divergence at char <= 30 (the first generated token
flips). Whether that tolerance is acceptable is a product call -- Prism ships the same
int8-activation design; this build now matches it. The plan's "FP32/FP64 oracle +
logit-diff qualification" gate should be closed on a coherent corpus before the route
ships beyond this machine.

## Phase 4 — the rest of the budget (do not start before Phase 2 lands)
### Phase 4 fresh buckets (dp4a build, 2026-10-21, per step)

gdn_mixer=17.16  gdn_mlp=21.34  attn_mixer=9.31  attn_mlp=6.97  (head 1.79)

gdn-mixer internal (NINFER_STEP_PROBE_FINE=1, per sampled layer, ~45 gdn layers):
  proj=0.229  out=0.075  rec=0.035  nc=0.023  norm=0.021   (sum 0.383/layer)

Lever (located 2026-10-21; corrected the same day after the A/B below): the gdn
input projection ("proj", 0.229/layer — the biggest gdn-mixer sub-cost) was thought
to sit on the Q8_G32 single-parent route (`q8_gdn_input_dispatch` ->
`launch_q8_a16_gemv`, k = 2048, A16-activation GEMV + fused conv epilogue), and a
byte-dot version was implemented there. It does not: swift bonsai's gdn input
projection is the **T2 two-parent route** (`t2_two_parent_project` over
{4096,5120} + {12288,5120}; conversion manifest bf16/fp32/t2_g128_fp16, and
parameters.cpp registers integer_route_pair for exactly those shapes), which
already passes the `t2_dispatch` dp4a gate at T = 1. The Q8 single-parent route
serves a different profile (qwen3.5-9b class) and never dispatches for this model
— verified by symbol presence (nm/strings) plus the 2026-10-21 server A/B: dp4a
24.72 s vs NINFER_Q8_DP4A=off 24.84 s wall, with per-bucket and fine-probe numbers
identical between legs. So the null A/B was real, not a stale binary.

Calibration 2026-10-21 (`/tmp/gemv_cal14.cu`, idle eGPU, shape n=12288 x k=2048):
the a16-SIMT bar clone (with the production `warp_reduce_sum` fold — an early fork
that dropped it mis-measured 160 us and failed its own reference) runs 59.1 us;
floor 20.5 us; dp4a-direct (int8 x, sym-free: Q8 codes go straight into
`v_dot4_i32_iu8`) 39.7 us, dp4a-with-LDS-staging 49.5 us. Both probe kernels verify
bad=0/64 against f64 pipeline mirrors. The 1.49x-over-bar result stands for the
Q8_G32 shape class, but is inert for swift bonsai (route not on its path). The
remaining gdn-proj lever for this model is inside the T2 pair path — the pair
GEMVs are already byte-dot, so the candidates there are the separate
conv/snapshot kernel behind the GEMVs and the small-T band, not a new Q8 route.
(Caveat from review: the fine "proj" bucket covers more than the isolated GEMV,
so a standalone kernel ratio cannot be multiplied across layers.)



gdn_mixer non-GEMV is ~20 % of its 18.3 ms; attn_mixer 9.9 ms includes softmax/flash and KV.
lm_head (2.7 ms) is the 248320-row word GEMV — same Phase-1 kernel family applies.
Sequence: re-measure buckets after each merged change; do not tune on old buckets.

Calibration 2026-10-21 (`/tmp/gemv_cal15.cu`): the cal14 direct-vs-staged question asked
of the T2 shapes — shipped staged partial (cal12 mode 2 + production fold) as control vs
direct-global-load variants (1 row/warp) and 2/4-rows-per-wavefront (shared x loads), all
bit-identical to the control's reference (the cal12-mode-2 original never stores; the
fork adds the production warp fold + lane-0 store so the comparison is real). NO-GO:
control stays within noise of the best direct variant on every shape (gate_up 258 vs
247, down 103 vs 102, attn 44 vs 44, value_z 84 vs 83 us; inter-run noise ~4-8 %). The T2
stage/lane mapping already sits at issue+load balance; de-staging is not a lever here.
Next T2-side candidate from the pair-path survey: the A16 two-parent route (the swift
bonsai decode path, band off) calls the GEMV three times per gdn layer and quantizes x
once per call (three quantizes of the same vector) plus two strided-assembly copies;
folding to one quantize + three GEMVs into the destination planes is a launch/overhead
cut of maybe ~10-20 us/layer, not a kernel-shape change.

SHIPPED (2026-10-21): the fold is in the pair wrapper (t2_gemv_dp4a_prepare +
run_prepared; the wrapper's width-1 branch quantizes once and lands the three GEMVs
at destination offsets — the T=1 qkv vector swallows both legs without strided copies).
Step probe after, same harness/recipe: proj 0.229 -> 0.191 ms/layer, gdn_mixer
17.16 -> ~16.2, step total 60.6 -> 58.8 ms (~+3 %); decode probe 16.4 -> 16.9 t/s.

CUDA graphs (2026-10-21): the server's graph path works on this HIP build (boot
"CUDA graphs ready | 173 ms") and is worth ~+0.4 t/s — server legs measured 17.3-17.4
decode with graphs on vs 16.9-17.0 with --no-cuda-graph, and the outputs are
byte-identical between the two legs (one 64-token temperature-0 request per leg,
content+reasoning compared; /tmp/graph_ab.sh). So the default launch (no flag) is
the faster canonical path for throughput runs. Measurement caveat: the step probe
mis-accounts under graph replay (its events sit inside captured graphs — the
graph-run tail printed garbage buckets: steps=5, layers=77.53 ms, fine n=1); keep
--no-cuda-graph for any bucket-level probing, graphs-on for wall/throughput.

## Phase 5 — next decode levers (external review 2026-10-21, supersedes the Phase-4 tail)

Source: a second Claude session read the plan + kernel and cross-checked the Oct-15 Prism
tuning session (`~/.claude/.claude-config ... /memory/prism-tuning-history.md`, transcript
`eb57dc99-…`). Decisions accepted from that review; the Prism **runtime stays unrun** (standing
rule — it is read for design only).

- **The GEMV gap is a schedule problem, not a code-unpacking problem.** Prism's PQ2_0 matvec on
  this card hits ~268 GB/s (~93 % of the 288 GB/s class) with the *same* perm-codec family
  (its `q2_0_symbols4_hip` ≈ `t2_gemv_sym4`); NInfer's dp4a gate_up runs ~63 % (247 µs vs 171 floor,
  cal11). Prism's win was the schedule: 4 rows per thread reusing activation loads, 2 warps,
  int8 activations in blocks of 32 with one scale each. So the earlier "~30 ALU ops vs 4 dp4a
  per word — attack the unpack" idea is **closed as a lever**; the schedule is what to move.
  cal15's 2/4-rows NO-GO does not cover this: it kept NInfer's per-word scale handling and
  compared against the staged kernel, not the mmvq schedule.
- **Next cal candidate (cal16): port the mmvq schedule** (rows=4, nwarps=2) as a standalone on
  the NInfer shapes, standalone-harness first, against the shipped dp4a kernel's cal numbers.
  If it lands near Prism's 93 %, the ~35 ms of GEMV in the 58.8 ms step shrinks a third →
  ~19.5-20+ t/s territory. Design read from Prism is allowed; A/B is always NInfer-vs-NInfer.
- **attn_mixer (9.31 ms/step) has never been broken down.** ~0.58 ms/layer vs gdn mixer's 0.34,
  while its two GEMVs are ~0.1 ms — ~0.45 ms/layer unexplained. Suspect: the same `mma.cuh`
  shuffle emulation that prefill uses (and whose `bcast_pair_t` bug was fixed 2026-10-21 —
  all Phase-4 buckets are **pre-fix evidence**, re-measure before quoting).
- **Trace before choosing.** One `rocprofv3 --kernel-trace --stats` over a few decode steps
  (GPU idle, lock held) gives per-kernel time + inter-launch gaps and sizes the fusions
  (norm+quantize, swiglu+quantize). Prism's whole tuning ladder came from exactly this.
- Prism's ladder for reference (19.77 → 23.65 t/s): matvec schedule retune +9 %, graphs off
  −2 %, FWHT-writes-int8-input (no separate quantize), DeltaNet in-place state read, gate/up
  fusion, RMS-norm fused into FWHT. The transferable ones for NInfer: quantize fused into the
  norm/rotation kernel, and the in-place state read idea.

Suggested order: kernel trace → cal16 (mmvq schedule) → attn_mixer breakdown → fusions.

## Verification protocol (every step)

- Server: the NINFER_T2_DECODE=simt / NINFER_T2_I8_SMALL=off launch recipe + ROCR_VISIBLE_DEVICES=1
  + Tensile path (see docs/rocm-port-analysis.md; env landmines in memory). Add
  `--no-cuda-graph` for step-probe/fine-probe runs (the probe mis-accounts under replay;
  see the CUDA-graphs note in Phase 4), leave graphs on for wall-clock/throughput legs.
- Bench harness: `/tmp/gemv_cal7.cu` patterns; run idle GPU (check `pgrep -x ninfer-serve`
  is empty AND lock-free before trusting a number).
- ctest area tests (build-hip has BUILD_TESTING=OFF → use step probe + server output).

## Do-Not

- Re-run cal2-6 sweeps (family exhausted, memory `gemv-word-path-plateau`).
- Touch prefill (explicitly deferred by user).
- Trust any run with the default T2 dispatch env (4.3x trap).

## Correction + RDNA3-native roadmap (2026-10-21, review against Prism)

**ISA correction:** the survey above lists `v_wmma_i32_*_i8` as rejected. The RDNA3 instruction is
`v_wmma_i32_16x16x16_iu8`, which does not appear in that survey; a wrong mnemonic is the likeliest
reason. Prism's prefill matmul uses it on this card
(`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, `~/src/llama.cpp-prism-tune/ggml/src/ggml-cuda/mma.cuh`,
RDNA3 branch) and reaches 17.8 TOPS. The Phase 1 NO-GO was rocwmma's `load_matrix_sync`, not WMMA:
llama.cpp's `mma.cuh` calls the builtins directly with documented RDNA3 register layouts. Copy those.

**Layout pinned (verified 2026-10-21, `.qual/wmma/cal16{c,e,f}.cu`):** the read units are whole
16-byte per-lane tuples; byte probes (cal16c) show every slot is atomic, and dual-sparse probes
(cal16e) pin the pairing exactly: `C(lane l, word w) = dot(Awin[(l<16?0:17)+2w], Bwin[l])`, byte-identical
pairing. Loader: Bwin slot l = A row l (dup at l+16); Awin window 2c (c<8) / 17+2(c−8) = B column c
(byte k = B[k][c]); product lands at `(A·B)(l&15, (l>=16?8:0)+w)`. Battery of 10 random/identity
cases passes (cal16f). Instruction-bound ceiling: ~58.4 TOPS (cal16 peak, 64 blocks × 32 warps ×
16 chains) — 3.3× Prism's end-to-end 17.8 TOPS, so schedule, not instruction support, is the gap.

**Prism decode reference (same card, same model, 2026-10-15):** the PQ2_0 matvec runs at ~268 GB/s
(93 % of 288) with the *same* perm codec as `t2_gemv_sym4`. The win was the schedule: 4 rows per
thread reusing activation loads, 2 waves per block. NInfer's dp4a gate_up is ~63 % (247 vs 171 us).
History: memory `prism-tuning-history`.

Where to drop the CUDA shape, ranked by estimated win:

1. **Prefill:** replace the shuffle emulation of `mma.sync`/`ldmatrix` (`src/ops/common/mma.cuh`)
   with native iu8 WMMA tiles (ternary -> int8 via perm, int8 activations). ~20 -> 200-300 tok/s.
2. **Decode GEMV:** Prism's mmvq schedule (4 rows/thread, x in registers, 2 waves, scale once per
   group). ~10 ms/token, ~17.5 -> ~23 t/s.
3. **int8 + per-128-scale as the inter-kernel activation format:** norm rotates and quantizes once;
   every GEMV sharing that input (qkv, z, gate/up) reads it. 2-4 ms.
4. **Fixed-shape fusions only a single-model engine can do:** DeltaNet prologue (conv + l2norm +
   alpha/beta + gate), norm+rotate+quantize, swiglu+quantize; stretch: one persistent kernel per
   layer. 3-6 ms.
5. **Small-T verify on 16-wide WMMA tiles:** verifying 2-16 tokens costs about one; enables n-gram
   copy drafting without a draft model (Prism + ngram-simple: whole-file edit 22 -> 61 t/s).
6. **Attention decode:** unprofiled, 0.58 ms/layer (more than a whole GDN mixer); RDNA3 split-K kernel.
7. **No cp.async on RDNA3:** hide latency with occupancy; 128 KB LDS/WGP and a 32 MB Infinity Cache
   (GDN state 3.1 MB/layer fits) should shape items 1-2.

Before any of it: one `rocprofv3 --kernel-trace --stats` decode run to confirm the GEMV share.
The "Touch prefill" Do-Not above is superseded: prefill is the current thrust (`remaining-work.md` §3).

## Phase 5 landing — mmvq decode GEMV (2026-10-21, cal17)

cal17 (`.qual/gemv/cal17.cu`) ported the mmvq schedule onto the T2 RowSplit layout and it
wins on every registered shape at bit-identical output: each warp owns R consecutive rows
and strides the FULL group range (one 128-group per step: `(lane>>3)` picks the group slot,
`(lane&7)` the slice), so one activation load feeds R code chains. The earlier cal16-era
probe mis-partitioned groups per warp (each warp got a disjoint subset, no cross-warp
combine) — that shape is invalid; the fixed schedule matches the shipped staged control
bit-for-bit (cal17 battery, 8 shapes x 8 variants, all `ok`).

Standalone (cal17, idle card, best of 9x5, vs staged control): m2w2 (2 rows/thread, 2-warp
CTAs) is the uniform winner — gate_up 185.6 vs 251.9 us (1.36x), xl 671.9 vs 951.3 (1.41x),
xxl 1280.5 vs 1847.4 (1.44x), attn_small 14.1 vs 34.2 (2.6x); R=4 always loses to R=2. It is
now the shipped T=1 dp4a route for all 12 kDp4aShapes (t2_gemv_dp4a.cu; the staged row-quad
stays in the header, unrouted).

Shipped-path measurement (same recipe as the Phase 4 fold note):
- step probe (--no-cuda-graph): step total **58.8 -> 54.4 ms** (gdn_mixer 16.6, gdn_mlp 20.6,
  attn_mixer 9.2, attn_mlp 6.7; head/embed flat), steady over 320 steps;
- decode: no-graph **16.9-17.0 -> 18.2 t/s**; graphs-on **17.3-17.4 -> 18.7 t/s** (two 64-token
  temperature-0 legs, identical content, valid output).
The GEMV share of the step moved from ~63 % of the memcpy-class floor toward Prism's ~93 %
class on the big shapes; the next biggest unexplained bucket is still attn_mixer (9.2 ms).
