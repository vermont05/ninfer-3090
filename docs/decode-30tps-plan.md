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

## Phase 4 — the rest of the budget (do not start before Phase 2 lands)

gdn_mixer non-GEMV is ~20 % of its 18.3 ms; attn_mixer 9.9 ms includes softmax/flash and KV.
lm_head (2.7 ms) is the 248320-row word GEMV — same Phase-1 kernel family applies.
Sequence: re-measure buckets after each merged change; do not tune on old buckets.

## Verification protocol (every step)

- Server: the NINFER_T2_DECODE=simt / NINFER_T2_I8_SMALL=off launch recipe + ROCR_VISIBLE_DEVICES=1
  + Tensile path (see docs/rocm-port-analysis.md; env landmines in memory).
- Bench harness: `/tmp/gemv_cal7.cu` patterns; run idle GPU (check `pgrep -x ninfer-serve`
  is empty AND lock-free before trusting a number).
- ctest area tests (build-hip has BUILD_TESTING=OFF → use step probe + server output).

## Do-Not

- Re-run cal2-6 sweeps (family exhausted, memory `gemv-word-path-plateau`).
- Touch prefill (explicitly deferred by user).
- Trust any run with the default T2 dispatch env (4.3x trap).
