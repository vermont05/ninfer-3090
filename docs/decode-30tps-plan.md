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

## Phase 2 — production integration (if spike passes)

1. New kernel family `t2_gemv_wmma` in `src/ops/linear/t2/` (schedule table extended,
   row-per-warp → row-quad-per-wave), env `NINFER_T2_WMMA=1` for A/B.
2. Dispatch: route the four decode shapes (gate_up 34816x5120, down 5120x17408,
   attn 6144x5120, and the 34816-row MLP pair) through it; keep bits for anything it loses.
3. Numerics: ternary A entries are exact bf16 (±1/0); wmma f32-accum keeps every term exact —
   same per-term accuracy as the SIMT path, only summation order moves. Verify with the
   existing linear test vectors + a 3-prompt logit diff vs the bits build.
4. Full step probe + server t/s; goal gate ≥ 20 t/s end-to-end, stretch 25+.

## Phase 3 — fallback levers (if spike fails)

Order, cheapest first; each is a cal-bench before any repo change:
1. `v_perm_b32` (SGPR-selector byte permute) for 2-bit-field extraction → fewer ops/word than
   the shl/and/neg/xor mix; only try fields that table-map (this is the permt-table idea,
   now with the one permute instruction that exists).
2. Split partial: move dequant out of the dot loop for the 34816-row shape (batch-dequant a
   code row, reuse across... n/a for GEMV) — expected NO, verify cheaply.
3. Reduce steps' fixed cost: launch-bounds/occupancy already tuned (cal4); skip.
4. Accept ceiling: ship the 14.6→17 band improvements and re-baseline the target.

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
