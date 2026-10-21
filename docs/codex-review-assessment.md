# Codex review assessment (2026-10-21)

The review (delivered during the Phase 2/3 window) asked for four things. Status
against the shipped tree, in the reviewer's own ordering:

1. **Pursue integer-dot GEMV instead of scalar-FP polish.**
   Done, and it is the shipped decode route: `t2_gemv_dp4a.{cuh,cu}` +
   `t2_dispatch` hook (commit 3f13fb32). cal12 bars 1.40-1.91x over the bits word
   partial; server step probe 70.26 -> 56.65 ms/step, wall 24.4 -> 17.8 s on the same
   256-token decode. The scalar-FP line (Phase 1 WMMA spike was NO-GO anyway) is
   dead; this is the route.

2. **Make the winning route the backend default; env-dependent dispatch is a
   4.3x trap.**
   Done: `t2_dispatch` enters the dp4a route by default for every registered decode
   shape; `NINFER_T2_DP4A=off` is only the escape hatch, not the switch that turns
   the route on. (The 4.3x trap itself lives one level up: the MLP decode dispatch
   needs the `NINFER_T2_DECODE=simt` launch recipe; that env is a launch-recipe
   requirement, not part of the dp4a gate.)

3. **Qualify numerics with FP32/FP64 oracle + logit diff.**
   Partially done, and the honest split is: the *kernel* is qualified (cal13q: the
   dp4a pipeline reproduces the float64 reference exactly at k=5120 and k=17408, and
   the sym4 table is exact over 8192 words), but the *pipeline* is an approximation
   of the bits route because the activation is requantized to int8 per 128-group.
   End-to-end that is visible: on a 10-prompt greedy suite the dp4a build agrees
   with the bits build on only 2/10 outputs (the rest diverge at the first token;
   server determinism verified by a repeated control run). Whether that tolerance is
   acceptable is a call for the user -- Prism ships the same int8-activation design
   -- but the gate should be closed on a coherent corpus + real logit deltas before
   this leaves the machine. (The local swift checkpoint's chat path is incoherent
   on all routes, so output text cannot adjudicate quality here.)

4. **Don't treat 23 t/s as the matched baseline.**
   Agreed and applied: the bar is 30 t/s. Current dp4a steady-state is ~17.5 t/s on
   the 256-token probe (was ~14.2 on the same build tree with the route off), so
   Phase 3 bought ~23% and Phase 4 (gdn q8 decode proj byte-dot, see plan) is the
   next identified lever, with gdn_mlp (21.3 ms/step) and gdn_mixer (17.2) the biggest
   remaining buckets.
