# task174 — 72ca375d

**Rule:** The 10x10 grid holds exactly 3 monochrome "boxes" (creatures) in 3 distinct colours.
Box-0 (`colors[0]`, the `if not idx` box that is copied to the output) is constructed to be BOTH
horizontal(column)-mirror symmetric AND 180-rotationally symmetric; boxes 1 and 2 are constructed to
be NEITHER. The output is box-0 cropped tight to its bounding box, placed at the top-left corner of a
fresh grid (channel-0 fills the holes inside the HxW bbox; every cell outside the bbox is
all-channels-off). **Key invariant (verified 0/8000 + fresh 500/500):** box-0 is the UNIQUE present
colour (c!=0) whose bbox-cropped shape equals its own HORIZONTAL mirror — `hflip`-symmetry alone
identifies it (rot-symmetry alone also works; vflip alone does NOT).

**Current (prior):** 14.60 pts, mem 32787, params 81 (108-node CumSum/ArgMax-style net).
**Target tier:** B (variable crop + data-dependent translate-to-origin). Identification collapses to a
closed-form per-channel reflection MatMul (NOT a symmetry-search/flood); the crop+shift is the
task036 idiom, landing well below the detection floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reflect-MatMul hsym + task036 crop/shift, all fp32 | B | 26566 | 104 | 14.81 | 266/266 | works, <+0.3 |
| 2 | + cast active region to fp16, whole symmetry pipeline fp16 | B | 18480 | 104 | 15.17 | — | win |
| 3 | + overlap(A,Mf)==count(A) (1 plane) instead of diff+diff^2 (2 planes) | B | 16500 | 103 | **15.28** | 500/500 | win |

## Best achieved
15.28 @ mem 16500 params 103 — adopted? N (write-only). Beats prior 14.60 by **+0.68** (≥0.3 ✓).
GENERALIZES: isolated fresh 500/500 against freshly-generated instances; the hflip-symmetry identifier
is 0/8000 exact.

## Irreducible-floor analysis
Dominant intermediates (all over the 10x10 active region, [1,10,10,10]):
- `A32` 4000 B fp32 — the `Slice(input)` of the active region; Slice inherits the fp32 input dtype, so
  this single fp32 entry plane is the documented 3600B-style floor (cannot slice straight to fp16).
- four 2000 B fp16 planes: `A` (fp16 cast of A32), `Cmat` (per-channel reflection matrix), `Mf` (=A@Cmat),
  `AMf` (=A*Mf overlap). Each is a genuine full-region working plane; the reflection axis a=c0+c1 is
  per-channel so Cmat and the MatMul cannot be shared/shrunk, and Mf is the MatMul output. The
  overlap-count symmetry test already fuses what would otherwise be two planes (diff + diff^2) into one.
Everything else is tiny (1-D [1,10,1,1] scalars, the WORK=5 crop window, the 900B uint8 Pad output).

## OPEN ANGLES (re-attack backlog)
- Kill the 4000B fp32 `A32`: would need a Slice that emits fp16 directly (ORT has none) or a way to feed
  the MatMul without a named fp32 region — untried, likely structural.
- Collapse `Cmat`+`Mf` 4000B: a single fused per-channel column-reflection op (data-dependent reverse)
  would remove the explicit reflection matrix, but per-channel axes block a single Gather; no opset-11
  op does a per-batch data-dependent reversal cheaply.
- The reflection MatMul could in principle run on a per-channel-cropped 5x5 window if the box column
  bbox were gathered first, shrinking every plane ~4x — but the window position is itself per-channel
  data-dependent (circular: need the box to crop the box), so it cannot precede identification.

## 2026-06-19 re-probe vs LB P=15.77 (kojimar7113) — MARGINAL/INFEASIBLE for +0.3
Current LB adopted a crowd net `ext:kojimar7113` at **15.77** (mem 10067, dominant plane `colors4_1`
f32 [1,9,10,10]=3600B, TopK+ArgMax×5 selection). To beat by +0.3 → need pts ≥ **16.07** → mem+params ≤ **~7546**.

⭐ FOUND a SINGLE-PLANE escape from the 10-channel reflection MatMul (NEW, built + verified exact 266/266):
collapse the one-hot to a colour-index plane `colf` [1,1,10,10] and reflect THAT one plane per-pixel —
  AX = Gather(a_vec, colf)  (per-pixel reflection axis a_{colf[r,c]}, a_k = cmin_k+cmax_k)
  mc = AX − c ;  refl = GatherElements(colf, clip(mc), axis=3)   (column reflection on ONE plane)
  bad = fg ∧ (refl≠colf ∨ mc∉[0,9]) ;  box-0 = the unique present colour (c≠0) with ZERO bad cells.
This eliminates the four [1,10,10,10] MatMul planes (A/Cmat/Mf/AMf = 9000B); all reflection work is on
single-channel [1,1,10,10] planes. BUT it STILL cannot reach 16.07 — two irreducible rocks remain (measured):
- **fp32 entry A [1,10,10,10] = 4000B** — the Conv (→colf) AND the per-channel presence/bbox ReduceMaxes all
  need the 10-channel active region; Slice inherits fp32. Slicing the active region ONCE and deriving
  everything from it (4000B) beats three separate full-canvas reductions (6000B): restructure dropped
  mem 18279→16199.
- **per-channel "which colour is symmetric" reduce = 3000B** — mapping the single-plane `bad` result back to a
  [1,10,1,1] per-colour verdict needs `Equal(badcolf, arange_ch)` [1,10,10,10] bool (1000B) → Cast fp16
  (2000B) → ReduceMax. ORT ReduceMax/Sum/Conv/MaxPool ALL reject bool & uint8, so the fp16 10-ch plane is
  forced; no scalar route exists (two distinct bad colours can't be summed/deduped without per-channel).
Floor = A 4000 + reduce 3000 + L 900 = **7900B BEFORE any of the ~25 working planes** ⇒ best-case ≈15.96,
measured build **16199B/15.30**. 7900 > the 7546 needed for 16.07 ⇒ **+0.3 INFEASIBLE**. Single-plane is
~+0.0 vs the restored MatMul 15.28 and MARGINAL (below) vs kojimar 15.77.
- 1-D column-profile-palindrome proxy uniquely picks target 260/266 (6 fail); rot180 263/266; only exact
  2-D hflip(mask)==mask is 266/266 — no cheap 1-D collapse.
LEFT ON DISK: the proven MatMul net (15.28, mem 16500, 266/266) — cleaner and same score class as the
single-plane variant; neither beats the adopted kojimar 15.77 by +0.3. To beat 15.77 one must golf kojimar's
OWN 9-ch fp32 TopK/ArgMax selection plane (3600→≤900B uint8), but that source isn't editable here.

## INSIGHT (transferable)
⭐ "Odd-one-out by SYMMETRY among monochrome shapes" is NOT a symmetry-search wall: per-channel
horizontal-mirror symmetry is a closed-form **reflection MatMul** — reflect each channel's columns about
its own axis `a = c0 + c1` via `Cmat[k,c',c]=Equal(c'+c, a_k)` (task112/250 reflection-matrix idiom,
batched over the channel axis), then test equality of binary masks cheaply by `overlap(A,Mf)==count(A)`
(reflection preserves pixel count, so equal-count + full-overlap ⟺ identical mask — uses ONE extra full
plane `A*Mf` instead of two `diff`+`diff^2`). Once the box colour is a scalar, the variable crop +
translate-to-origin is the task036 Gather-shift + uint8 sentinel-Pad + final Equal finish.
Also: the canvas was a fixed 10x10 → slice to the active region and run every working plane at fp16
(whole-geometry fp16 works under ORT_DISABLE_ALL); the lone irreducible fp32 cost is the entry Slice.

## Safe-golf pass (S4, 2026-06-30)
Bit-identical dtype narrowing: the two Gather index intermediates (`cidx` [5], `ridx2` [5])
were produced by `Cast(to=int64)`; both feed ONLY `Gather` index inputs and hold grid
coordinates (≤30, fit int32). Narrowed both `Cast`→int32 (`to=6`) + matching value_info.
- **mem 7013 → 6973** (−40B), params 142 (unchanged), **pts 16.1244 → 16.1300 (+0.0056)**.
- Gate: bundled fail=0; equivalence vs incumbent = **0 divergences / 1596** random
  in-domain recolorings (layout preserved). Grader-safe (int32 Gather index, not *ND, not TopK).

## S11 (2026-07-03) — ADOPTED: Equal-then-Pad crossover + fp16 subtree recasts (+0.1891)
16.130 → 16.319 (7115B → 5889B; mem −1230, params +4). Two levers:
1. Crossover (playbook 7 addendum, −650B): 900B [1,1,30,30]u8 Pad-then-Equal carrier →
   Equal(Lin[1,1,5,5]u8, chan) → [1,10,5,5] bool (250B) → bool Pad to free output.
   Needed opset 11→13 (bool Pad illegal at 11; ReduceSum → axes-as-input form).
2. fp16 recasts (−580B): only clean fp16-capable subtrees (mismatch/nmis, box10 chain,
   is_box0_f) — integers ≤10. sig-chain skipped: Slice can't emit fp16, a bridging Cast
   costs +600B > 560B saved (⭐chain-bound variant of the recast trap). sig30 PRODUCER_BOUND.
Gates: bundled fail=0; fresh 2000 divergence 0 vs real incumbent; re-verified post-rebuild.
Backup: reports/retired_networks/task174_pre_s11_cross.onnx.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.041)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 4378 (mem 3982 + params 396), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
