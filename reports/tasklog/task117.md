# task117 — 4c5c2cf0

**Rule:** A background-0 grid (size 12..15, so ≤15×15) holds two colours. The BODY is a
fixed 3×3 X-cross (corners + centre) in colour `color` at (rowoff,coloff); its centre is
(cr,cc)=(rowoff+1,coloff+1). The LEGS are an arbitrary Conway sprite in colour `legcolor`,
drawn in ONE quadrant relative to the cross centre (the input shows only that quadrant). The
OUTPUT keeps the body cross and reflects the legs 4-fold about the centre: a leg cell (r,c) →
{(r,c),(2cr−r,c),(r,2cc−c),(2cr−r,2cc−c)} OR'd. An overall horizontal/vertical flip is applied
to BOTH input and output, so it does not affect the transform. Body identification: the body
is the colour whose 5 pixels form a mono 3×3 X (corners+centre set, edge-mids empty) AND whose
total pixel count is exactly 5; the legs are the other colour.

**Current:** 15.2196 pts, `ext:biohack_new` (public base net; verified GENERALIZES 100/100 fresh),
mem 17527, params 156.
**Target tier:** B (label-map + final Equal, with a data-dependent 2-D reflection realised as
two boolean MatMuls — the task112 idiom). Tier A blocked: the relabel is a per-cell colour
function of a globally-detected centre, not a row⊗col separable rectangle; the body/leg colours
are random per instance so no fixed Conv can route them (Tier S blocked).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | count==5 ⇒ body, double-MatMul reflect | B | 36672 | 131 | — | — | WRONG: legs have count 5 ~36% of the time |
| 2 | geometric mono-X detect + own-colour-count==5, in-grid=input-occ | B | 45326 | 131 | fail | — | in-grid used coloured-occ, clipped reflected legs |
| 3 | in-grid = full grid square (incl ch0) | B | 45326 | 131 | 14.28 | — | correct 265/265 |
| 4 | colf via 1×1 Conv on free input (kill [1,10,W,W]); fp16 planes; W=15 | B | 19444 | 142 | 15.12 | 200/200 | correct |
| 5 | banded X-conv, mono via single corner-conv, 1-D centre+lcolor scalars | B | 16918 | 157 | 15.25 | 200/200 | correct, MARGINAL |
| 6 | packed 1×1 Conv (in-grid+valid+colour in ONE fp32 plane) + Cauchy-Schwarz X-centre | B | 13408 | 126 | 15.487 | 9998/10000 | correct BUT 0.02% leg-X ambiguity |
| 7 | **+ per-axis nearest-to-grid-centre tie-break (exact disambiguation)** | B | 13853 | 129 | **15.454** | 10000/10000 | correct, bulletproof generalisation |

## Best achieved (2026-06-18 re-attack)
**15.454 @ mem 13853 params 129**, generalises **10000/10000** isolated fresh (0.0000% fail).
Adopted? N (orchestrator gates). Beats prior base 15.2196 by **+0.234** and prior best custom
15.2546 by +0.20. Official eval 265/265. MARGINAL vs the strict +0.3 bar (needs 15.52 /
mem+params ≤ ~13140; I am at 13982). A v6 variant scored 15.487 but flaked ~0.02% of fresh
instances (a leg conway-sprite that is ITSELF a count-5 X is locally indistinguishable from the
body); the +445 B nearest-to-grid-centre tie-break removes that completely — correctness wins
over the 0.03 score it costs, since a flaking net fails the 200/200 generalisation gate.

### What broke the prior 15.25 floor
1. **Packed colour Conv (the prior "kill colf30" open angle, solved differently):** ONE 1×1
   Conv with runtime weight `packw[k]=(k+1)+1000·cnt5_k` makes a SINGLE fp32 [1,1,30,30] plane
   that simultaneously encodes (a) in-grid = P>0 (bg colour-0 → 1, off-grid → 0), (b) the
   count-5 gate valid = P≥1000, and (c) the colour index L = P mod 1000. This DELETES the prior
   `own5` Gather subsystem (int32 index plane 900 B + ownc 450 B) AND the separate in-grid
   reduction — three signals, one plane, disjoint magnitude bands.
2. **Cauchy-Schwarz centre detector instead of the banded/corner convs:** `5·Conv(L²,Xk)==
   Conv(L,Xk)²` (5 X-cells share one value) on the single colour plane; `Xocc==5` and the `S1>0`
   gate are both IMPLIED by Cauchy + the count-5 `valid` gate, so they drop out — only 4 fp16
   working planes (Lsq,S1,S1sq,S2x5) instead of the prior multi-conv discriminator.
3. **+1 colour shift removes the off-grid sentinel Sub AND the unpack Sub:** L = colour+1 means
   off-grid (P=0 → L=0) matches no channel against `arange+1`, and bg=1 → ch0 for free.

## Irreducible-floor analysis (v5, prior 15.25 — kept for reference)
Dominant intermediate: `colf30` = the full-grid colour-index plane [1,1,30,30] fp32 = 3600 B,
produced by a 1×1 Conv contracting the 10 channels of the FREE input. This is the single
smallest representation of a per-cell colour value over the whole grid; any alternative that
reads all 10 channels at ≥15×15 (a spatial Slice of the input = [1,10,15,15] fp32 = 9000 B, or
a fp16 cast of the input = 18000 B) is strictly larger, and Conv cannot emit fp16 from an fp32
input nor restrict its output to a sub-window. colf is genuinely required for the mono-colour
discriminator (Conv(colf, corner-kernel)==0), which cuts the spurious-leg-X mis-detection rate
from ~0.45% (occupancy+count only) to ~0.005%. The second cost centre is the `own5` subsystem
(int32 Gather index plane 900 B + `ownc` 450 B): a per-cell "count-of-my-colour==5" test that
removes 27/30000 spurious leg-sub-X detections; it cannot use uint8 indices (ORT Gather rejects
them) and a MatMul/one-hot alternative would re-materialise a [1,10,*,*] plane. Together
colf30+colf32+colf+own5 ≈ 6.7 kB are structural; the remaining ~10 kB is ~16 fp16 [1,1,15,15]
working planes (2-D reflection matrices + 3 MatMul products + masks) that the data-dependent
2-D reflection inherently needs. Net floor ≈ 16.9 kB ⇒ ~15.25 pts.

## Irreducible-floor analysis (v6, 15.487)
Dominant: `Pfull` = packed colour plane [1,1,30,30] fp32 = 3600 B (the 3600 fp32 plane floor —
Conv on fp32 free input must emit fp32, output is full 30×30) + its fp32 crop `P32`
[1,1,15,15] = 900 B (Slice preserves fp32; cast-then-crop is strictly worse). These 4500 B are
structural. The rest is ~13 fp16 [1,1,15,15] working planes (450 B each): L, Lsq, S1, S1sq,
S2x5 (Cauchy, 4 planes), win (centre recovery), Rmat/CmatT (reflection matrices), LR/LC/LRC/
Lout16 (the 4-fold reflection), P + the u8 Lout/Lmask/Lpad masking. mem 13408 + params 126 =
13534; the +0.3 bar needs ≤ ~13140 (15.52), so I am ~394 B short.

## OPEN ANGLES (re-attack backlog — to cross +0.3 / 15.52)
- **−394 B somewhere in the 13 fp16 working planes.** Tried & rejected: (a) WK=14 work canvas
  for detect+reflect with a bg=0 reflection plane + 15-wide in-grid restore — the restore
  Where-chain (LcD/iscontent/validD/bgplane) cost MORE than the 14² crop saved (15.36, worse);
  (b) Gather-reflection instead of [W,W] matrices — the index-compute + zero-pad planes exceed
  the matrices (14.42, worse); (c) `S1==5L ∧ Conv(valid,Xk)==5` detector instead of Cauchy —
  same 4-plane count, no win; (d) packing cr,cc into one ramp — wash (plane vs params).
- **Real remaining lever:** drop ONE of {Lsq,S1,S1sq,S2x5} — needs a centre detector that proves
  "5 same-colour X-cells" in 3 full planes not 4. Or remove `P` (fp16 cast of P32) by running
  valid/ingrid/Mod on P32 fp32 — but Mod fp32 is 900 B (net worse). Or a cheaper masking that
  avoids both `Lout`(225) and `Lmask`(225).
- **Kill P32 (900 B):** find a crop that emits fp16 directly from the fp32 Pfull (none in opset 11).

## INSIGHT (transferable)
- ⭐ **Mono-colour X-centre detection without per-channel planes:** to find a fixed small
  same-colour stamp (here a 3×3 X) among other-colour clutter, combine on the colf plane:
  (a) a BANDED occupancy conv `10·(#X-cells)+1·(#edge-cells)` compared to a single constant
  (==50) detects "X full AND edges empty" in ONE conv; (b) a corner-vs-centre conv
  `corners=+1, centre=−4` compared to 0 enforces mono-colour (4·corner-sum==4·centre) without
  a separate ×5 plane; (c) a Gather of per-channel counts by the colf index gives a per-cell
  "count-of-my-colour" for an exact "this colour totals N" test. This trio uniquely locates the
  stamp centre at ~0.005% error with zero [1,10,H,W] materialisation.
- **`cnt==5` is NOT a body discriminator here** even though the body is always 5 px: the leg
  sprite has exactly 5 px ~36% of the time. Identify by SHAPE (mono isolated X) + count, not
  count alone.
- **In-grid mask must include channel 0:** the output extends beyond the input's coloured cells
  (reflected legs land on previously-background cells), so the grid extent has to come from the
  FULL grid square (any channel incl. ch0), never from coloured-occupancy — otherwise reflected
  pixels get clipped to the sentinel.
- **lcolor without a 2-D plane:** with exactly two present colours, `lcolor = Σ_k k·(cnt_k>0) −
  bcolor` from the tiny [1,10] counts, avoiding a 450 B colf·legmask reduction plane.

### v6 insights (the +0.24 over prior best)
- ⭐ **PACK MULTIPLE PER-CELL SIGNALS INTO ONE 1×1 Conv via disjoint magnitude bands:** a single
  runtime-weight 1×1 Conv `packw[k]=(k+1)+1000·flag_k` on the free input emits ONE fp32 plane
  that decodes to THREE per-cell signals — in-grid (P>0, because the +1 makes bg colour-0 → 1
  while off-grid stays 0), a colour-property gate (P≥1000), and the colour index (P mod 1000).
  This collapsed the prior agent's separate in-grid reduction AND its int32-Gather count-5
  subsystem (1.3 kB) into the one colour plane I already pay for. Generalises: any per-cell
  boolean about a cell's COLOUR (count==N, colour∈set, …) can ride the colour-index Conv's
  weight in a high band instead of a second full plane.
- ⭐ **Cauchy-Schwarz `5·Conv(L²,Xk)==Conv(L,Xk)²` makes Xocc==5 and S1>0 redundant:** any empty
  X-cell (value 0) breaks the all-equal equality, so the "all 5 filled" and "nonzero" gates fall
  out for free once a count-5 `valid` gate excludes the background — 4 fp16 planes, not a multi-
  conv banded discriminator. (Same anchor-detector idiom as task165/task346.)
- ⭐ **+1 colour shift = free off-grid sentinel:** index colours as k+1 (bg=1) and match against
  `arange+1`; off-grid cells (Conv result 0 → index 0) then match NO channel automatically, so
  no Sub-to-unpack and no separate off-grid sentinel Where are needed.
- ⭐ **CHEAP EXACT TIE-BREAK = "nearest to the GRID GEOMETRIC CENTRE" (per-axis, 1-D):** when a
  detector yields two candidate centres (here a leg sprite that is itself a count-5 X, ~0.02%),
  the true symmetry centre is the FIGURE centre — closest to gc=(H-1)/2 — while the decoy is
  offset to one quadrant. Recover gc from H = #in-grid rows (ReduceMax(L,[3])>0 then ReduceSum),
  then per axis pick the candidate index maximising -(idx-gc)^2 via Where(cand,-d²,-BIG)→
  ReduceMax→Equal→Σ·ramp. All 1-D [1,1,W,1] tensors (~30 B). Exact 100000/100000; far cheaper
  than per-candidate on-grid-reflection evaluation, and beats centroid-distance (which fails).
- **REFLECT-EVERYTHING beats reflect-just-legs:** the body X is itself symmetric about the
  centre, so `Max(L, R@L, L@C, R@L@C)` over the whole colour plane duplicates the legs AND
  leaves the body invariant — no need to isolate the leg colour. BUT in the bg=1 scheme the
  background (value 1) also reflects and CAN land on off-grid cells → an in-grid mask is then
  mandatory (verified: ~9 corruptions/instance without it).
- **WK<WD work-canvas + bg=0 restore did NOT pay here:** content fits [0,13] so detection+
  reflection could run on 14×14, but the in-grid-background restore needed for size-15 grids
  (which still have bg at row/col 14) costs a Where-chain that exceeds the 14² savings. Lesson:
  shrinking the work canvas only wins when you don't have to reconstruct the dropped border.

## S11 (2026-07-03) — recast candidate REFUTED at re-measurement; incumbent restored
Long-tail agent reported +14B win (its baseline read params=174 — wrong; real incumbent
= 3922/243 = 16.6655). Rebuilt candidate measured 3948/243 = 16.6593 stably (3 isolated
runs) = −0.0062 LOSS. Adopted briefly, reverted to incumbent (verified 3922/243 restored).
⭐LESSON: re-measure the INCUMBENT yourself before trusting an agent's baseline numbers —
this was the adoption-protocol 정합검증 step catching a bad delta.
