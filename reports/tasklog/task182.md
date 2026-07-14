# task182 — 776ffc46

**Rule:** A 20x20 (canvas 30x30) grid holds 5-6 small sprites (each one of 10 hardcoded
shapes), placed non-overlapping with a >=1-cell gap. Sprite #0 is drawn in a "special"
colour (2 or 3) and is enclosed by a complete 7x7 gray(5) box outline; the default
generator guarantees a same-SHAPE duplicate (idxs[1]==idxs[0]). Every other sprite is
colour 1. OUTPUT: recolour EVERY colour-1 sprite whose shape EXACTLY matches the boxed
sprite's shape to the special colour; leave all other sprites (and the box) untouched.
The 4 curated train + 1 test examples add "fakeout" gray boxes (always partially
off-grid) and decoy special-coloured sprites, so the reference must be located by the
BOX, not by colour.

**Current (stored):** 14.23 pts, ext:biohack_new, mem 46548, params 1039.
**Target tier:** B (per-cell label rewrite via runtime cross-correlation). NOT a
detection/flood-fill wall: exact shape match is closed-form via two small convolutions
with a runtime kernel; not Tier S because the recolour is a non-local shape-correspondence.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | runtime 5x5 frame kernel, ref-by-COLOUR; window-count exact match | B | 35045 | 133 | 0.0 | 200/200 fresh | FAIL 3 curated (decoy special sprites + fakeout boxes) |
| 2 | locate reference by COMPLETE 7x7 gray box (perimeter conv ==24), gate ref to box interior; 7x7 ring-complement superset reject | B | 44113 | 253 | 14.30 | 267/267 | correct but barely beats stored (+0.07); fp32 throughout |
| 3 | fp16 all working planes downstream of the one fp32 colf entry | B | 27731 | 250 | 14.76 | 250/250 | +0.53 |
| 4 | uint8 label plane (900B vs 1800B fp16 for the 30x30 carrier) | B | 26832 | 250 | **14.79** | 250/250 | FINAL (+0.56) |

## Best achieved
**14.76-14.79 pts @ mem ~27k, params 250 — 267/267 stored, isolated fresh 250/250.**
Adopted? **N** (orchestrator gates). Beats stored 14.23 by **+0.53/+0.56 (Y, generalizes).**

## Irreducible-floor analysis
Dominant intermediates:
- **colf30 3600B fp32 [1,1,30,30]** — the 1x1 colour-index Conv entry. Any linear combo of
  the FREE fp32 input is fp32; the input-read floor.
- **colf32 1600B fp32 [1,1,20,20]** — active-grid slice (Cast source for the fp16 colf). One
  redundant plane; removing it saves ~0.03 pt only (still need a fp16 value plane downstream).
- **3 conv responses ~1352B each fp16 [1,1,26,26]** — corr_inner / corr_ring / paint. The 26x26
  size is forced: pad=F-1=6 is the MINIMUM that covers all 20x20 pattern-top-left anchors with a
  7x7 kernel (output 20+2*6-6 = 26). Can't shrink without shrinking the frame F.
- **L30 900B uint8 [1,1,30,30]** — value carrier feeding the free final Equal. uint8 is the
  cheapest dtype; sentinel-99 pad makes off-grid all-channels-off.

## OPEN ANGLES (re-attack backlog)
- Merge corr_inner + corr_ring into ONE 2-output-channel Conv (kernel [2,1,7,7]) — same byte
  total (2704 vs 2x1352) but fewer node intermediates; ~0 pt.
- Drop the fp16 `colf` and route all boolean comparisons off `colf32` (fp32 → bool, free), keeping
  only one fp16 value cast — saves ~800B (~0.03 pt).
- The 26x26 conv floor is the binding non-entry constraint; no smaller frame admits the
  5x5-max-bbox + 1-cell ring.

## INSIGHT (transferable)
⭐ EXACT SHAPE-MATCH ("recolour every sprite congruent to a reference shape") is NOT a
shape-correspondence/flood-fill wall — it collapses to a runtime cross-correlation: extract the
reference into a fixed-size runtime Conv kernel (ORT accepts a non-initializer Conv weight), then
a colour-1 sprite matches iff `Conv(occ, PF)==S` (all pattern pixels present) AND
`Conv(occ, ring)==0` where `ring = dilate(PF) - PF` (no extra pixel touches it → rejects supersets;
same-count distinct shapes are never subsets of each other, and connected supersets always have an
adjacent extra pixel). Paint the match back with `Conv(anchor, flip(PF))>0`. The reference kernel's
bbox-top-left comes from per-row/col presence + a ramp ReduceMin; place the pattern at frame offset
(1,1) so the ring fits inside the F=(maxbbox+2) frame. ⭐ When "fakeout" decoys share the marker
colour, locate the TRUE reference by a structural feature (here: the COMPLETE on-grid box —
perimeter Conv peak == perimeter count; partial/off-grid fakeouts never reach it) and gate the
reference plane to that region, rather than keying on colour.

## S11 (2026-07-03) — mech-15 finder scout: KILL — recolour targets are arbitrary sprite SHAPES matched by runtime cross-correlation vs a reference extracted into a Conv kernel; cost = conv-response planes + 1600B entry. Non-local congruence ∉ mech-15.

## 2026-07-03 solid-marker profile transfer probe — KILL

Follow-up from task008's `solid_marker_profile_reconstruction` win.  Hypothesis:
the complete gray 7x7 box locator might not need the shared 20x20 colour-index
plane.

Current graph already uses a single `cidf32 [1,1,20,20]` Conv output for three
load-bearing roles:

- locate the boxed reference via `AveragePool` row/column scores;
- provide the compact `cid uint8 [1,1,20,20]` label plane used to gather the
  5x5 reference template and center colour;
- produce the blue occupancy mask (`cid == 1`) for runtime template matching.

Therefore replacing only the gray-box locator with row/column profile
contractions does not delete `cidf32`; template extraction and blue-mask
construction still need the same 20x20 label carrier.  Also checked the obvious
`cid` cast-delay idea: removing the 400B uint8 `cid` forces `cflat/tv` through
fp32 and loses more than it saves.  No source candidate adopted.

Conclusion: task182 is not a task008-style solid-marker-profile transfer.  Its
floor is the shared colour-index entry plus runtime cross-correlation planes,
not marker localization.


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 6442 -> 6100 (+0.055)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].