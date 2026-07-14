# task245 — a1570a43 ("shift the red sprite back into the green box")

**Rule:** A 7×7 box is marked ONLY by its four green(3) corners at (brow,bcol),
(brow,bcol+6), (brow+6,bcol), (brow+6,bcol+6). A red(2) conway sprite belongs in
the box interior (rows brow+1..brow+5, cols bcol+1..bcol+5). In the INPUT the
sprite is translated OUT of the box by a uniform offset — either UP by k∈1..4
(roff=−k, coff=0) OR LEFT by k∈1..4 (roff=0, coff=−k); the generator guarantees
|coff|≤bcol+1 and |roff|≤brow+1 (sprite never wraps). The green corners are
IDENTICAL in input and output. Transform = shift the red pixels DOWN/RIGHT by
(dr,dc) (exactly one nonzero) back into the box; green fixed. Only colours
{0,2,3}; grid ≤10×10 (width,height∈7..10). Detection (byte-exact, 5000 fresh):
brow/bcol = min green row/col, rmin/cmin = min red row/col,
dr=max(0,brow+1−rmin), dc=max(0,bcol+1−cmin); output_red = red shifted by (dr,dc).
**Current:** 16.58 pts (custom:task245), mem 4420, params 125. Prior 14.72 (public gen:thbdh6332).
**Target tier:** B (variable-offset gather). The shift is an INPUT-DERIVED scalar
translation → not S (no fixed conv/permute window), not separable A (a 2-D
translate is row-gather∘col-gather of indices, not an AND of row/col conditions).
B (label-map + final Equal) is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 10×10 canvas; green/red bbox→scalar dr,dc; shift red via 2 Gathers + fp32 validity Muls; uint8 L=2red+3green sentinel-10; Pad 30×30; Equal | B | 6100 | 125 | 16.26 | 200/200 | working |
| 2 | combine shift validity + in-grid in BOOL (drop fp32 Muls/Adds) | B | 4920 | 125 | 16.47 | — | trim |
| 3 | cast red→uint8 before the 2 Gathers (gather planes 400→100B each) | B | **4420** | 125 | **16.58** | 200/200 + 500/500 bulk + valid edge cases | best |

## Best achieved
**16.58 @ mem 4420 params 125 — fresh 200/200, +500/500 bulk, valid edge cases
(max shift 4, single-px sprite, box at corner/center).** Beats prior 14.72 by
**+1.86**. Adopted? **N** (main adopts via `python -m src.adopt 245`).
Note: a hand-built input with |coff|>bcol+1 (sprite wrapped past the left edge)
"fails" — but the generator never produces it; all generator-valid outputs pass.

## Irreducible-floor analysis (after attempt 3)
At/near the Tier-B floor. mem_profile dominant tensors:
- **900 B uint8 Pad** (the 30×30 label feeding the FREE final Equal) — irreducible:
  the output spans 30×30 and Equal must write every cell; the sentinel-10 Pad is
  the only way off-grid cells become all-channel-0.
- **3 × 400 B fp32 channel slices** (ch0/bg, ch2/red, ch3/green at 10×10) — the
  cheapest gateway to per-cell colour: Slice preserves the input's fp32 dtype, and
  10×10 is the TRUE active region (grid ≤10×10, sprite stays inside the box inside
  the grid). bg is load-bearing (in-grid mask = bg∨red∨green; off-grid sentinel
  needs it); red/green feed ReduceMax min-index detection (ReduceMax rejects
  uint8/bool, so the planes must enter as fp32).
- Everything else is ≤100 B (uint8/bool 10×10 working planes, the two uint8
  shift-gathers, scalars).

## OPEN ANGLES (re-attack backlog)
- **Drop the bg fp32 slice (−400).** in-grid could come from ReduceMax(input,axis=1)
  but that's a 3600 B [1,1,30,30] plane — worse. A [1,10,10,10] crop = 4000 B —
  worse. No cheaper in-grid signal than the single bg channel slice found.
- **Fuse red detection + shift.** red is sliced fp32 (for ReduceMax) AND cast to
  uint8 (for the gathers); if the min-row/min-col could be read without a fp32
  ReduceMax (e.g. argmin over a uint8 ramp) the red fp32 slice (−400) could go —
  but ORT ReduceMin/Max reject uint8, blocking it.
- **Tier-S long-shot:** none — the translation distance is data-dependent, so no
  fixed Conv/permute realizes it; a variable shift is fundamentally a gather.

## INSIGHT (transferable)
⭐ **A data-dependent uniform TRANSLATION (shift sprite by an input-derived scalar
offset) is the same Tier-B pattern as a variable-offset crop (task091): recover the
offset as a SCALAR (here from two bbox min-indices: brow+1−rmin), then shift the
bit-plane with Gather(axis=2, clip(arange−dr)) ∘ Gather(axis=3, clip(arange−dc)),
zeroing out-of-range rows/cols with `Not(Less(arange−d,0))` validity masks
(opset-11 has no GreaterOrEqual).**
⭐ **Cast the plane being GATHERED to uint8 first** — Gather preserves dtype, so a
uint8 bit-plane makes each gather a 100 B tensor instead of 400 B fp32; combine the
validity masks in BOOL (And) rather than fp32 Mul to keep every post-gather plane
at 100 B. This trimmed 6100→4420 (+0.3 pts) with zero accuracy change.
⭐ A "corners-only" box is detected purely by min/max green index (a 1-D bbox), no
2-D corner plane needed; the box interior offset (+1) is a constant added to the
scalar.
