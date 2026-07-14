# task163 — 6d0160f0

**Rule:** 11x11 "hollywood_squares": a 3x3 arrangement of 3x3 mini-blocks separated by single
gray(5) lines at row/col index 3 and 7 (line ⇔ r%4==3 or c%4==3). Block (R,C) occupies rows
R*4..R*4+2, cols C*4..C*4+2; cells filled with random colours, exactly ONE is yellow(4) at block
(R,C), mini-position (mr,mc). OUTPUT = the same blank gray-square grid EXCEPT the 3x3 contents of
input block (R,C) are stamped into the output block at position (mr,mc) (identical intra-block
offsets). Verified exact against the generator (500 fresh + 267 stored, 0 mismatch).
**Current:** was 14.92 pts (public). Now 15.98 pts, mem 8046, params 187.
**Target tier:** B (separable block copy as a double boolean MatMul + colour-index label-map).
Not S: the source block position (R,C) and target position (mr,mc) are data-dependent globals
recovered from the unique yellow pixel — no fixed Conv/permute window emits the copy. The copy
DOES factor row⊗col (block copy preserves intra-block row/col), so a double MatMul over selection
matrices realises it without a full [1,10,H,W] product; but the colours are arbitrary input colours
(not a fixed set), so a colour-index plane (1x1 [0..9] Conv) is required → Tier B, not A.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-idx Conv→slice 11x11→fp16; yellow pos via ch4 slice; Rmat@V@CmatT (fp16 sel-matrices, off+inrange); +gray bg (fp32); uint8 Pad; Equal | B | 8651 | 195 | 15.91 | 200/200 | works |
| 2 | yellow from Equal(V16,4) (drop ch4 slice); label arithmetic in fp16 (drop two fp32 11x11 planes) | B | **8046** | **187** | **15.98** | 500/500 | FINAL |

## Best achieved
**15.98 pts @ mem 8046, params 187 — 267/267 stored, fresh 500/500 (under ORT_DISABLE_ALL).**
Beats prior 14.92 by **+1.06**. Adopted? **N** (main adopts via `python -m src.adopt 163`).

## Irreducible-floor analysis
Two intermediates dominate and are both documented floors:
- **3600 B fp32 [1,1,30,30] Conv (`Vfull`)** — the colour-index gateway `V=Σ k·input_k`. The
  harness input is fixed [1,10,30,30]; a 1×1 Conv preserves spatial extent and follows the fp32
  input dtype, so the entry colour-index plane is 3600 B. Every alternative is worse: slicing the
  10-ch input to [1,10,11,11] first = 4840 B; casting input→fp16 = 18000 B; the copy reads
  arbitrary colours so the colour index is genuinely needed. (Same floor as task035.)
- **900 B uint8 [1,1,30,30] Pad (`L`)** — the 30×30 label feeding the FREE final Equal; off-grid
  cells get sentinel 99 so they are all-channel-0. Pad must be uint8 (ORT Pad rejects bool); doing
  Equal at 11×11 then padding the bool output is impossible (bool Pad rejected). Irreducible.
- Everything else ≤ 484 B: V/V16/yel/copied/sel-matrices on the small 11×11 canvas, already fp16
  where ORT allows; aggregate ~3.5 KB and hard to push much below.
mem+params = 8233 → score 25−ln(8233) ≈ 15.98. This sits at the Tier-B arbitrary-colour-copy
floor (cf. task035 16.20 on a 10×10 canvas; the 11×11 canvas costs the extra ~0.2).

## OPEN ANGLES (re-attack backlog)
- **Eliminate the 3600 B colour-index Conv.** Would require the copy to route the 10-ch one-hot
  straight into the FREE bool output (Tier S). Blocked: the double-MatMul placement needs a float
  operand, and MatMul on the full [1,10,30,30] one-hot = 18000 B; slicing first = 4840 B > 3600.
  No cheaper per-cell 10→1 reduction exists (the documented FLOOR_RESEARCH result).
- **Shrink the small-plane aggregate (~3.5 KB).** Merge yellow-position recovery and the
  selection-matrix build to share fewer ramps; marginal (~0.05–0.1 pt), not pursued.

## INSIGHT (transferable)
⭐ A "stamp one data-dependent sub-block into a data-dependent block slot on a FIXED separator-grid
background" is a Tier-B separable block copy, NOT a connectivity/correspondence wall. Recover the
selector from the unique marker pixel as scalars (mr=yr%4, source-block-top=yr−mr, target-block-top
=mr*4, constant offset off=src−tgt), then realise the placement with the task250 double boolean
MatMul where each selection matrix is `Equal(s, o+off) AND (mt ≤ o < mt+3)` — the per-axis offset
+inrange gate copies a fixed-height run to a data-dependent slot with NO Scatter/NonZero. The
fixed gray-square background is a free additive fp16 initializer (copied content never lands on a
line cell, so no overlap). Costs only the one 3600 B colour-index Conv (arbitrary colours) + 900 B
Pad above the small 11×11 working planes.

## S8 (2026-07-02) — rect-recipe conversion ADOPTED, div 0
two scalar einsums 'bchw,c,h->' locate the single yellow; yellow11 plane dropped; 2345→1797, +0.266. Fresh: agent uncached 2500 div0 + my uncached 400 div0.
