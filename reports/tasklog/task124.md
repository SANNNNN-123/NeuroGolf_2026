# task124 — 53b68214

**Rule:** Input is an H×10 grid (H 5..8) holding the TOP of a vertically (and optionally
diagonally) periodic sprite tiling. A sprite of height `tall` (1..3) repeats vertically with
period `tall`; if it repeats diagonally it also shifts right by `shift=(wide-1)*diag` (0..2) every
period. The 10×10 OUTPUT extends the same pattern over all 10 rows:
`out(r,c) = P[r%tall][c - shift*(r//tall)]` (0 if source col OOB), P = first `tall` input rows.
**Current:** 15.71 pts (public CumSum/OneHot net), mem 10520, params 353
**Target tier:** B (closed-form periodic extension; pure index Gather of a recovered value plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 9-cand 2D-mismatch detect + 10x10 value plane + Pad30 fp32 carrier | B | 23604 | 141 | 14.92 | 267/267 | working but heavy |
| 2 | 3-cand (leftcol-derived shift) + fp16 + uint8 carrier + padded-gather | B | 12834→10334 | ~120 | 15.32→15.75 | ok | iterating down |
| 3 | 8-row crop + 9-ch slice entry + per-cand scalar mismatch | B | 9540 | 122 | 15.82 | 267 | closing in |
| 4 | **1-D occupancy-bitmask consistency** `bm[r]==(bm[r-t]·2^s) mod 1024` | B | 7694 | 119 | **16.04** | 200/200 | ADOPTED |

## Best achieved
16.04 @ mem 7694 params 119 — beats prior 15.71 by **+0.33**. Fresh 200/200 (and 3000/3000 stress).

## Irreducible-floor analysis
Dominant intermediate is the 9-channel input crop `inHK [1,9,8,10]` fp32 = 2880B — the colour-index
entry (channels 1..9 only, ch0 weight 0; cropped to the 8×10 active canvas, cheaper than a 30×30
Conv plane + slice). Next is the uint8 30×30 carrier (900B) feeding the FREE one-hot `Equal`. The
whole (tall,shift) DETECTION is 1-D (≤[8] vectors) and essentially free — the key win.

## OPEN ANGLES (re-attack backlog)
- The 2880B `inHK` slice is the floor; a single-channel occupancy/colour plane still needs either a
  30×30 Conv output (3600) or a multi-channel spatial crop (2880). No escape found.
- The diagonal output col-shift machinery (SCi 400 + Ap 380 + GatherElements) only matters when
  shift>0 (~32% of instances); could special-case shift==0 to a pure axis-2 gather, but the branch
  cost likely outweighs.

## INSIGHT (transferable)
⭐ **A vertically-periodic (single-colour) tiling's period+diagonal-shift is recoverable with PURELY
1-D arithmetic — no 2-D shifted comparison planes.** Encode each row's occupancy as a base-2 bitmask
`bm[r]=Σ_c 2^c·occ[r,c]` (fp32-exact, ≤1023); a right-shift-by-s with grid clipping is EXACTLY
`(bm[r-t]·2^s) mod 1024`. So self-consistency for candidate (t,s) is `bm[r]==(bm[r-t]·2^s) mod 1024`
over rows gated by `bm[r-t]>0 AND bm[r]>0` (the predecessor-exists gate is load-bearing: it excludes
the first t rows AND beyond-H rows, otherwise boundary mismatches corrupt the tie-break and a larger
true `tall` loses to a spurious `tall=1`). The diagonal shift per candidate is just
`clip(leftcol(row t) − leftcol(row 0), 0, 2)`. This collapses what looked like a 9× full-plane
detection (~9KB) into a handful of [8]-length vectors. Pairs with the task231 idiom (output = index
Gather of a recovered small value plane, expansion routed into the FREE bool `Equal` output).
