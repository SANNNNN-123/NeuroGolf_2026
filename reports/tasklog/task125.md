# task125 — 543a7ed5

**Rule:** 15×15 grid, background cyan(8). Several non-overlapping (separated by ≥2) solid pink(6)
rectangles, each with a smaller rectangular HOLE punched out of its interior (the hole shows as
background cyan inside the pink rect). OUTPUT, per rectangle: pink cells stay pink(6); the interior
hole becomes yellow(4); a 1-cell green(3) outline is drawn around the rectangle's bounding box.
Closed-form (verified 0/800): pink=in==6; enc = 4-directional prefix/suffix-OR of pink all true
(input has only pink & cyan, so first-non-cyan==first-pink ⇒ enclosed cyan == hole); region=enc∨pink;
green = dilate3×3(region) ∧ ¬region; out = 8; green→3; enc∧¬pink→4; pink→6.
**Current (prior):** 15.50 pts.
**Target tier:** A — closed-form bbox/enclosure, no flood-fill, no connectivity, output routed into FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-dir triangular-MatMul prefix/suffix-OR; hole=enc∧¬pink; green=maxpool−region; L=arith; Equal | A | 12825 | 484 | 15.50 | — | works, at floor |
| 2 | enc via product of 2 col/row sums; drop cyan slice (use ¬pink) | A | 11925 | 484 | 15.574 | — | trim |
| 3 | keep masks BOOL (225 vs 450); index via Where priority chain (3 Where) | A | 10350 | 484 | 15.71 | — | trim |
| 4 | priority-chain folds hole=enc & green=dilation (pink/enc override later → no ¬-masking) | A | 9450 | 484 | 15.796 | — | dropped 2 bool ANDs + 2 NOT |
| 5 | Cast index plane to uint8 BEFORE the 30×30 Pad (uint8 pad 900 vs fp16 1800) | A | 8775 | 484 | 15.867 | 500/500 | adopted-as-best |

## Best achieved
15.867 @ mem 8775 params 484 — adopted? N (per instructions). Beats prior 15.50? Y (+0.367). Fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: pink_f32 slice (900B fp32, Slice preserves input dtype), the L30 padded
index plane (uint8 30×30 = 900B, the one full-canvas plane the Equal one-hot must broadcast against),
and ~11 fp16 15×15 working planes (450B: pink-f16 for the 4 matmuls, the 4 directional sums, 2 row/col
products, region/dilation, 3 Where index stages). The fp16 15×15 planes are the active-canvas floor
(225 elems × 2B); the 4 matmuls are intrinsic to the 4-direction enclosure test. uint8 pad already
halves the 30×30 carrier; can't go lower than 900 there (bool can't be padded).

## OPEN ANGLES (re-attack backlog)
- Collapse the 3-stage Where index chain to 2 (encb⊂dilb): build inner=Where(encb,4,3),
  mid=Where(dilb,inner,8), L=Where(pinkb,6,mid) — same 3; a genuine 2-Where needs a packed
  pink/hole inner color routed by region. ~+0.05 if it lands.
- Pack the two row matmuls into one (pink @ [SU|SL]) — but the split+multiply re-adds planes; no net.

## INSIGHT (transferable)
⭐ Multi-component "find each rectangle + fill its hole + outline its bbox" is NOT a flood-fill /
connectivity wall when the input has only TWO colors (shape + bg): an interior hole = a bg cell with
shape-pixels in ALL FOUR directions = `aL∧aR∧aU∧aD` where each is a strict-triangular prefix/suffix-OR
MatMul (the task070 bbox idiom applied per-direction, NOT globally — so it handles many separated
boxes without merging them). And in a Where PRIORITY chain you can DROP the ¬-masking of every
lower-priority mask (set hole:=enclosed, green:=full-dilation) because the higher-priority Where
(pink, then hole) overwrites the over-marked cells — this killed 2 ANDs + 2 NOTs (1.5kB) for free.
