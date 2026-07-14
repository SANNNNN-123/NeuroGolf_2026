# task032 — 1e0a9b12

**Rule:** Grid is s×s (s∈[4,6]) of bg(0) in the 30×30 top-left corner. Each column c
holds exactly ONE colour colors[c] in cnt[c] arbitrary rows. Output drops those cnt[c]
cells to the BOTTOM of the column (rows s-1 … s-cnt[c]). Per-column gravity: output(r,c)
is coloured iff in-grid (r<s, c<s) AND r ≥ s-cnt[c]; colour = the column's unique colour.
Empty in-grid columns (cnt=0) DO occur and must still show bg(ch0) for r<s.
**Current:** 16.76 pts (public net)
**Target tier:** B (count-parametric per-column rebuild; output colours copy arbitrary
input colours → needs a colour-index route, not a fixed Conv).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf-Conv label-map, 6 bool/u8 planes, 30×30 Equal | B | 6814 | 678 | 16.08 | — | works, too many planes |
| 2 | fp16 additive band (colidx·col + 50·offrow + 50·offcol) | B | 13176 | 647 | 15.47 | — | MORE f16 planes, worse |
| 3 | 4 planes: coloured + offgrid OR + Lc + L (u8) | B | 5018 | 650 | 16.36 | — | better |
| 4 | 2 planes: vector-clamp coloured + Where→L, Equal→bool out | B | 3154 | 651 | 16.76 | — | = P |
| 5 | 1 plane: coloured + Where(colmask_u8,elsemask_u8)→out | B | 3454 | 651 | 16.68 | — | mask casts cost 1200>L plane |
| 6 | #4 with 3-ch conv (drop ReduceMax) | B | 3034 | 952 | 16.71 | — | +300 params > saved mem |
| 7 | #4, s from colin only (drop 2nd ReduceMax plane), f16 vectors | B | 2974 | 651 | **16.80** | 500/500 | BEST |

## Best achieved (2026-06-19 re-attack — WIN)
**17.50 @ mem 1160 params 648** — beats adopted P=16.867 by **+0.63**. fresh 200/200
(truly isolated, independent generator load). src/custom/task032.py.

The prior "2-plane floor → MARGINAL" verdict was WRONG: it never tried CROP-TO-ACTIVE
nor the colour-0-is-bg observation. Two structural escapes broke it:
1. **colour-0 == background** (scoring is out>0): a colour-0 column writes value 0 to
   its active cells == bg, so it is indistinguishable from an empty column in BOTH
   input and output. Handle ONLY colour>=1 columns; colour-0 columns fall out as
   all-bg automatically. This removes the entire "distinguish bg from colour-0" plane.
2. **CROP-TO-ACTIVE (grid <=6x6)**: build EVERY working plane at 6x6. coloured drops
   900B->36B, the routed one-hot drops to a [1,10,6,6] u8 (360B) Pad'ed back to 30x30.
3. **Do NOT slice the input** to [1,10,6,6] (1440B f32). Run the colsum/cnt Conv on the
   FULL free input (W[2,10,30,1] -> [1,2,1,30], 240B) and slice only the cheap 30-wide
   vectors to width 6. The 600 conv params cost far less than a 1440B input window.
4. **ONE Where into the output, NO L plane**: coloured (6x6 bool) selects between
   Xonehot[k,c]=(colidx99[c]==k) ([1,10,1,6]) and Yonehot[k,r]=(k==0 AND r<s)
   ([1,10,6,1]); off-grid columns -> cnt2=100 (coloured) + colidx99=99 (sentinel) =>
   all-zero; off-grid rows -> Yonehot=0. Where needs uint8 branches (bool Where is
   NOT_IMPLEMENTED under ORT_DISABLE_ALL, still true on this build).

Dominant cost now: conv params 600 (the full-height kernel) + the [1,10,6,6] u8 Where
(360B). Remaining open angle: shave the 600 conv params (every cheaper colidx/cnt route
costs a >=1200B presence plane, net worse).

## (superseded) Best achieved
16.80 @ mem 2974 params 651 — beats prior 16.76 by **+0.04** → MARGINAL (< +0.3).

Key encoding (2 full 30×30 planes only):
- ONE Conv W[2,10,30,1]: ch0 weight=k → per-col colour-SUM; ch1 weight=1(k≥1) → per-col
  count cnt[c]. Output [1,2,1,30] (NO 30×30 colour plane). colidx = round(colsum/max(cnt,1)).
- in-grid: colin = ReduceMax(input,axes=[1,2]) [1,1,1,30]; s = ReduceSum(colin) (square grid).
- VECTOR sentinel tricks (no extra plane): for off-grid cols (colin==0) set cnt2=100 AND
  colidx99=99 → they become "coloured" but route to sentinel; elsevec[r]=99 if r≥s else 0.
- bot[r]=(s-1)-rowramp, clamp bot<0 → 99 so off-grid rows fall outside [0,cnt).
- PLANE 1: coloured = Less(botc[1,1,30,1], cnt2[1,1,1,30]) (bool 900B).
- PLANE 2: L = Where(coloured, colidx99, elsevec) (u8 900B).
- Equal(L, chan) → FREE bool output.

## Irreducible-floor analysis
Two 30×30 planes (1800B) are the floor: `coloured` is a genuinely 2-D threshold
(rowramp[r] vs per-column threshold s-cnt[c]) — not row⊗col separable, so it cannot
collapse to a vector; and the label `L` must merge the colour-by-column with the
bg-vs-offgrid-by-row split, which Where does in one u8 plane. Routing into the FREE
output via one Where(colmask_u8,elsemask_u8) instead removes L but the two Equal→Cast
uint8 masks cost 1200B > the 900B L plane, so the 2-plane Equal route is cheaper.
The Conv (600 params) is also at floor: colidx needs a per-col colour-SUM (a ReduceMax
presence route is a 1200B [1,10,1,30] plane; the Conv is 240B+600 params, far cheaper),
and cnt needs the count channel; a 3rd all-ones channel for in-grid costs +300 params >
the 180B it saves. Total ≈ 1800 + 600 + ~1225 scaffolding = ~3625 → 16.80, ~0.04 over P.

## OPEN ANGLES (re-attack backlog)
- Single 900B plane: would need a comparison op that emits the colour-index label directly
  (compare-and-select-from-2-vectors in one op) — no ORT op does this. Where needs a
  precomputed bool condition tensor, so ≥2 planes.
- Eliminate the conv: MatMul row-contraction keeps the 10 channels → [1,10,1,30] 1200B
  plane (worse than Conv's 240B + 600 params). No win.
- The +0.3 bar is structurally out of reach: public net already at 16.76 ≈ the 2-plane floor.

## INSIGHT (transferable)
⭐ Per-column "gravity/stack-to-bottom" with one colour per column is closed-form tier-B:
colidx=colsum/cnt and cnt come from ONE Conv W[2,10,30,1] (no 30×30 colour plane); the
colored region is a per-column bottom-run threshold `r ≥ s-cnt[c]` = ONE bool plane via a
VECTOR clamp (`botc=Where(bot<0,99,bot); coloured=Less(botc,cnt)` puts the r<s upper bound
into a [1,1,30,1] vector, not a second plane). ⭐ The bg-vs-offgrid-vs-colour 3-way split
needs NO extra plane: push off-grid COLUMNS into the "coloured" branch by setting their
cnt=BIG and colidx=99 (sentinel, both VECTOR ops), and resolve in-grid-bg vs off-grid ROWS
with a [1,1,30,1] `elsevec` (0 if r<s else 99) — so the whole label is ONE Where over two
vector branches. Net floor = exactly 2 full planes (coloured bool + label u8). But when the
public net is already at the 2-plane floor (~16.76 here), there's no +0.3 to take.
