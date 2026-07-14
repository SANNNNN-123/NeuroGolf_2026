# task165 — 6d58a25d

**Rule:** Active 20x20 grid. Scattered single pixels in colour `color`, plus ONE fixed
10-cell "kite" shape in colour `kite` anchored at top cell (row,col) (row in [1,10],
col in [5,14]). Kite cells rel to anchor: (0,0)(1,-1)(1,0)(1,1)(2,-2)(2,-1)(2,1)(2,2)
(3,-3)(3,3). For each kite column (col-3..col+3): if a `color` pixel lies BELOW the
topmost kite cell of that column, the column "drips" — every cell from just below the
LOWEST kite cell of that column down to bottom row 19 is painted `color`. Per-column
offsets: toprel{-3:3,-2:2,-1:1,0:0,+1:1,+2:2,+3:3}, lowrel{-3:3,-2:2,-1:2,0:1,+1:2,+2:2,
+3:3}. trigger(dc)=exists color pixel at col+dc with r>row+toprel(dc); fill = rows
[row+lowrel(dc)+1 .. 19].
**Current (stored):** ~13.89 pts (UNTRIAGED pending pool).
**Target tier:** detection/closed-form hybrid — anchor-detect + per-column gated vertical
fill. NOT a flood-fill (each drip is a deterministic fixed-offset bar gated by a per-column
scalar trigger), so it beats the multi-plane fill floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Cauchy-Schwarz anchor + per-tooth full-plane trigger + Where output | det | 63281 | 187 | 13.94 | — | works, too heavy |
| 2 | replace 7 full-plane triggers with lastcolor[1,1,1,20] vector + startrow chain | det | 46061 | 229 | 14.26 | — | better |
| 3 | + slice S1/S2 conv to anchor region; build fill30 directly (drop uint8 pad roundtrip) | det | 29875 | 259 | 14.69 | — | better |
| 4 | + slice colf to conv region [0:17,0:18] BEFORE squaring/conv (kills colf2 3600, shrinks S-planes) | det | 20809 | 254 | 15.04 | 200/200 | **BEST** |

## Best achieved
15.04 @ mem 20809 params 254 — adopted? N (per instructions, do not adopt). Beats prior
~13.89? YES (+1.15). Fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate = `colf` [1,1,30,30] fp32 = 3600B, the per-cell value plane
(sum_k k*input_k via 1x1 Conv). Irreducible because a 1x1 Conv over the 30x30 input
preserves 30x30; shrinking it would require slicing the [1,10,30,30] input (16000B+, far
worse). Secondary: colf20 (1600, the 20-row value slice needed for colormask over the full
active height) and rmasked (1600, the Where(colormask,rowramp,-1) used to get per-column
last-color-row). Both are genuine 20x20 planes. The Cauchy-Schwarz S-plane chain is already
shrunk to [1,1,14,12]=672B each by region-slicing the conv.

## OPEN ANGLES (re-attack backlog)
- Merge rmasked + colf20: colormask could come from a direct Gather of the `color` channel
  (input[:,color]) instead of colf20==color, but Gather still yields a [1,1,*,*] plane — no
  net win, and color is data-dependent so the Gather index is a scalar tensor (fine).
- Pack roww/colw/kw (three iswin*ramp Muls, 672 each) into ONE MatMul(iswin, W[.,.,.,3])
  packing rowramp/colramp/S1 as 3 contraction columns → could drop 2 planes (~1300B).
- The fill30 trio (filllt/fillge/fill30, 900 bool each) could shrink if the row<20 cap were
  folded into startrow, but no per-row fold exists without a 2nd plane.

## INSIGHT (transferable)
⭐ A data-dependent "drip to the bottom" / vertical-fill-from-a-detected-shape is NOT a
flood-fill wall when (a) the fill geometry is a FIXED offset from a uniquely-detectable
anchor and (b) the per-column on/off trigger reduces to a scalar. Recipe: detect the anchor
with a Cauchy-Schwarz Conv pair (10*S2==S1^2 AND S1>0 finds the unique window where all K
pattern cells share one nonzero value — generalizes "8-neighbours-all-equal" to ANY fixed
multi-cell shape, and recovers the shape's COLOUR as S1/K for free), recover (row,col) as
ReduceMax(iswin*ramp), then build a per-column `startrow` vector [1,1,1,W] (BIG where no
drip) and fill = (rowramp >= startrow) in ONE broadcast — no per-tooth full planes.
⭐ Region-slice the value plane BEFORE the detection Conv (not after): the anchor's bounded
coordinate range (from generator bounds) lets you slice colf to the exact conv-relevant
window first, which simultaneously kills the colf^2 full plane and shrinks every downstream
S-plane (here 3600->1224 and 2592->672). Slicing the conv OUTPUT only shrinks the latter.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** Folded a redundant `Flatten` (26→25 nodes). −8B, params flat.
**Old→new:** mem 5836→5828, params 230→230.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task165_pre_s10.onnx`; source `public_candidates/bobmyers7186/task165.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.224)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 4840 (mem 4635 + params 205), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: QLinearConv u8 signed-renderer replaces ConvTranspose+GatherND scatter (fp16 kernel->u8); mech (c) self-applied.
