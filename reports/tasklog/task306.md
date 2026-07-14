# task306 — c444b776

**Rule:** A `width×2` lattice of 9×9 quadrants separated by single yellow(4) gridlines
(height/size FIXED at 2/9 ⇒ 19 rows, W=10·width−1 cols, width∈{1,2,3}). The INPUT has up
to 10 coloured pixels (colours in {1,2,3,5,6,7,8,9}, never bg/yellow) in exactly ONE
quadrant; all other quadrants empty. The OUTPUT stamps that 9×9 pattern into EVERY
quadrant, gridlines unchanged. Pure fixed-lattice SPATIAL-COPY: out[r,c] = donor colour at
local (r%10, c%10).

**Current (prior):** 15.84 pts, mem 8818, params 692 — fold-value-plane + 6 fp32 9×9
slice-Max fold + 19×29 uint8 unfold + glmask Where + Pad + Equal (never-analyzed import).

**Target tier:** Tier-A closed-form. (Not pure Tier-S mem-0: the Equal index-feeder pins
ONE full 30×30 carrier plane; the one-hot value of each cell depends on the colour, so a
value carrier + Equal is the floor unless a one-hot table gather is used — that pays a
[1,10,11,30]=3300 middle plane, strictly worse.)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (prior import) fold value-plane, 6 fp32 9×9 slices, 19×29 unfold | A | 8818 | 692 | 15.84 | — | baseline |
| 2 | reshape→ReduceMax fold (Lr doubles Lin to 3600×2) | A | 10101 | 106 | 15.77 | — | worse (reshape plane) |
| 3 | 6 fp32 9×9 slice-Max fold + 11×11 table + 2-D Gather + Equal | A | 8450 | 161 | 15.94 | — | marginal +0.10 |
| 4 | DEPTHWISE dilated(10) fold→[1,10,10,10] + 1×1 collapse | A | 6512 | 211 | 16.19 | 200/200 | +0.35 |
| 5 | **FUSED dilated-conv fold+collapse → [1,1,10,10] in ONE Conv** | A | **2512** | **201** | **17.09** | 500/500 | **ADOPT, +1.25** |

## Best achieved
**17.09 @ mem 2512, params 201** — beats prior 15.84 by **+1.25**. ReferenceEvaluator
cross-check 0 mismatches (no ORT grouped/dilated-conv silent bug). Fresh isolated:
200/200, 500/500, plus 300/300 with width∈{1,2,3} all covered.

## Key construction
1. ONE non-grouped Conv with a 3×3 kernel DILATED by 10, weight[0,k,i,j]=k for the 8
   content channels (0 for bg-ch0 & yellow-ch4): fuses "fold the 3×3 quadrant grid" AND
   "collapse 10→1 to a colour value" into a single op → patt[1,1,10,10] fp32 (400 B).
   Exactly one quadrant is filled so the dilated SUM equals the donor colour (no
   double-count); bg/yellow are double-summed across quadrants so they are EXCLUDED and
   reconstructed in the lookup table.
2. Cast→uint8, Slice to 9×9, two Pads build an 11×11 lookup table: index 9 = gridline
   lane (value 4), index 10 = off-grid sentinel (99).
3. 2-D Gather (col map [30] then row map [30]) replicates the pattern into every quadrant
   AND overlays gridlines from the same table; rows≥19 map to the off-grid index.
4. Off-grid COLUMNS (width-dependent) masked on the SMALL [1,1,11,30] gc plane via the
   always-present row-9 gridline presence input[4,9,c] (avoids a 2nd full carrier).
5. Equal(L[1,1,30,30] uint8, arange_ch[1,10,1,1]) → BOOL output (10-ch expansion FREE;
   4→yellow, 99→nothing→off-grid all-zero one-hot).

## Irreducible-floor analysis
Dominant intermediate = L[1,1,30,30] uint8 (900 B), the Equal index-feeder — one full
30×30 carrier is the floor for the Equal route (ORT Equal accepts uint8, so it's 900 not
int32 3600). Next: gc/gcm [1,1,11,30] uint8 (330 each) and patt [1,1,10,10] fp32 (400,
Conv output must be fp32). The fold/collapse — the prior net's biggest cost (Lin 3600 +
6×324 slices) — is GONE, fused into the 400 B patt Conv.

## OPEN ANGLES
- Eliminate L: route directly to output via a one-hot lookup table [1,10,11,11] gathered
  to [1,10,30,30] (no Equal). Blocked: the middle gather plane [1,10,11,30] = 3300 uint8
  (×10 channels) is far worse than gc 330 + L 900. Not pursued.
- gc/gcm pair (660): could collapse if the off-grid col-mask folded into a runtime col
  INDEX vector, but that introduces an int64 [30] ceff (240) — measured net-neutral.

## INSIGHT (transferable)
⭐ FIXED-LATTICE QUADRANT-TILE = ONE DILATED CONV that fuses FOLD + CHANNEL-COLLAPSE.
When a "copy a donor quadrant into every cell of a fixed gridline lattice" task has exactly
ONE filled source region, a single non-grouped Conv with a (kernel)×(kernel) all-tap kernel
DILATED by the quadrant pitch and weight[0,k]=k (the colour value, bg/separator channels
zeroed) directly emits the donor's colour value plane at the *small local* resolution
(K×K), collapsing 10→1 AND max/sum-folding all quadrants in ONE op — no per-quadrant slices,
no 30×30 value plane, no separate collapse Conv. Then a tiny K×K→(K+2)×(K+2) lookup table
(sentinel lanes for separators + off-grid) + a separable 2-D const-index Gather reconstructs
the full canvas into the FREE output. This took the prior fold-value-plane net 8818→2512
(15.84→17.09, +1.25). Generalises any "tile one block on a periodic separator lattice".
