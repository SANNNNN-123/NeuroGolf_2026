# task316 — cdecee7f

**Rule:** Input is a 10×10 grid with 6..9 single coloured pixels (colours 1..9, may
repeat), exactly ONE per distinct column, with columns sorted ascending. The 3×3 output
is filled in reading order with the colours taken in ascending-column order:
`out[0]=[c0,c1,c2]; out[1]=[c3,c4,c5] then REVERSED -> [c5,c4,c3]; out[2]=[c6,c7,c8]`.
Missing slots (fewer than 9 colours) stay background 0; cells outside the 3×3 are background.
**Current (stored before):** 16.73 pts, public net.
**Target tier:** B/closed-form — colours are arbitrary per-instance (Conv can't route to a
fixed S-tier output) but the whole transform is a closed-form compaction+permutation with a
TINY 3×3 active output, so no full plane is ever needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf 1×1 Conv→[1,1,10,10]→ReduceSum rows; scatter MatMul[30,9]; Equal→Pad | B | 6972 | 59 | 16.14 | 200/200 | works |
| 2 | ReduceSum(rows) then 1×1 Conv (drop 3600B colf plane) | B | 4572 | 59 | 16.56 | 200/200 | better |
| 3 | ONE no-pad Conv [1,10,30,1] folds row-sum + channel-contract | B | 3372 | 349 | 16.78 | 200/200 | better |
| 4 | Slice colcolor/idx/P to 10 cols (grid is 10×10) | B | 1412 | 352 | 17.52 | 200/200 | better |
| 5 | fp16 scatter MatMul (colours 0..9 exact in fp16) | B | 1216 | 352 | 17.64 | 500/500 | **best** |

## Best achieved
**17.64 @ mem 1216 params 352** — adopted? N (orchestrator gates). Beats prior 16.73 by
**+0.91** (≥+0.3 ✓). Stored 266/266, fresh ISOLATED 500/500.

## Method
1. `colcolor[c] = Σ_r Σ_k k·input[k,r,c]` in ONE no-pad Conv (kernel [1,10,30,1], w[0,k,r,0]=k)
   → [1,1,1,30]; Slice to 10 cols → [1,10] (per-column colour, one nonzero per occupied col).
2. `occupied = colcolor>0`; `idx = exclusive CumSum(occupied)` = destination reading index.
3. `dest = Gather(perm, idx)` with `perm=[0,1,2,5,4,3,6,7,8]` folding the middle-row reversal.
4. Scatter via runtime one-hot `P[c,j]=(dest[c]==j)&occupied[c]`; `colorvec[1,9]=colcolor@P`
   (fp16 — values 0..9 exact). Reshape → L[1,1,3,3].
5. `Equal(L, arange[1,10,1,1])` → [1,10,3,3] one-hot → Cast uint8 → Pad to [1,10,30,30].
   Off-grid pads stay 0 → background; empty slots are colour 0 → ch0=1, correct.

## Irreducible-floor analysis
No full 30×30 plane exists anywhere (active output is 3×3). Largest intermediates: the fp16
scatter matrix P [10,9]=180B and the Conv output [1,1,1,30] fp32=120B. The 300-element Conv
kernel (params) is structural: collapsing all 30 grid rows to height-1 in one op needs kh=30
(a 1×1 Conv on a ReduceSum'd [1,10,1,30] would instead cost a 1200B intermediate — worse). The
score denominator (~1568) is split roughly half mem / half the row-collapse kernel.

## OPEN ANGLES (re-attack backlog)
- Replace the [10,9] scatter MatMul with ScatterElements into a length-9 vector (would drop
  the [10,9] P and its two bool builds, ~360B of tensors) — marginal, score already near 17.6.
- Drop the 300-param Conv kernel: if a cheaper row-collapse to [1,10] exists (e.g. MatMul with
  a [30,1] ones contracted on the row axis + separate channel weight) it could shave ~300
  params; unclear it beats the current mem/param split.

## INSIGHT (transferable)
"Read scattered pixels in column order into a small fixed grid" = compaction by EXCLUSIVE
CumSum of an occupancy indicator → destination index, then a runtime one-hot scatter MatMul;
a fixed output PERMUTATION (here the reversed middle row) folds for free into a Gather(perm,idx)
table. Combined with the ⭐ one-no-pad-Conv row+channel collapse and a 3×3 active canvas (no
full plane), this lands a 16.7→17.6 closed-form win. ⭐ When the output grid is tiny, never
materialise a 30×30 carrier — one-hot the small label and Pad uint8 into the FREE output.
