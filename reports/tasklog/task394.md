# task394 — f9012d9b

**Rule:** A `size x size` grid (size in 4..7) is tiled by an `m x m` colour pattern
(`m = minisize = 2 if size<7 else 3`) of 2 random colours:
`grid[r][c] = colors[(r%m)*m + (c%m)]`. A `bs x bs` bite (bs in 1..3, always <= m) at
(row,col) is blacked out in the INPUT; the OUTPUT is that bite window holding the
ORIGINAL pattern colours. So output(r,c) = pattern value at phase ((row+r)%m,(col+c)%m).
**Current:** 16.53 pts (public CumSum-scan net), mem ~big, params ~big
**Target tier:** A (closed-form copy/gather — periodic reconstruction, no detection wall)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 phase double-MatMul into 30x30 Equal | A | 44698 | 172 | 14.29 | — | colored slice [1,9,30,30] floor |
| 2 | 7x7 crop + phase MatMul, fp16 pad carrier | A | 9539→6884 | ~104 | 15.8→16.1 | — | Pad-to-30x30 plane dominated |
| 3 | 1x1 Conv colf/occ + double-MatMul pattern, uint8 carrier | A | 4944 | 114 | 16.47 | — | matmul machinery heavy |
| 4 | spatial-shift Gather (no matmul) + uint8 carrier | A | 3907 | 70 | 16.71 | — | carrier still 900B |
| 5 | + fp16 scalar logic | A | 3585 | 70 | 16.80 | — | close |
| 6 | Pad 3x3 one-hot straight into FREE output | A | **2865** | **70** | **17.02** | 500/500 | ADOPTED |

## Best achieved
17.02 @ mem 2865 params 70 — adopted? Y. Beats prior 16.53? Y (+0.49).

## Irreducible-floor analysis
Dominant intermediate = the `[1,9,7,7]` fp32 crop (1764 B). It is the cheapest entry
that lets a 1x1 Conv collapse the 9 colour channels to a 7x7 value plane AND gives the
per-row/col coloured counts; ORT Slice preserves fp32 and a full-input channel-collapse
Conv would cost a 3600 B 30x30 plane, so 1764 is the structural floor for this approach.
Everything else is <= 196 B (colf) and tiny fp16 scalars. The old ~3600 B colour-index
carrier was REMOVED by padding a [1,10,3,3] one-hot directly into the FREE output.

## OPEN ANGLES (re-attack backlog)
- xc 1764 B: could drop to 6x6 (source indices are always <=5 since size>=2m) IF size
  detection (needs row/col index 6 for size=7) were recovered without the 7th row/col —
  e.g. detect m from pattern periodicity instead of size. Would save ~480 B (~17.2).
- crow32/ccol32 are fp32 (56 B) only because ReduceSum(fp32 xc) is fp32; no clean fp16 path.

## INSIGHT (transferable)
⭐ "Bite/occlusion of a PERIODIC pattern" is a closed-form spatial COPY, not a fill/detection
wall: the blacked cell at index a has an intact same-phase twin one period away, and because
the generator guarantees `size >= 2*m`, `srcR(a) = a-m if a>=m else a+m` ALWAYS lands on an
intact on-grid cell — two tiny Gathers read the answer with NO pattern matmul.
⭐ Carrier-elimination: when the output is a small region (here <=3x3) of an otherwise-
background [1,10,30,30], build the [1,10,bs,bs] one-hot small and `Pad` it (value 0) DIRECTLY
into the FREE uint8 output — removes the ~900 B uint8 (or 1800 fp16 / 3600 fp32) 30x30 colour
carrier entirely. Declare the graph output as UINT8 so the final Pad emits `output`.
