# task045 — 22eb0ac0

**Rule:** size=10 grid. For idx=0..len-1, row r=2*idx+1 (rows 1,3,5,7,9) has a LEFT endpoint colour at col 0 and a RIGHT endpoint colour at col 9 (=size-1). Input shows only the two endpoint cells per such row. OUTPUT = INPUT plus: every row whose LEFT colour equals its RIGHT colour gets its whole row filled with that colour (interior cols 1..8). Rows with differing endpoints stay unchanged.
**Current:** 16.09 pts (public)
**Target tier:** A/S — closed-form per-row line fill routed into the FREE Where output; no 10-ch plane materialised.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Where(qual∧interior, both, input); both=Slice(c0)*Slice(c9) | A | 4650 | 36 | 16.55 | 200/200 | ADOPT |

## Best achieved
16.55 @ mem 4650 params 36 — beats prior 16.09 by +0.46 (Y).

## Irreducible-floor analysis
Three [1,10,30,1] fp32 tensors dominate (leftcol/rightcol/both, 1200B each = 3600B); fillcond [1,1,30,30] bool is 900B; qual scalars 120B. The two endpoint slices + their product are genuinely needed: `both` is the per-row colour one-hot used as the Where value and must be [1,10,30,1] to broadcast across interior columns. fp16-casting the small slices adds tensors (the fp32 Slice output is unavoidable) and Where requires X/Y to match the fp32 `input`/`output`, so fp16 buys nothing. No [1,10,30,30] is ever materialised — it IS the free output.

## OPEN ANGLES (re-attack backlog)
- Replace the two 1200B endpoint slices with a single [1,10,30,2] gather of cols {0,9} then a product over the new axis (likely same/worse since the gathered tensor is also 1200B-class and adds a reduce).
- Collapse leftcol*rightcol via a Conv contracting the width axis with a 2-tap kernel hitting cols 0,9 — but Conv can't do the per-channel AND (it sums, not multiplies), so not equivalent.

## INSIGHT (transferable)
⭐ When a "fill" colour VARIES per row/region but is exactly a one-hot already present in the input, use that one-hot directly as the Where VALUE in [1,10,30,1] form — it broadcasts to the free [1,10,30,30] output, so you never need a colour-index plane or per-region Equal. Endpoint-match = elementwise product of the two boundary column slices; ReduceMax over channels gives the per-row qualify flag.
