# task296 — bc1d5164

**Rule:** Input is a height=5 × width=7 grid; a single colour's pixels are scattered but only
where NOT (row==2 OR col∈{2,3,4}), so active rows={0,1,3,4}, active cols={0,1,5,6}. Each pixel
(r,c) maps to a 3×3 output cell by r'=r if r<2 else r−2, c'=c if c<2 else c−4. Output cell is
painted the colour iff ANY source pixel maps there (OR; collisions allowed). ONE colour/instance.
**Current:** 17.32 pts, fixed separable gather + Pad-place one-hot, mem 2096, params 67
**Target tier:** A (separable row⊗col gather) — output is a CONSTANT linear map of input cells, so
it factors into two constant selector matrices; the only state is one colour scalar.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-canvas colf + occ + int32 label + Equal | B | 11949 | 202 | 15.59 | — | passes but < P |
| 2 | 5×7 slice + int32 label Pad(30) + Equal | B | — | — | — | — | Pad rejects int32 (invalid graph) |
| 3 | 5×7 slice + 3×3 one-hot + scatter via 2 placement MatMuls | A | 3896 | 247 | 16.67 | — | ties P, not +0.3 |
| 4 | 5×7 slice + 3×3 one-hot + Pad(0) place → FREE output | A | 2096 | 67 | 17.32 | 200/200 | ADOPT |

## Best achieved
17.32 @ mem 2096 params 67 — beats prior 16.68 by **+0.64**. Fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate is the 5×7 active-region slice `act` [1,10,5,7] fp32 = 1400B; it is the
input materialised at its true extent and Conv needs fp32 (casting it to fp16 ADDS a second tile,
net worse — attempt verified 17.08). The 3×3 gather/label/one-hot tiles are all <360B. There is
NO 30×30 colour/label plane: the 10-channel expansion lands on the tiny 3×3 (Equal→[1,10,3,3]
fp16, 180B) and the placement to 30×30 is a single **Pad with 0** whose result IS the FREE output.

## OPEN ANGLES (re-attack backlog)
- Drop the 5×7 slice by contracting occupancy off the FREE full input via MatMul(Rsel_full[1,1,3,30],
  input) → [1,10,3,30] (900 elems) — bigger than the slice, so not obviously a win; would need the
  colour scalar routed without a 10-ch tile.
- Recover colour without colf: ReduceMax over the occupancy-gathered tile times a channel ramp.

## INSIGHT (transferable)
⭐ For a FIXED (data-independent) fold/compaction to a SMALL output grid: do the whole 10-channel
one-hot on the tiny output-sized tile (Equal(L_small, arange)→bool [1,10,k,k]), then place it into
the 30×30 canvas with a single **Pad(value=0)** — Pad accepts fp16, off-grid stays all-zero (the
exact target since `convert_to_numpy` leaves off-grid cells unset, NOT bg=ch0), and the padded
tensor IS the FREE output. This beats both (a) a full-canvas int32 label+Equal (3600B int plane)
and (b) scatter via two placement MatMuls (1800B [1,10,30,k] intermediate). Net: only the input
slice survives as a real intermediate. Also: a constant separable gather factors into two {0,1}
selector matrices Rsel[k,H]/Csel[k,W] (presence = Rsel@occ@Csel^T > 0), no data-dependent matrix.
