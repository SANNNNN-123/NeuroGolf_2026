# task175 — 73251a56

**Rule:** Fixed SIZE=21 square grid. `color = 2` on the diagonal (r==c), `(r+2)//(c+2)`
for r>c, `(c+2)//(r+2)` for r<c; `value[r,c] = (color + modset) % mod + 1` with
`mod` in 5..9 and `modset` in 1..4 per puzzle. The INPUT is this value plane with up to
5 black (colour-0) rectangular cutouts; the OUTPUT restores the clean plane. value>=1
everywhere, so black marks ONLY cutouts -> output is a pure function of (COLOR, mod, modset)
and the input only supplies the two scalars.
**Current (prior):** 15.82 pts, LUT-scoring/ArgMax label-map, cidx 900B + basegrid + 20 candidate LUTs.
**Target tier:** B (general per-cell deterministic rule, COLOR is a constant matrix not a
copy of input colours, and value=ratio(r,c) is not row⊗col separable) — reached the B floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | arithmetic plane, fp32 everywhere, modset via masked-Mod ReduceMax | B | 23038 | 478 | 14.93 | 200/200 | too many fp32 21x21 planes |
| 2 | same, cast to fp16 downstream | B | 15974 | 478 | 15.29 | 200/200 | fp16 helped, still many planes |
| 3 | gather COLOR==1 values from flattened entry, fp16 | B | 10839 | 703 | 15.65 | 200/200 | duplicated 30x30 entry plane |
| 4 | **mod & modset from per-channel COUNT vector (ArgMax(cnt)==v1); NO 30x30 plane** | B | **3297** | **483** | **16.76** | **500/500** | ADOPT |

## Best achieved
16.76 @ mem 3297 params 483 — beats prior 15.82 by **+0.94** (>> +0.3). fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: the padded uint8 label L [1,1,30,30] = 900B, plus two fp16 21x21
value planes (cm, vmod) = 882B each, plus the uint8 21x21 value = 441B. These are the
minimal Tier-B label-map cost: the output colour per cell is `(ratio(r,c)+modset)%mod+1`,
a per-cell deterministic rule that is NOT a spatial copy (Tier S) and NOT row⊗col separable
(the ratio couples r,c), so a single uint8 label plane + final Equal is the floor (~16.8).
Recovery added ZERO 30x30 working planes — both scalars come from a [1,10,1,1] count vector.

## OPEN ANGLES
- Combine cm+vmod (the +modset and %mod) — both fp16 21x21, seem irreducible (Add then Mod).
- Pad L is 900B uint8; building the value plane directly at 30x30 (fp16, COLOR padded) would
  let Equal skip the Pad but makes the value planes 1800B each — strictly worse.
- Tier A is blocked: value=ratio(r,c) is not separable, so no row⊗col one-hot form exists.

## INSIGHT (transferable)
⭐ When a deterministic plane is parameterised by a SMALL set of scalars (here mod, modset)
recovered from a cutout-corrupted input, you often need NO full-canvas working plane: derive
the scalars from the per-channel pixel-COUNT vector `cnt=ReduceSum(input,[2,3])` ([1,10,1,1],
40B). Here `mod = max present index` and, because the dominant residue band (COLOR==1, 220 of
441 cells) so over-represents the histogram that ARGMAX(cnt)==v1 survives up to ~125 cut cells
(min top1-top2 margin 62 over 5000), `modset=(v1-2) mod mod`. The whole LUT-scoring/ArgMax/
basegrid-Gather machinery of the prior net collapses to one ReduceSum + one ArgMax. Also: fold
the `+1` of `value=...%mod+1` into the channel constant (chan[k]=k-1, chan[0]=255 unreachable)
to drop a full-canvas Add plane.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 1267 -> 1255 (+0.010); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].