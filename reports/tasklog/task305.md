# task305 — c3f564a4

**Rule:** 16x16 grid of diagonal stripes `cell = (r+c) % colors + 1`, `colors` a per-instance scalar in {4..9}. INPUT blacks out (color 0) several random rectangles; OUTPUT is the clean full stripe pattern. Entire output is a deterministic function of one scalar `colors`. Off-grid (beyond 16x16) is background everywhere.
**Current:** 16.72 pts (public net)
**Target tier:** B (closed-form arithmetic of one recovered scalar) — exact analogue of task061 (r*c table).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | recover colors=max-colour; (r+c)%colors plane -> Pad -> Equal into free BOOL output | B | 1754 | 285 | 17.38 | 200/200 | ADOPT |

## Best achieved
17.38 @ mem 1754 params 285 — beats 16.72 by +0.66.

## Irreducible-floor analysis
Dominant intermediate is the padded 30x30 uint8 label plane (900 B) feeding the free
Equal, plus the 16x16 fp16 (r+c)%colors plane (512 B) + its uint8 cast (256 B). No
[1,10,30,30] is ever materialized — the 10-channel expansion is routed into the FREE
BOOL output via Equal(L_uint8, chan-1). The "+1" colour offset is folded into the
compare constant chan-1=[255,0,1,..,8], so no extra add plane.

## OPEN ANGLES
- Could trim the uint8 cast by Mod-then-Cast directly inside Equal operands, but Equal
  needs matching dtypes and the Pad gateway is fp-incompatible (Pad rejects bool/needs
  the uint8 carrier); current form is near the structure-escape floor for a label map.

## INSIGHT (transferable)
"Restore a deterministic (f(r,c) % param)+1 grid with random blackouts" = recover param
as max-colour-present (ReduceMax channel-weighted), recompute arithmetically with one
fp16 Mod (exact <2048), Pad to 30x30 with a no-match sentinel, route into free BOOL
output via Equal against chan-1 (folds the +1). Direct sibling of task061 (r*c). Any
"clean table from corrupted table" task collapses to one scalar + one Mod plane.
