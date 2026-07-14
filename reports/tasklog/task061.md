# task061 — 29ec7d0e

**Rule:** The grid is 18x18; every cell holds `(r*c) % mod + 1` with `mod` a single
per-instance integer in {4..9}. The INPUT blacks out (value 0) a handful of random
rectangles; the OUTPUT is the clean multiplication-table with NO blackouts. The entire
output is therefore a deterministic function of one scalar (`mod`) — nothing spatial to
detect. Verified `input.max() == mod` on 2000 fresh instances (every colour 1..mod
appears in an 18x18 r*c range; nothing > mod).
**Current:** 17.25 pts (custom:task061), mem 1958, params 353 — was 15.15 (public ext:kojimar6275).
**Target tier:** B-ish (single label plane → Equal), but the spatial work collapses to a
single scalar `mod`. No higher tier: `(r*c)%mod` is NOT row⊗col separable (Tier A blocked)
and is not a per-cell-local linear function of the one-hot (Tier S blocked) — the value
depends on the global product r*c reduced mod a recovered scalar.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | recover mod=ReduceMax(k*pres); look up [6,18,18] uint8 table by mod-4; Pad; Equal | B | 1640 | 1978 | 16.81 | 200/200 | works (sentinel-0 bug fixed: outside 18x18 is all-channels-off, pad 10 not 0) |
| 2 | same, table reshaped [6,1,1,18,18] so Gather→[1,1,18,18] (drop Reshape plane) | B | 1320 | 1974 | 16.90 | 200/200 | better; TAB params (1944) now dominate |
| 3 | DROP table: PROD=r*c fp16 plane, Mod(fmod=1) by mod, +1 fp16, Cast uint8, Pad, Equal | B | 2606 | 354 | 17.01 | 200/200 | params 1974→354; arithmetic beats lookup |
| 4 | fold +1 into the constant: compare uint8 remainder vs chan-1=[255,0,1..8], pad 254 | B | 1958 | 353 | 17.25 | 500/500 | ADOPTED-candidate; removes the fp16 Add plane |

## Best achieved
17.25 @ mem 1958 params 353 (sum 2311) — adopted? **N** (build-only; main adopts). Beats prior 15.15? **Y** (+2.10).

## Irreducible-floor analysis
Three intermediates: padded 30x30 uint8 label plane (900 B), fp16 Mod result (648 B),
its uint8 cast (324 B), plus ~80 B mod-recovery aggregates. The 900 B plane is the
irreducible gateway — Equal's first input must be [1,1,30,30] to broadcast against
chan-1[1,10,1,1] into the free [1,10,30,30] output, and uint8 (900) beats fp16 (1800).
The 648 B fp16 Mod output is the cheapest integer-exact carrier for r*c (max 289; int32
would be 1296 B, and ORT Mod needs int or fmod-float). PROD is a 324-element param.

## OPEN ANGLES (re-attack backlog)
- Mod directly on a 30x30 PROD (pre-padded) to skip the Pad — blocked: a sentinel can't
  survive Mod (any value mod m lands in 0..m-1 and collides with a real colour), so the
  pad-after-Mod ordering is required. Pad output stays the 900 B floor.
- Cast remainder to uint8 BEFORE any further op and Pad that (current) vs padding fp16
  then casting — current order is already optimal (uint8 Pad 900 < fp16 Pad 1800).
- Eliminate the 648 B fp16 rem: would need an integer-Mod path, but int32 rem = 1296 B
  (worse) and ORT Mod rejects uint8/int8 small dtypes. fp16 is the floor.
- Recover mod without the 80 B pres aggregates (e.g. a single Conv collapsing argmax) —
  marginal (~80 B / negligible ln payoff), not worth complexity.

## INSIGHT (transferable)
⭐ **When the output is a closed-form arithmetic function of a recovered scalar, REPLACE
the lookup table with the arithmetic.** Here a [6,18,18]=1944-param table (params dominate
the score) became a single 324-param `PROD=r*c` plane + one `Mod(fmod=1)`: fp16 Mod is
integer-exact for products < 2048 and ORT supports `Mod` with `fmod=1` on fp16 (int32 also
works but costs 4x memory). ⭐ **Fold scalar offsets into the final Equal constant, not a
separate Add plane:** comparing the remainder against `chan-1` (with channel-0's slot set
to an unreachable 255) bakes the `+1` colour offset in for free and deletes a whole 30x30
intermediate — and lets the outside-grid pad sentinel (254) be chosen to match no channel,
correctly producing the all-channels-off region beyond the 18x18 grid (the generator's
output is only 18x18, so cells r,c>=18 must be ALL-FALSE, not channel-0-true — the sentinel
must exceed every channel value, never 0).
