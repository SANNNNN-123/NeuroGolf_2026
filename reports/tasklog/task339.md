# task339 — d631b094

**Rule:** Input is a 3x3 grid holding `count` (1..9) pixels of a single random color C
at random positions; background is black (0). Output is a grid of shape 1 row x `count`
cols (`common.grid(count, 1, color)` = width=count, height=1), every cell colored C. In
the [1,10,30,30] one-hot encoding: channel C, row 0, cols 0..count-1 are 1; all else 0.
The whole output is determined by TWO scalars — the count N and the color C.
**Current:** 18.22 pts, ext:kojimar6275, mem 776, params 108
**Target tier:** COUNT->FIXED-PATTERN (task399 idiom) — output content is fully fixed by a
scalar count + scalar color, so it's the cheapest tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-chan ReduceSum cnts; N=sum of chans1..9; C=ArgMax+1->Equal ramp; colmask=Less(ramp9,N); Pad small[1,10,1,9] u8 into free output | count | 299 | 32 | 19.20 | 200/200 | pass |
| 2 | drop ArgMax/Equal chain: keep-mask zeroes ch0, colorhot=Greater(masked,0), N=ReduceSum(masked) (shared product) | count | 283 | 29 | 19.26 | 200/200 | adopted |

## Best achieved
19.26 @ mem 283 params 29 — adopted? Y. Beats prior 18.22? Y (+1.04).

## Irreducible-floor analysis
Dominant intermediates: `cnts` and `masked` [1,10,1,1] fp32 = 40B each (ReduceSum rejects
uint8/bool, so the per-channel count plane MUST be fp32), and the `small` carrier
[1,10,1,9] uint8 = 90B (Pad rejects bool, so the routed one-hot block must be uint8). The
90B carrier is the minimal block that Pad-expands into the free [1,10,30,30] output (10
channels x 9 max cols). The two 40B count planes are the count-rule floor. Background-only
3x3 never occurs (count>=1), and grid is always 3x3 so N in 1..9 — col-ramp width 9 exact.

## OPEN ANGLES (re-attack backlog)
- Merge `cnts`/`masked` into one plane: fold the ch0-keep mask into the count step. ReduceSum
  can't take a per-channel weight, but a single [1,10,1,1] could serve if colorhot is taken
  from `cnts` with an AND against a const keep-mask (saves one 40B plane, ~ -5 mem) while N
  comes from ReduceSum(cnts)-cnts[0]. Marginal (~+0.02 pts), not worth the op churn.
- Carrier as bool routed by Equal/Where into output instead of uint8 Pad — but Pad rejects
  bool and the col-mask+color-onehot already broadcasts cheaply; no clear sub-90B path.

## INSIGHT (transferable)
⭐ "Single color present + output keyed only on its pixel COUNT and the color" is the
COUNT->FIXED-PATTERN tier even when the output is a variable-length 1-D strip (not a fixed
KxK stamp): build a [1,1,1,Nmax] col-ramp mask `Less(ramp, N)`, AND it with the color
one-hot `Greater(masked,0)`, and Pad the tiny [1,10,1,Nmax] uint8 block straight into the
free output. The color one-hot AND the count N both fall out of ONE keep-masked count
product `masked = cnts * keep` (keep zeroes ch0) — no ArgMax/OneHot/Equal-ramp chain
needed when exactly one non-bg color exists. (task339 18.22 -> 19.26, +1.04)
