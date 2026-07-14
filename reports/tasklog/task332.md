# task332 — d406998b

**Rule:** Grid is height=3 by W columns (W=10..20), bg=0; each column c holds one gray(5)
pixel at row rows[c]. Output keeps every pixel in place but recolours it: green(3) if
`c % 2 != W % 2` else gray(5). Equivalently green iff (c+W) is odd (column parity opposite
to width parity); the rightmost column is always green, alternating leftward.
**Current:** 16.32 pts (base net method conv1x59+b — FAILS isolated fresh ~38/40, GAP-CLOSER candidate scoring ~0 on real LB)
**Target tier:** A (closed-form, separable) — recolour is a Where-into-free-output over a separable cond.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Where(gray∧greencol, green_oh, input); W=Σcolpres; greencol=((C+W)%2==1) | A | 2912 | 61 | 17.00 | 500/500 | ADOPT |

## Best achieved
17.00 @ mem 2912 params 61 — beats prior 16.32 by +0.68 AND fully generalizes (prior did not).

## Irreducible-floor analysis
Dominant intermediate = the [1,1,30,30] uint8 cond plane (900B) needed because Where's cond
must broadcast over all 30 rows. cond is built on a tiny [1,1,3,30] active slice (grid height
is always 3) then padded; the 10-channel expansion is routed entirely into the FREE output via
Where(cond, green_onehot[1,10,1,1], input). W recovered as a scalar (ReduceSum of per-column
presence), no per-cell colour plane. fp16 Mod is integer-exact at these magnitudes (C+W ≤ 49).

## OPEN ANGLES (re-attack backlog)
- Could shave the uint8 pad plane by emitting cond as a 3-row bool directly if Where allowed
  partial-row broadcast — it doesn't, so 30 rows are required. Marginal; already tier-A clean.

## INSIGHT (transferable)
"Recolour-by-column-parity-vs-width" is closed-form separable: recover W as ReduceSum of the
per-column presence vector (one mark per in-grid column ⇒ count == width), then the parity
predicate is a single fp16 Mod-2 on a (colramp + W) ramp — no per-cell colour plane, no argmax.
The whole recolour collapses to Where(gray_mask ∧ parity_col, color_onehot, input).

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
