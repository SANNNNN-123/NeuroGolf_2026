# task252 — a5f85a15

**Rule:** A size×size grid (size 3..15) at the top-left of the 30×30 canvas. Anti-diagonals {r−c in diags} are painted one `color`; rest is black. OUTPUT keeps `color` on EVEN columns of those painted cells but flips ODD-column painted cells to YELLOW (4); background unchanged. So output == input everywhere except painted cells at an odd column → yellow.
**Current (prior):** 16.30 pts, 15×15 bool masks + runtime-weight 1×1 Conv decode (g/se/so → e0/e_C/e4), mem 5970, params 62. The fp32 [1,3,15,15] Conv input (2700B) dominated.
**Target tier:** A (route the 10-ch expansion into the FREE Where output; no full-canvas value plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp16 Conv decode | — | — | — | — | — | INVALID: 10-ch Conv result must be the fp32 free `output`; fp16 forces a counted 18KB intermediate + Cast. Conv input must stay fp32. |
| 2 | Where(cond30, yellow, input), cond built on 15×15 patch + zero-Concat to 30×30 | A | 3705 | 713 | 16.607 | 200/200 | works |
| 3 | + fold odd-col into the column occupancy vector (drop a 15×15 plane) | A | 3495 | 713 | 16.655 | 200/200 | **adopted** |
| 4 | ConstantOfShape zero pads | A | 4380 | 46 | 16.605 | — | no gain (just moves 675 from params→mem; score adds them equally) |
| 5 | uint8/fp16 Pad instead of Concat | A | — | — | — | — | uint8 rejected by opset-10 Pad; fp16 Pad costs 1800 (worse than Concat). |

## Best achieved
16.655 @ mem 3495 params 713 — adopted? Y. Beats prior 16.30? Y (+0.36, > +0.3).

## Irreducible-floor analysis
Dominant intermediates: (a) **900B fp32** ch0 slice `input[:,0:1,:15,:15]` — the only per-cell 2D background readout; the painted mask is a diagonal (NOT row⊗col separable) so it cannot be recovered from cheap 1-D occupancy profiles, and Slice preserves the fp32 input dtype → 900B is the floor for a 2D fp32 read of the 15×15 active region. (b) **900B bool** cond30 — Where requires a [1,1,30,30] condition broadcastable against the free [1,10,30,30] input/output. (c) 450B condW (mid-Concat). The 675 zero-pad elements are unavoidable (the 30×30−15×15=675 padded cells live in either params or mem — the score adds the two equally, so moving them doesn't help; only ELIMINATING the pad would, which needs a full-30 cond from the start whose fp32 readout is 3600B — strictly worse).

## OPEN ANGLES (re-attack backlog)
- Eliminate the 900B fp32 ch0 slice by reading background from a cheaper signal — not found: a diagonal mask is non-separable so 1-D profiles can't reconstruct it, and any 2-D fp32 read is ≥900B. Likely at floor for this structure.
- Build cond30 directly without the 15×15→30×30 Concat (would drop condW 450 + zero-pad 675) — blocked because the only full-30 background readout is a 3600B fp32 plane.

## INSIGHT (transferable)
A pointwise recolor that ONLY changes a sparse set of in-grid cells = `Where(cond30, color_onehot[1,10,1,1], input)` routed into the FREE output; build `cond` on the small ACTIVE patch (generator size bound, here 15×15) then expand to 30×30 by **zero-Concat of bool blocks** (Pad rejects bool, and opset-10 Pad rejects uint8; fp16 Pad doubles the plane). ⭐ Fold a per-column predicate (odd-column) straight into the 1-D column-occupancy bool vector before the row⊗col And, killing a whole 15×15 intermediate plane. NOTE: a 10-channel Conv decode that lands in the free fp32 `output` CANNOT be made fp16 — the output must be fp32, so any fp16 Conv forces a counted ~18KB intermediate; prefer the Where-into-free-output idiom over Conv-into-free-output whenever only a sparse recolor is needed.
