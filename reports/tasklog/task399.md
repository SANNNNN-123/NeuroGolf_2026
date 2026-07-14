# task399 — ff28f65a

**Rule:** Input is a size×size grid (size 3–7) holding `num_boxes` (1–5) non-overlapping
red(2) 2×2 squares. Output is ALWAYS a fixed 3×3 grid whose blue(1) cells turn on by a
count-only schedule n=num_boxes: n≥1→(0,0), n≥2→(0,2), n≥3→(1,1), n≥4→(2,0), n≥5→(2,2);
all other 3×3 cells black(0); everything outside the 3×3 unset. Boxes never overlap so red
pixel count = 4·n exactly, ⇒ n = ReduceSum(red)/4 (integer-exact in fp32).
**Current:** 17.92 pts (public import wguesdon6304: ReduceSum→Sub→Abs→ReduceSum→Neg→ArgMax→Gather→Conv)
**Target tier:** S/closed-form — output is data-INDEPENDENT given the scalar count, fully built from constants.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | n=Σred/4 → 3×3 sched, lab plane Pad(sentinel10)+Equal, int32 | S | 7262 | 50 | 16.10 | 200/200 | full 30×30 int+fp32 planes |
| 2 | per-ch count (no 30×30), lab fp16 Pad+Equal | S | 1884 | 50 | 17.43 | 200/200 | 1800B fp16 lab dominates |
| 3 | uint8 lab Pad+Equal (one-hot {0,1}) | S | 975 | 50 | 18.07 | 200/200 | 900B uint8 lab dominates |
| 4 | build [1,2,3,3] uint8 one-hot, Pad INTO free output (ch+spatial pad) | S | 102 | 40 | 20.04 | 500/500 | ADOPTED |

## Best achieved
20.04 @ mem 102 params 40 — adopted? Y. Beats prior 17.92? Y (+2.12).

## Irreducible-floor analysis
Dominant intermediate is the `cnts = ReduceSum(input,[2,3])` [1,10,1,1] fp32 = 40B (ORT
ReduceSum rejects uint8/bool so this must be fp32). Everything else is ≤9-element 3×3
tensors. No 30×30 plane is ever materialised: the final Pad expands the tiny [1,2,3,3]
uint8 one-hot directly to the [1,10,30,30] graph output (free), padding both the 8 trailing
colour channels and the spatial border with 0. Near practical floor for a count-parametric task.

## OPEN ANGLES (re-attack backlog)
- Could drop the 40B cnts plane by counting red via a 1-D Conv/MatMul that emits a scalar
  directly, but at mem 102 the score gain (ln) is negligible (<0.4) — not worth complexity.

## INSIGHT (transferable)
⭐ COUNT→FIXED-PATTERN tasks (output depends only on a scalar count, content is constant):
build the tiny K×K one-hot from constants gated by a single Greater(schedule, n) threshold,
then **Pad the small [1,fewch,K,K] uint8 one-hot DIRECTLY into the free [1,10,30,30] output**
— pad both the trailing colour channels AND the spatial border with 0 in ONE Pad whose output
IS the graph output. No carrier/label plane is ever materialised (mem → ~100B). Pad accepts
uint8 (rejects bool); declaring the output uint8 is fine since the harness scores (out>0).
This is escape (1)+(2) combined: a constant-content output is pure Tier-S, no per-cell plane.
