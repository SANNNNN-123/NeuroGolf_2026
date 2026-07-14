# task041 — 22168020

**Rule:** ~6 non-overlapping "downward-V / chevron" shapes (size=10 grid), each a DISTINCT colour. Each shape's two top tips share an apex row; arms slope down-inward to a 2-px base. Output fills, in every column a shape occupies, the contiguous run from the shape's apex row down to that column's lowest coloured pixel: `out[r,c]=k iff apex_k<=r<=bottom_k(c)`, `apex_k=min row of colour k`, `bottom_k(c)=max row of colour k in col c`. Distinct colours ⇒ purely per-colour (no flood-fill).
**Current (prior):** 15.63 pts.
**Target tier:** B (closed-form per-cell colour-index plane routed into the FREE bool output).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-colour [1,9,10,10] tri-MatMul below-OR ∧ above-OR | B | 16250 | 247 | 15.29 | 266✓ | exact but below prior (5× 1800B fp16 planes) |
| 2 | single-plane: nearest-below colour (suffix-max of colour+100·height) ∧ r≥apex_nb | B | 11068 | 122 | 15.68 | — | exact; shift+Max doubling chain fat |
| 3 | replace doubling chain with ONE MaxPool([N,1], pad bottom) = suffix-max | B | 8968 | 74 | 15.89 | — | exact |
| 4 | apex pipeline + downstream compares in fp16 (drop two fp32 360B + fp32 RROW) | B | 8294 | 73 | 15.97 | 200/200 | ADOPT |

## Best achieved
15.97 @ mem 8294 params 73 — beats prior 15.63 by **+0.34** (≥+0.3). fresh 200/200 isolated.

## Irreducible-floor analysis
Dominant intermediate = the 3600B fp32 channel slice `ch [1,9,10,10]` — the single fp32 entry plane that serves BOTH the colour-index collapse (1×1 conv → colf) AND the per-colour apex vector (ReduceMax). Cannot drop below 3600 because apex genuinely needs per-colour horizontal presence (the apex tips live in OTHER columns), so a single index plane is insufficient for apex. Second cost = the 900B uint8 `L30 [1,1,30,30]` carrier for the final `Equal(arange)` (Pad rejects bool, uint8 is the cheapest carrier). Everything else runs on tiny [1,1,10,10] (200B f16) / [1,9,10,1] (180B) planes.

## OPEN ANGLES (re-attack backlog)
- Could fold the apex vector out of `colf` alone (avoid the 3600B 9-ch slice) if a cheap per-colour row-presence reduction from a single index plane existed — none found without re-expanding to 9 channels (≥3600 again). The 3600 entry looks structural.
- The 900B L30 carrier: if a bool plane could be padded (it can't in opset-11), could shave ~900.

## INSIGHT (transferable)
⭐ NEAREST-pixel-in-a-direction on a SINGLE colour-index plane = pack value into units + (distance-from-far-edge) into hundreds, then a suffix/prefix MAX selects the nearest one; recover the value by `Mod(M, base)` (fp16-exact when packed < 2048). ⭐ A SUFFIX-MAX (or prefix-max) down a column over an N-tall grid is ONE `MaxPool` with an [N,1] kernel and asymmetric pad on the far side — collapses the whole log-depth shift+Max doubling chain into a single op. ⭐ A vertical "fill from a per-colour apex down to each column's pixel" is NOT connectivity: pay one fp32 entry, collapse to an index plane, and the cross-shape vertical-gap bleed is killed for free by the `r >= apex_{nearest-below-colour}` test (the gap cell's nearest-below pixel belongs to the LOWER shape whose apex sits below it).

## S11 (2026-07-03) — FLOOR CONFIRMED at 1789B (both dossier levers refuted by measurement)
- fp16 recast of color_f [1,1,10,10] fp32 (400B): REFUTED — producer Conv is fed by the fp32
  FREE input and ORT binds Conv output dtype to input dtype. Cast-after leaves the fp32 plane
  counted and ADDS 200B (1930B, worse); fp16-input Conv needs an 18000B input copy; mixed-dtype
  Conv = ORT type error. ⭐TRANSFERABLE trap: "fp16 recast" is invalid for any float plane whose
  producer consumes the free input directly (PRODUCER_BOUND class).
- task259 sub-900 carrier trick: REAL but NOT portable here. 259 builds the 10-ch one-hot at
  CONTENT resolution (3×3 → [1,10,3,3] bool = 90B) and Pads bool straight to the free output
  (bool Pad works, opset 13). Crossover rule: Equal-then-Pad costs 10·h·w vs Pad-then-Equal
  fixed 900B → Equal-then-Pad wins iff content area < 90 cells. task041 content = 10×10 = 100
  → 1000B, 100B WORSE. ⭐TRANSFERABLE rule for the full_label_plane_floor cohort.
- Toggle machinery (330B: Split/XOR/OR/Concat prefix-parity) and params 59 near-minimal.
- Control candidate (bit-identical floor reconstruction): reports/candidates/task041_signed.py,
  fresh 2000/2000, divergence 0. Frontier-queue rank #3 was a false positive (oracle char-count
  ⊥ output-carrier byte floor).
