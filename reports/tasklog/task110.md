# task110 — 484b58aa

**Rule:** A `size×size` grid is a DOUBLY-PERIODIC colour tiling (colours 1..9, no
0 in-grid). FRESH instances (`generate()` with no args, the only thing genverify
exercises) use the `colors is None` branch: `v=(R²+C²)%mod+1` with
`R=(offset+row)%length−length//2`, so the ROW and COL period are BOTH `length∈{4..9}`
(rp==cp). The 4 fixed `validate()` stored examples use the `colors` branch where
row-period = `len(colors)//mod` and col-period = `mod` differ (rp∈{6,7,8,9},
cp∈{6,8,18,none}). The INPUT overlays ≤5 black (colour-0) rectangular cutouts
(≤5×5). The OUTPUT restores every black cell to its periodic colour.

**Current:** 15.46 pts, `ext:kojimar7113` (crowd net), mem 13275, params 650.
**Target tier:** B (tile-detect + re-tile Gather). Reconstruction is closed-form
(no flood/connectivity wall) — but the deployed crowd net is already near floor.

## What the kojimar net actually does (decoded from networks/task110.onnx)
- `labels_f = Conv(input,[0..9])` → colour-index plane (f32, **3600B**).
- `inv30_f  = Conv(input,[0,9,8,..,1])` → "inverted" index = (10−colour) at
  nonblack, 0 at black (f32, **3600B**). These TWO f32 planes = 7200B = 54% of mem.
- For each candidate period q∈{5,6,7,8,9}: a **dilated MaxPool (dilation=q)** of
  `labels_f` takes the max over ALL periodic copies of each residue class → the
  clean fundamental tile in ONE op (output tiny, ≤9×9). The matching dilated
  MaxPool of `inv30_f` gives max(10−colour). **Validity = ReduceMax(label_tile)+
  ReduceMax(inv_tile)==10** ⟺ every copy in that residue agrees (min==max
  uniformity). This picks the correct period as a scalar.
- Re-tile: `Gather(selected_tile, arange mod p)` rebuilds the 30×30 → output Equal.
  All per-candidate planes are ≤324B; only the two entry f32 planes + final
  gather (uint8 900B) are large.

## Reconstruction algorithm verified (numpy, my rebuild reference)
Separable dilated-MaxPool, **no inv plane needed for correctness** — detection
replaced by scalar period: row-clean MaxPool(dil=rp) → re-tile rows by `arange%rp`;
col-clean MaxPool(dil=cp) ONLY for cp∈{4..9} → re-tile cols by `arange%cp`, else
leave row-cleaned plane. **0 fails / 40000 fresh + all 4 stored.** (ks=max(1,30//p)
copies, black→−1 sentinel so black loses the max.) Confirmed buildable & exact.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior committed src/custom/task110.py (period-detect + 3-pass directional Gather fill) | B | 69860 | 1344 | 13.83 | 4/4 stored | WORSE than deployed 15.46 — never adopted |
| — | rebuild as separable dilated-MaxPool tile, drop inv30 | B | est ≥16k | — | <15.5 | — | detection planes cost MORE than the 3600 they would save (see floor) |
| 1 | 2026-06-21: FULLY-DERIVED general algo — per-axis dilated-MaxPool residue tile + inv-uniformity, LARGEST-valid period over {5,6,7,8,9}, re-tile via Mod+Gather; rows then cols | B | 73073 | 285 | 13.80 | 0/3000 fresh + 266/266 stored | EXACT & generalizes but BELOW kojimar 15.46. The general (non-hardcoded) col-period machinery floors here. |

## 2026-06-21 — robust general algorithm DERIVED (no hardcoding) but still sub-floor
The real rule is now fully reverse-engineered AND a clean ONNX build exists (0/3000
fresh, 266/266 stored, src/custom/task110.py):
- ENTRY V = Conv(input,[0..9]) fp32 3600; INV = (10−V) at nonblack.
- Per axis INDEPENDENTLY, candidate p∈{5,6,7,8,9}: residue-class max-tiles tv,ti via
  dilated MaxPool (kernel=ceil(29/p), dilation=p, stride=1 → first p positions =
  residue max). Uniformity `tv+ti==10` everywhere ⟺ class uniform&black-free.
- {5,6,7,8,9} covers EVERY observed true period 2..9 via an in-range MULTIPLE (period
  2→6/8, 4→8); pick the LARGEST valid p — robust to cutouts that fake a smaller false
  period (smallest-valid LEAKS ~0.1% fresh; verified). No valid p → axis identity (a
  doubly-periodic grid is fully recovered as long as ONE axis has a period; train#2 cp
  absent is filled by the row axis). Re-tile clean[i]=tile[i%p] via Mod+Gather.
- Decoded kojimar (networks/task110.onnx): it uses 2-D SAME-period tiles (9×9, tiny)
  for the rp==cp path + a Where priority chain picking the LARGEST of {5..9}, AND
  HARDCODES the 4 stored rp≠cp answers (`row7_public_table`,`row9_public_table`,
  is_public_pixel gate). That hardcoding is what keeps it at 13275.

WHY a general build can't reach +0.3: my separable per-axis approach pays ~20 residue
tiles of size p×29 (≈640–840B each, fp16) — kojimar's 2-D same-period tiles are 9×9
(~160B). The stored rp≠cp cases FORCE separable (or hardcoding). Even ignoring ALL
tiles, the unavoidable planes (fp32 entry 3600 + V/INV/cleaned ~1682×5) ≈ 12010 →
25−ln(12295) = 15.58 = ONLY +0.12. With the real tile cost it lands 13.80. ⇒ a
non-fitted derivation is structurally **below** kojimar; the +0.3 bar is unreachable
without the 2-D-tile-plus-hardcoded-stored hack (which is fitting → rejected).

## Best achieved
No improvement. Deployed `ext:kojimar7113` 15.46 stands. **MARGINAL — provably
cannot reach +0.3.**

## 2026-06-30 — independent re-confirmation (FLOOR)
Re-measured current src/custom/task110.py: evaluate ok=True, fail=0, mem **13155**,
params **634**, 15.468 pts. Per-tensor: two 3600B fp32 Conv planes (labels_f, inv30_f)
= 7200B (55%), 900B uint8 recolour Gather carrier, period-ladder MaxPools 100–324B each.
Intermediates already fully golfed: **0 int64** (all indices int32), 5 Casts all
load-bearing (fp32→uint8 before each table Pad/Gather), no redundant plane.
Independently re-derived the irreducibility: any per-shift correlation period-detector
must slice the 10-channel input → [1,10,29,~22] ≈ 88KB per shift (vs collapsing to 1
channel = 3600B), so the dual 1-channel value/negated-value planes ARE the cheapest
detection+validation. Native 29×29 crop impossible (Conv on the 10-ch input already
emits 3600B; slicing after only ADDS planes). Fresh periods span 2–9 (square); the
5–9 ladder covers all via in-range multiples — none droppable. Verdict unchanged: FLOOR.

## Irreducible-floor analysis (decisive)
Two mandatory f32 30×30 planes:
1. `labels_f` (3600) — the colour-index value plane; needed for the dilated
   MaxPool tiles AND the final re-tile Gather → output. Unremovable.
2. The "negated value" plane `inv30_f` (3600) — load-bearing for the MaxPool-tile
   **uniformity validation** (max(v)+max(10−v)==10 ⟺ min==max ⟺ residue uniform),
   which is how the correct period is selected. inv=10−v at nonblack is itself a
   full-grid f32 plane (Conv, or Sub+Where = same 3600B); a window slice would
   need a [1,10,18,18] input slice = 12960B (worse). Irreducible at 3600.

**The scoring math kills it even in the impossible best case:** removing inv30 with
ZERO replacement cost gives mem 9675 → 25−ln(9675+650) = **15.758 = +0.299 < +0.3**.
Any real replacement detection (per-candidate shift-compare on a window ≈ 6 q ×
2 axes × ~3 planes × ~324B ≈ 11.7KB, OR a re-projection Gather per candidate ~900B
full-grid) adds FAR more than the 3600 it saves, pushing a from-scratch rebuild to
≥16KB (worse than the deployed 13275). No detection is cheaper than the inv-plane
uniformity trick kojimar already uses. ⇒ structurally below the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- ONLY viable path: hand-edit the deployed onnx to remove inv30 AND ALSO shave
  ~700B+ from its ~6KB of small per-candidate planes — BOTH must land together to
  clear total ≤9650 (→15.76, +0.30). High effort, razor-thin margin, requires
  surgical edits to the external safe-named graph. Not a from-scratch rebuild
  (detection reintroduces the cost).

## INSIGHT (transferable)
⭐ DILATED-MAXPOOL = PERIODIC TILE RECONSTRUCTION IN ONE OP: for a doubly-periodic
in-paint, a MaxPool with **dilation = period q** takes the max over every periodic
copy of each residue class, yielding the clean fundamental tile in a single tiny
plane (black=0 loses the max, so no mask). Re-tile via `Gather(tile, arange%q)`.
Period is a static attribute → run one MaxPool per candidate q and select by a
scalar. This is the cheap escape from the plane-heavy iterative "copy ±p donor"
fill (which materialises ~14 full planes/pass). Separable (row-clean then
col-clean, col only when cp∈{4..9}) is exact (0/40000 fresh).
⭐ MAXPOOL-TILE UNIFORMITY = max(v)+max(K−v)==K (K=10 here): a "negated value"
plane lets a dilated MaxPool prove every periodic copy agrees, selecting the true
period as a SCALAR with only ≤9×9 working planes — but that negated plane is itself
a full 3600B f32 carrier and is the cheapest known period detector, so it pins the
floor at TWO f32 planes.
⭐ FEASIBILITY-MATH FIRST: when the only lever is removing ONE 3600B plane from a
~13.3KB net, compute 25−ln(mem−3600+par) BEFORE building — here it's 15.758, below
the +0.3 bar even at zero replacement cost. Bail fast.

## S8 (2026-07-02) — WALK-EINSUM WIN: 13789 → 5811 (+0.864) ADOPTED
Old "irreducible two-plane floor" (labels_f/inv30_f 3600B Convs) REFUTED under the
grader-counting lens. New mechanism (see src/custom/task110.py docstring):
- per-axis period validity = ONE einsum each → [6] conf vector (24B): 'nvxy,nwzy,pxz,vw->p'
  with stacked congruence matrices A_stack[p] (p=0 identity fallback, p=1..5 ↔ L=5..9) and
  color-difference matrix D; nonneg terms → ==0 exact.
- largest-valid one-hot gate: g = valid · [U·valid == 0], strictly-lower-tri U.
- reconstruction directly to free output: 'p,q,v,nvxy,pxr,qyc->nvrc' (input free, >0 free).
Counted outputs 264B total; params 5547 (A_stack dominates). Gates: stored 266/266 fail0;
fresh 2500+1500 div=0 vs incumbent AND 800 fresh div=0 vs real networks/task110.onnx.
TRANSFERABLE: period/tiling tasks with small selection vectors → gates-as-einsum-operands
(selection one-hot multiplies INSIDE the einsum; heterogeneous fallback stays stacked).
