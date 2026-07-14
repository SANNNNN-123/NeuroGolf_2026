# task008 — 05f2a901

## 2026-06-29 mechanism screen

Rule: move red pixels relative to cyan marker structure under flip/transpose
variants.  The current graph uses fixed channel slices and compact geometric
logic rather than a full colour-index plane.

Current source score: 16.365306 @ mem 5561 params 65.  Dominant tensors are the
two fixed-colour slices (`c2f`, `c8f`) and a 30x30 label/mask carrier; the palette
is fixed enough that direct channel slices are cheaper than a generic 1x1
colour-index Conv.

No rewrite adopted.  This is not a good transfer target for colour-LUT mechanisms:
the two relevant colours are fixed, and the remaining cost is geometric routing
plus full-canvas output conditioning.

## 2026-06-30 S1 — FLOOR confirmed (deep structural re-route attempt failed)
Hypothesis: colours fixed (red/cyan), cyan unchanged, red rigidly translates → route cyan
from FREE input + native-cropped red plane, drop the 30×30 carrier (target ~+1pt). Built a
correct from-scratch geometric closed form: oracle 266/266, bundled fail=0, fresh 2500/2500
== incumbent (fully generalizing). But it measured 5938B > incumbent 5561B.
**Floor proof (per-tensor, irreducible):** 2×16×16 fp32 slices = 2048B (red nibble holes need
the exact mask read; no slice-with-cast — uint8/bool mask requires materialising the fp32
slice first; cyan locator likewise); ONE 30×30 uint8 carrier = 900B (cheapest one-hot
expansion is Equal(carrier[1,1,30,30],chans[1,10,1,1]); Concat/Scatter alternatives are
WORSE — a [1,10,30,30] base = 9000B); 7×256B working planes = 1792B INCLUDING the **channel-0
background ingrid-rectangle plane** (the "route everything via free input" idea overlooks that
output ch0 must be 1 across the whole in-grid rect → forces the carrier + ingrid plane);
+~820B strips/scalars = ~5561 = incumbent. VERDICT: incumbent (ext:biohack) already optimal,
no per-task lever. No change.

## 2026-07-01 sequential deep pass

Fresh recheck: **1000/1000 pass**.

Memory profile still matches the prior floor analysis:

- `c2f`, `c8f`: two `[1,1,16,16]` fp32 channel/spatial slices = **2048B**.
- `lab30`: final scalar label carrier = **900B**.
- 16x16 mask/route planes (`m2`, `m2b`, `m8b`, `base`, `lab2`, `lab`,
  `mvr`) = **7 x 256B**.

Rechecked possible substitutions:

- Fixed colours mean generic colour-index or task001-style colour factorization
  is not useful; direct channel slices are cheaper.
- `Gather` channel-first would materialize a 30x30 fp32 channel plane before
  cropping, worse than the current precise 16x16 `Slice`.
- Replacing the scalar label carrier with 10-channel construction is much larger.

Conclusion unchanged: no adoptable improvement found.

## 2026-07-03 cyan profile rewrite — ADOPTED

Human review questioned whether the cyan 2x2 marker needs a full `[1,1,16,16]`
channel slice.  It does not: the marker is always a solid 2x2 block, so row and
column occupancy profiles are enough to recover both the locator and the cyan
overlay mask.

Adopted source-owned rewrite:

- remove `c8f [1,1,16,16]` fp32 slice (**-1024B**) and its direct bool cast;
- compute cyan row/column profiles from the free full input with two `Einsum`
  contractions against `ch8f [10]`, `ones30 [30]`, and a singleton;
- crop those profiles back to 16 and rebuild the cyan 2x2 mask as
  `Cast(rp8) AND Cast(cp8)`;
- keep the red 16x16 slice and label-carrier output path unchanged.

Stored verification: **266/266 pass**,
`memory=4809`, `params=104`, `points=16.50035996783135`.
Fresh generator verification: **3000/3000 pass**.

Net improvement over the previous deployed source:
`memory 5561 -> 4809`, `params 65 -> 104`, total counted cost
`5626 -> 4913`, score `16.364846 -> 16.500360` (**+0.1355**).

Border-crop note: dropping a fixed two-cell border is not safe.  Over 20k fresh
samples, cyan and red input/output bboxes each reached all four borders after
the generator's flip/xpose transforms.  The persistent 16x16 crop remains needed.

## 2026-07-05 S15 — mixed-axis floor CONFIRMED; "cost 148" is the single-axis-Gather class (NOT 008)

User relayed a high-scorer's claim that task008 is doable at cost 148 (=20 pts).
Investigated exhaustively.

**Rule re-locked (265/265):** red(2) rigid-translates along ONE axis until edge-adjacent
to the fixed cyan(8) 2×2; colours always {0,2,8}. Move axis is per-example V (137) or
H (128) — MIXED. Output IS a pure single-axis permutation of the input rows (vertical)
OR cols (horizontal): `output = input[rowperm][:, colperm]`, one perm identity (verified).

**The "148 mechanism" is REAL but single-axis-only.** For ONE axis,
`output = Gather(input, perm[30], axis)` is one node; input/output are FREE; only the
int32 index (120B) + scalars materialise → cost ~136-140 = 20 pts. This is exactly
what task150 (mem 136, 20.07) and task155 (mem 140, 20.04) already do. Gather applies a
permutation with a [30] INDEX, not a [30,30] matrix — that is the whole trick.

**task008 is the mixed-axis blind spot — measured every route, all worse than 4809:**
| approach | cost | pts |
| single Gather (one axis, static perm) | 30 | 21.6 (but can't: needs both axes) |
| einsum 2 perm-matrices, STATIC param | 1800 | 17.5 (can't: perms are data-dependent) |
| einsum 2 perm-matrices, COMPUTED (fp32) | 9480 | 15.8 |
| two-Gather chain (row then col) | 36060 | 14.5 (intermediate [1,10,30,30]=36000) |
| colour-plane collapse + 2 Gathers + Equal | ~6300 | ~16.0 (einsum collapse alone = fp32 3600) |
| **current crop-based net** | **4913** | **16.50** |
Root cause: a data-dependent 2-axis separable reindex has NO cheap ONNX primitive.
Gather (cheap [30] index) is single-axis; einsum needs [30,30] matrices that can't be
static (data-dependent) so they count as fp32 intermediates; any 2-Gather chain or
channel-collapse materialises a ≥3600B plane. 148 params cannot encode a rank-30 perm.

**Verdict:** task008 stays FLOOR at 4809. The 148/20 figure is a task-number confusion
with the single-axis-Gather class (150/155). No change adopted. New reusable artifact:
`reports/scripts/perm_axis_scan.py` (finds single-axis Gather-feasible tasks).


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 4913 -> 3241 (+0.416)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].
## S16 (2026-07-06) — public bit-identical golf (llccqq624) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
