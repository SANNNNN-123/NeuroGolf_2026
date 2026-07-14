# task286 — b782dc8a

## Current live (WIN 2026-07-01)

`memory=41822`, `params=742`, `points=14.3412` (was 46272/741/14.2418; **+0.0994**).
Front-end restructured 2026-07-01; bit-identical to the prior incumbent
(0 divergences on 2000 fresh arc-gen + 265 bundled).

### WIN: uint8-preserving front-end restructure (−4450B)
The prior front-end cast the input to a uint8 one-hot `[1,10,30,30]` (9000B) **then
sliced** the full 10-channel plane to `[1,10,25,25]` (6250B), feeding QLinearConv
(colour idx) + ReduceMax (active) + 2 Gather (ch8, ch0). The 6250B 10-channel crop
was pure slack:
- Collapse channels **at 30×30** instead: QLinearConv→`cidx30` and ReduceMax→`act30`
  are single-channel `[1,1,30,30]` (900B each), then Slice the small planes to
  `[1,1,25,25]` (625B). This deletes the 6250B 10-channel crop.
- Derive the `≠8` / `≠0` masks from the colour index (`Equal(cidx,8/0)+Not`) instead
  of the two channel Gathers, dropping `v1`'s last two consumers.
Net: front-end ~20875B → ~16425B. Everything stays **uint8**, so the ~9-plane output
reconstruction is untouched (an fp32/int32 colour-index front-end measured 54828B —
WORSE — because the output chain balloons 1B→4B; and fp32→uint8 via QuantizeLinear is
a forbidden op). Pitfall logged: `t8` is the value **10** (out-of-grid sentinel), NOT
background colour 8 — the literal `8` needs its own uint8 init (`eight_u`).

## FLOOR on the remainder
The 9000B uint8 one-hot cast is the documented detection floor for the integer
(QLinearConv) path and is structurally required (the cheap uint8 output chain forces
an integer colour index). The bit-parallel BFS (row-packed uint32 bitsets, ~2900
4-byte scalar BitShift/And/Or ops) plus the 2×2500B pack planes and 2500B unpack are
inherent to the flood and already minimal. Lowest mem reached = **41822**.

## Original live-exact note

`src/custom/task286.py` was updated to a live-exact source builder on 2026-06-28.

## Semantic rule

Input is a cyan wall maze (`8`) with black corridors and an adjacent seed pair of two
non-cyan colours. Output flood-fills the 4-connected black component containing the seeds.
The fill colour alternates by checkerboard parity:

`out[r,c] = seed_color_for_parity[(r+c) % 2]`

Cyan remains cyan; black corridors outside the seed component remain black.

## Architecture assessment

The deployed graph is already the right low-level family: bit-parallel flood fill.
Rows are packed into 25-bit `uint32` bitsets, propagated with `BitShift`, `BitwiseAnd`,
and `BitwiseOr`, then expanded back to a 25×25 mask.

Major bottlenecks:

- full one-hot input cast: about 9000B;
- cropped 10-channel task area: about 6250B;
- bitset pack/seed/final expansion planes: about 2500B each;
- many scalar uint32 bit-propagation nodes.

## Low-level opportunities

- Source-owning the deployed `QLinearConv` colour-index extraction is done; this reproduced live
  `46272` memory and fixes the older source/live mismatch.
- Beating live materially is unlikely without a new connectivity representation. The remaining
  plausible tweak is projecting/gathering required channels before cropping the 10-channel plane,
  but expected gain is only ~1–2KB and may be offset by added maps.

Do not re-try MaxPool flood, pointer-jump connected components, or byte-chunked bitsets without a
specific new reason; they are expected to be worse under static-shape ONNX memory accounting.

# (appended) S8 2026-07-02 — WALK-EINSUM WIN (+0.574) ADOPTED (cand_v4), BEATS incumbent fresh
3037-node bit-packed uint32 BFS → 4 chained walk einsums (45+48×3 slots, Sign re-binarize after
link2). Rule: cyan walls, flood 4-conn non-cyan from non-{0,8} seeds, paint by CHECKERBOARD
(r+c)%2 parity → pair colours. t25 via single-tap 6×6 Conv (free 30→25 crop). Epilogue: label25
Conv(bias=10 sentinel), rank-1 stacked-parity einsums 'nqrc,sr,sc,q->nq', Where chain, Pad(10).
21090+2881 vs 41822+742 → 14.341→14.915. Fresh: cand 0 vs inc 20/2500 (re-run 0 vs 7/1000) —
incumbent's unrolled BFS UNDER-REACHES 0.8%; candidate margin 189 slots vs p-max 171 (80000
measured). Latency 31ms/run (free-input-repeat variants were 330ms — repeated-operand SIZE
dominates ORT einsum latency; one cheap mask plane beats shared-colour-letter when the plane
is small). params = ELEMENT count ⇒ P[30,25] projections inside einsum nearly free.

## S9 (2026-07-03) — epilogue-fold 2nd pass: FLOOR (fold refuted by 3-state output)
Fold candidate (reuse t25 as else-branch, delete label_f 2500+label_u8 625) built and
byte-verified (−2234) but FAILED stored 263/265: unreached branch is 3-state
(cyan→8, black→0, off-grid empty→sentinel) and t25 gives cyan=0, empty=0 — inseparable.
Fixing needs an occupancy plane = another 2500B fp32 Conv = exact 1:1 wash (task044).
Other floors re-priced: walk chain 4×2500 hard (189 slots, 52-letter cap ~48/einsum),
fp16 W overflow (~3e19), free-input t25 = 330ms latency fail, label_f fp32 locked.
Max theoretical epilogue win ~+0.13 and blocked. DO NOT re-probe.

# (appended) 2026-07-06 — FLOOR RE-CONFIRMED on the walk-einsum family (4 levers killed)
Deployed = walk-einsum (mem 21090, params 2881, 14.92). Measured mem split:
4 walk einsums 10000 + Sign(B2) 2500 + t25 Conv 2500 + label Conv 2500 = 17500 fp32
(seven 25×25×fp32 planes @2500B); epilogue already u8/lean (Where 1250, Pad 900,
Cast 707, Greater 645). Every fp32 plane is structurally required. Levers tried:
- **einsum merge (W1+W2, W3+W4 into single planes)**: BLOCKED by the 52-letter einsum
  alphabet. Each walk step needs a distinct contraction letter; link1=7 fixed+45 steps=52,
  link2/3/4=4 fixed+48 steps=52 — already maxed. 4-way split is FORCED, not arbitrary.
- **fp16 recast of walk planes**: BLOCKED. Path weights span 1e-28..3e19 (0.5^93 min);
  fp16 min subnormal ~6e-8, so 0.5^26→0 (empirically 0 by 96 halvings). A reached cell
  deep in a corridor underflows to 0 → Greater(W4,0) false → missed. fp32 is mandatory.
- **drop the mid Sign** (re-normalize via a non-0.5 step weight): BLOCKED. Over 189 steps
  no single per-step factor f avoids BOTH fp32 overflow (need 3f<1.6 → f<0.53) and
  underflow (need f^189>1.2e-38 → f>0.63). The Sign at step 93 is the only fix.
- **tighter crop than 25×25**: BLOCKED. 6000 fresh grids → H,W ∈ [10,25], max 25. The
  25×25 static plane must cover the largest grid; no room.
- **sparse_initializer for S(625→73)/P(750→25)/PARb(625→312)** to cut params −1590:
  BLOCKED. src/harness sanitize_model renames graph.initializer but NOT sparse_initializer,
  so node inputs → safe_name_N while sparse stays "S" → ORT load fails ("not a graph input,
  initializer, or output"). Local-breaking; won't gamble a submission (uint8-TopK lesson).
No cheaper task286 in reports/public_bundle_candidates.json. Only remaining external lever =
fresh public 400-net dump pull (Kaggle, user action) then min-merge/gate. LB stays 14.92.

## public-dump cross-check (2026-07-06) — ours WINS
Pulled the best public min-merge (prvsiyan/neurogolf-7235-49-w-visualizations, the
min-merge of ALL public sources). Its task286 = 26136 mem / 1170 params = **14.785**
(different mechanism: lower params, +5000 mem). OURS = 21090/2881 = **14.915** — we beat
it by +0.13. No public source has a cheaper 286. Confirmed global-best for this task.
