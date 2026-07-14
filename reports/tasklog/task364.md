# task364 — e509e548

**Rule:** 1+max(H,W)//3 non-overlapping (gap>=1) GREEN boxes on a black canvas (grid H in
[10,20], W=H+[-2,2], so <=20x22). Each box is one of three sprite skeletons in any of 8
dihedral orientations: "el" (L-shape: a full col + a full perpendicular row), "aitch"
(H-shape: full left col + half right col + middle cross-row), "you" (U-shape: two full
parallel cols + one connecting row). Output recolours every green pixel by its box's shape
class: L->1 (blue), H->2 (red), U->6 (pink); background stays 0.
**Current:** 13.75 pts, custom flood (unique-label MaxPool flood + ScatterND histogram of
endpoint/turn counts), mem 75136, params 1402.
**Target tier:** detection (per-component shape classification requires component-level
aggregation = a flood; no per-pixel-local discriminator exists since a straight-arm cell is
locally identical across L/U/H).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | local-degree seeds + 3 MaxPool floods (8 iters) + uint8 Where chain | det | 76748 | 101 | 13.75 | 200/200 | correct, no gain |
| 2 | + Conv neighbour sums, drop ScatterND/int64, uint8 Where chain | det | 62060 | 65 | 13.96 | 200/200 | +0.21 |
| 3 | + per-seed iters (J=6, V/H=8), drop final flood gate, WHERE-PRIORITY drop notH | det | 55900 | 65 | 14.07 | 300/300 | **+0.317 WIN** |

## Best achieved
14.068 @ mem 55900 params 65 — adopted? N (write-only). Beats prior 13.75 by +0.317 (Y).

## Discriminator (verified 0 mismatch over thousands of fresh boxes)
Compute 4-neighbour degree on the green mask (two 3x3 Convs: vert=up+down, horiz=left+right;
deg=(vert+horiz)*mask). Three LOCAL seeds:
- J = deg>=3 (a T-junction) — only H has one.
- vend = deg==1 & vert==1 (lone neighbour vertical); hend = deg==1 & horiz==1.
Flood-MAX each seed through the mask (kernel-3 only — inter-box gap can be 1, so a bigger
kernel leaks across it; re-gate by mask each step). Per-seed iters = measured BFS reach from
that seed type to the farthest box cell: junction reach<=6, endpoint reach<=8. Classify:
- isH = Jf>0
- isU = hasV XOR hasH  (U's two tips point the SAME way -> only one orientation present)
- isL = hasV AND hasH  (L's tips are perpendicular -> both)  [chain default]
H Where applied LAST (priority) so isU needn't exclude H.

## Irreducible-floor analysis
Flood dominates: 36,080 B (J 6 + V 8 + H 8 iters x [pool+gate] fp16, last gate dropped).
Irreducible because (a) the colour is a per-COMPONENT property and a straight-arm cell is
locally indistinguishable across the 3 shapes -> aggregation over the whole component is
mandatory = a flood; (b) kernel must be 3 (1-wide inter-box gap forbids a bigger pool);
(c) measured BFS reach is 8 for endpoints, so 8 iters are necessary; (d) MaxPool requires
float, fp16 (2B) is its dtype floor (no uint8/bool MaxPool in ORT); (e) MAX cannot OR three
independent flags, so 3 separate floods are required (an H-shape shares U's (hasV,hasH)
XOR-signature, so J cannot be dropped). The two fp32 entry slices (green+bg, 3520 B) are the
Slice-dtype floor; bg is needed for the in-grid mask (off-grid-inside-the-20x22 region must
map to the 255 sentinel, not bg=0).

## OPEN ANGLES
- Reduce endpoint flood reach below 8 by seeding interior relay cells — but interior cells
  don't carry the endpoint-orientation flag, so no valid relay exists (explored, dead end).
- Single combined flood for L-vs-U via a MAX-survivable encoding — fails: MAX(1,2) loses the
  lower flag, can't recover both-present (L) vs one-present (U). 2 floods minimum.
- Eliminate the bg fp32 slice by recovering grid H,W as scalars — needs bg occupancy anyway.

## INSIGHT (transferable)
⭐ A "per-component shape classify + recolor" connectivity task is NOT at floor just because
the public net floods unique labels + ScatterND-counts: the discriminating per-component
counts often reduce to a few MAX-floodable BOOLEAN flags computed from LOCAL degree features
(junction = deg>=3; endpoint-orientation = deg==1 & (vert|horiz)==1). Flooding 2-3 booleans
(kernel-3, per-seed iters = measured BFS reach) + a uint8 Where-priority chain beats the
int64-label + ScatterND-histogram net by ~10-20 KB. ⭐ L-vs-U (both 2 endpoints, 0 junctions,
identical local stats) separates ONLY by endpoint ALIGNMENT = (hasVertEndpoint XOR
hasHorizEndpoint): U's tips share an axis (one orientation), L's are perpendicular (both).
⭐ Two free flood cuts: (1) per-seed iteration counts set to the measured BFS reach of each
seed type, not a single worst-case; (2) drop the re-gate on the FINAL flood step — leaked
gap-cell values are non-green and get discarded by the downstream green-gated Where chain
(gate the classify bools by the green mask once instead).

## 2026-06-29 verified uint8 QLinear neighbor-index rewrite

The current live/source graph was the URAD teacher overlay (`points=14.900493`,
`memory=24220`, `params=111`), structurally identical across public candidates.

Parallel read-only analysis identified the fp32 3x3 neighbor-index Conv as an
exact uint8 quantization target:

- old: `Conv(Gf, convW) -> nb`, fp32 `1x1x20x22`, then `Cast(nb)->idx`;
- new: `QLinearConv(Gu, ql_w) -> nb`, all scales `1.0`, zero-points `0`,
  with the same bit weights `[[0,1,0],[8,16,2],[0,4,0]]`.

Verification:

- source build stored eval: `points=14.956012`, `memory=22900`, `params=117`,
  stored `266/266`;
- fresh side-by-side against previous live graph: `1000` eligible examples,
  output divergence `0`, both `0` failures.

Adopted as source-owned semantic compression. This is a direct instance of the
`public_teacher_qlinear_conv_rewrite` mechanism on a local-stencil bit-index
encoder.

Follow-up cleanup reused `ql_x_scale`/`ql_x_zp` for the weight and output
scale/zero-point inputs instead of carrying four duplicate scalar initializers.
Stored eval is now `points=14.956185447372594`, `memory=22900`, `params=113`;
fresh side-by-side against the previous live graph on 500 eligible examples had
output divergence `0` and candidate failures `0`.

# (appended) S8 2026-07-02 — WALK-EINSUM WIN (+0.159) ADOPTED (candE)
14 MaxPool + 12 Mul max-propagation (~11.4KB) → TWO walk einsums on the 20×22 crop:
sprite-type features from 4-neighbor codes (V-endpoints {17,20}, H-endpoints {18,24},
T-junction {23,27,29,30}; 20000/20000: aitch⟺T-junction, el⟺V∧H endpoints, else you).
Einsum1 = 8-conn reach from T seeds (8 steps); Einsum2 = chain V-end →(mid-plane constraint
at step 10 = seedH)→ cell (20 steps) — computes reach-from-V AND component-has-H in ONE
einsum. candE extras: strided-Slice 2-ch crop, code=32·black+16·green+neighbors → single u8
Gather emits base value plane; green channel selected in-einsum via shared-letter sel[z].
18500+1131 vs 22900+113 → 14.956→15.115. Fresh 2500+5000+2000 all 0/0 div0.
Rows≠cols ⇒ two S matrices (20×20, 22×22; 884 params — params = ELEMENTS not bytes).
fp16 einsum REJECTED (0·inf NaN risk under masking). Conservative fallback cand.py (+0.142,
LB-proven ops only) kept in S8 scratchpad if candE ever hits a grader issue.

# (appended) S16 2026-07-06 — fp16 EINSUM-SUBGRAPH RECAST (+0.144) ADOPTED
Surgical mem golf on the live QLinearConv+walk-einsum net (18500/1131). Recast the whole
Einsum float subgraph to fp16: initializers sel/Sr/Sc/zf/tabV/tabH/tabT -> fp16 + inserted
GBh=Cast(GB->fp16) feeding both Einsums (GB stays f32 for the uint8/QLinearConv path).
seedV/H/T (1760->880 ea) and WLU/WT (1760->880 ea) halve; net -2640 after +1760 GBh.
mem 18500->15860, pts 15.115->15.2596. GB is a 0/1 one-hot mask; WLU/WT are bounded
fractional path-products (max 4.5e-4 / 0.28) -> no overflow, no decisive underflow.
Verified bit-identical: 266 bundled fail=0 + 2000 fresh (0 divergence) + rebuilt-source
fresh 1200/1200 fail=0. Source edit in src/custom/task364.py; params unchanged (elements).
