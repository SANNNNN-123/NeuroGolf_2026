# task192 — 7e0986d6

**Rule:** Grid (10..20 per axis) holds 3..5 SOLID rectangular boxes of one colour
`boxcolor` (wides/talls in [3,10] → every box ≥3×3) plus isolated single "noise"
pixels of a DIFFERENT colour `color` (remove_neighbors → no two noise pixels
4-adjacent; a noise pixel MAY land ON a box cell, overwriting its colour in the
INPUT, but the OUTPUT keeps box colour there). OUTPUT = the boxes only, every box
cell painted `boxcolor`, all noise deleted, off-grid → background. Closed-form
local discriminator: with occ = any-colour-present, vsum = occ(up)+occ(down),
hsum = occ(left)+occ(right): **box ⟺ occ ∧ vsum≥1 ∧ hsum≥1** (a ≥3×3 solid cell
always has both a vertical and a horizontal occupied neighbour; an isolated noise
pixel — even one sitting on a box or abutting it on one side — never has both).
boxcolor = most-frequent non-bg colour = ArgMax over channels 1..9 of pixel counts.
**Current:** 15.94 pts (ext:kojimar7113 crowd net), mem 8515, params 107
**Target tier:** B+ (single-cell-local but non-separable: box-membership is a
conjunction vsum≥1 ∧ hsum≥1, not a single linear threshold → needs 2 thresholds on
one conv plane; plus a 3-way output box-colour / in-grid-bg / off-grid).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | part-of-filled-2×2 (task193 idiom), recolor→boxcolor, 20×20 crop, uint8 tail | B+ | 11789 | 58 | 15.62 | — | correct 265/265 but crop double-counts entry (g30 3600 + g 1600); below P |
| 2 | crop the BOOL occ/ingrid instead of f32 g | B+ | 11989 | 58 | 15.60 | — | worse (two 30×30 bool planes added) |
| 3 | single 3×3 conv `6·occ+3·vsum+1·hsum`, box⟺score>9.5∧score≠12, profile in-grid, Equal→bool output | B+ | 8515 | 108 | **15.94** | **200/200** | EXACTLY matches kojimar — structural floor |

## Best achieved
15.94 @ mem 8515 params 108 — adopted? N (ties deployed kojimar net, no +0.3 gain).
Beats prior 15.94? **N (TIE / MARGINAL)**. Fresh 200/200 isolated; stored 265/265.

## Irreducible-floor analysis
Dominant = `score` [1,1,30,30] f32 = 3600B (the single 3×3 Conv on the fp32 input;
Conv inherits fp32 dtype and 30×30 extent — fp16 would need an 18000B input cast).
The remaining 5 derived full planes are ALL already bool/uint8 (900B each): `kov_b`
(score>9.5), `vgap_b` (score==12, the noise vertical-sliver collision value 12 =
6+3·2+1·0, which genuinely occurs in ~9% of instances so cannot be dropped),
`bg_or_out_u8` (in-grid-bg=9 vs off-grid=10, broadcast from row/col sum-profiles —
no 30×30 in-grid plane), and two nested-Where colour-index planes (`box_or_bg`,
`target_u8`) feeding the FREE `Equal(codes,target)→bool` output. 3600 + 5×900 = 8100
+ ~415 tiny profile/count vectors = 8515. To beat by +0.3 needs mem+par ≤ 6374
(−2249B ≈ 2.5 full planes), which the rule's structure does not admit.

## OPEN ANGLES (re-attack backlog)
- Single-threshold box detector to drop `vgap_b`: PROVEN INFEASIBLE — box configs
  {(c1,vs≥1,hs≥1)} vs noise {vs=0 ∨ hs=0} are a conjunction, not linearly separable
  by any (V,H,T) (V+H>2V ∧ V+H>2H is contradictory); any single-conv kernel collides
  noise value 12 with the box band. Separate vsum/hsum convs cost MORE planes.
- Drop `bg_or_out` by handling off-grid at the output: blocked — the Equal output
  is the FREE graph output, so it can't be post-AND-ed with an in-grid mask without
  materialising a 9000B [1,10,30,30] plane; off-grid ch0 suppression must live in
  `target`, which needs the profile-derived in-grid/outside distinction.
- Crop the score plane to 20×20: NET-ZERO — the 30×30 score still counts at 3600
  (trace) AND the 20×20 crop adds 1600, exactly cancelling the downstream savings.

## INSIGHT (transferable)
⭐ "Remove isolated noise, keep solid boxes, RECOLOUR to box colour" (noise on a
DIFFERENT colour that may overwrite box cells) ≠ task193's same-colour 2×2 rule. The
keep predicate is **occ ∧ (≥1 vertical occ-neighbour) ∧ (≥1 horizontal occ-neighbour)**,
captured by ONE 3×3 conv `[[0,3,0],[1,6,1],[0,3,0]]` over channels 1..9 read by TWO
thresholds (`>9.5` keep, `==12` exclude the vertical-sliver noise collision). boxcolor
is a scalar ArgMax of per-channel counts; in-grid vs off-grid comes FREE from row/col
sum-profiles (no 30×30 in-grid plane). This single-conv form (3600 f32 + 5×900 bool/
uint8) is the plane-count floor for a 3-way recolour output with non-separable
box-detection — already what the kojimar crowd net does. A "filled-2×2 + 20×20 crop"
re-encoding is STRICTLY WORSE here because the crop double-counts the forced 30×30
entry plane (3600 + 1600). When the entry plane must stay 30×30 and is trace-counted,
cropping never pays.

## S9 (2026-07-03) — never-materialize-30×30 crop rebuild (+0.209) ADOPTED
Old "crop = NET-ZERO" verdict refuted: that assumed downstream crop of a 30×30 score.
New: 11×11 single-tap valid Conv → occ 20×20 direct; QLinearConv (scale1/zp0 exact)
runs [[0,3,0],[1,6,1],[0,3,0]] with pads=1; all interior 20×20; target Pad fill=10 →
free Equal output. mem 8515→5745, params 106→1251 (w_occ 1210 irreducible).
Bit-identical 2500+600 uncached 0/0/0. Latency 0.095ms. Floors: occ_f 1600, target30 900.
Fallback pure-f32 variant (no QLinearConv) exists in scratch at 6545+1248 (+0.101).
Backup task192_pre_s9.onnx.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k3(cost 6996): SGD fail; k5: LP numerically infeasible(HiGHS primal infeasible). cost<7700이라 phase-2도 uneconomical. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 6996 -> 6837 (+0.023)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].