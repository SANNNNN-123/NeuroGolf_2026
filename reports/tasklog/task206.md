# task206 — 88a10436

**Rule:** A connected 3x3 conway sprite (center cell (1,1) always occupied; the
sprite always spans exactly rows {0,1,2} and cols {0,1,2} because conway_sprite
never lets a row/col vanish) is drawn in colours from {1,2,3,6}. The INPUT shows
this colored sprite at location 0 plus a SINGLE gray(5) marker pixel elsewhere.
The OUTPUT keeps the colored sprite AND stamps an identical copy centered on the
marker (gray removed). The copy is a pure TRANSLATION of the input sprite by
delta = (gray_row - center_row, gray_col - center_col), where center = (min
colored row + 1, min colored col + 1). Generator forces |dr|>=4 OR |dc|>=4 so the
two copies never overlap -> output = per-cell MAX(colour plane, shifted plane).
**Current:** ~13.98 pts (tier A, stored), target P=17.0
**Target tier:** B (data-dependent translation) — output colour per cell is a
data-dependent shift of the input; the shift is realized with two boolean MatMuls
(Srow @ Lc @ ScolT). Not Tier A because the shift offset is per-instance, not a
fixed separable row⊗col stamp.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | double-MatMul translate, WORK=12 canvas, full 10-ch colour Conv, no in-grid mask | B | 8948 | 73 | 0 | fails | bg leaked into out-of-grid cells inside WORK window |
| 2 | + in-grid mask via [1,10,W,W] ReduceMax | B | 15572 | 84 | — | — | mask plane 5760B dominated |
| 3 | + in-grid as rowprof/colprof rectangle (1-D ReduceSum of free input) | B | 9596 | 79 | 266/266 | — | correct; Vbig 3600 + L 900 dominate |
| 4 | 6-ch slice [1,6,W,W] colour Conv + gray fused (weight 50) | B | 8972 | 70 | 266/266 | 200/200 | BEST — fp16 shift, single gray/colour Conv |

## Best achieved
15.89 pts @ mem 8972 params 70 — adopted? N (orchestrator gates). Beats prior
13.98 by +1.9. Below target P=17.0 (MARGINAL vs P, solid vs stored).

## Irreducible-floor analysis
Two unavoidable structures dominate: (a) one colour-extraction plane touching the
colour channels — cheapest is the [1,6,12,12] f32 channel-cropped slice = 3456B
(a full 30x30 Conv is 3600B; both within ~150B); (b) the final 30x30 uint8 label
L = 900B, required because the output Equal must run on a 30x30 plane (Pad rejects
bool, so the label cannot stay at WORK size). The fp16 shift path (Lc/Srow/ScolT/
rowshift/Lshift/Lboth = 6×288) adds ~1700B and is intrinsic to a data-dependent
translation. Sum of all these ~= e^(25-15.9). No reformulation removes both the
30x30 colour plane and the 30x30 output label simultaneously.

## OPEN ANGLES (re-attack backlog)
- Split colour slice into ch1:3 (1728B) + ch5:6 (1152B) = 2880B in two planes to
  shave ~576B from in16; offset by extra Conv/Add planes (tried mentally, ~wash).
- Tier A long-shot: if the sprite shape+colours could be stamped via a fixed
  separable row⊗col onehot routed into the free output, L could vanish — but the
  sprite is an arbitrary conway shape with per-cell random colours, so it is NOT
  a separable rectangle (Tier A blocked, per the playbook's separability note).
- Investigate whether the two copies can be written directly into the free output
  via Where(shifted_onehot, ...) without materializing L — blocked by Pad-on-bool.

## INSIGHT (transferable)
Data-dependent pure TRANSLATION = the task250 double-boolean-MatMul idiom with
SHIFT matrices Srow[R,r]=(r+dr==R), ScolT[c,C]=(c+dc==C) instead of clamp
matrices; fp16 is exact for colour values <=9. "Copy a sprite onto a marker"
collapses to MAX(colour_plane, shifted_colour_plane) when the generator
guarantees non-overlap (here |dr|>=4 OR |dc|>=4). ⭐ An origin-anchored full
rectangle in-grid mask is recoverable for FREE from 1-D row/col occupancy
profiles: ReduceSum(input, axes=[1,otherspatial]) -> [1,1,30,1]/[1,1,1,30]
(120B each) -> Greater -> And, instead of a [1,10,W,W] channel-max plane (5760B).
A single colour Conv can ALSO encode a sentinel marker channel (gray weight=50)
so colour label and marker position both come from ONE plane (>9 = marker).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.003)
**Mechanism (op-census diff):** Dropped one `Unsqueeze` + its `unsq_col` [1] const (shape-op fusion). −12B, −1 param.
**Old→new:** mem 4122→4110, params 77→76.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task206_pre_s10.onnx`; source `public_candidates/bobmyers7186/task206.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 4186 -> 3766 (+0.106)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].

## S17 (2026-07-06) — ADOPTED from udit 7237.17 dump (NEW uploader re-mine): 3766 -> 1795 (+0.741, points 16.766->17.507)
Gate fresh_verify 1500: **inc=0, cand=0, divergence=0** (clean pass, safe for private LB). Source-owned via `live_to_exact_source 206 --write-src`, re-measured fail=0 (mem 1660 + params 135).
⭐ TRANSFERABLE MECHANISM: udit dropped the whole double-boolean-MatMul-translate + 6-ch colour Conv slice +
30x30 label approach (OLD 57 nodes: Conv + ScatterND + 4×ArgMax + 6×ReduceMax multi-branch detector) and rebuilt
the data-dependent sprite TRANSLATION as **RoiAlign crop off the FREE input + 5 Einsum contractions** (NEW 31 nodes:
RoiAlign×1, Einsum×5, Round×2, Div×2). The sprite is cropped once via RoiAlign (value_info-legalized bbox off the
free fp32 input, urad mech 1/2 family) and the shifted stamp is realized by Einsum contraction rather than
materializing a 30×30 boolean shift-matrix + label plane. This BEATS the "no reformulation removes both the 30x30
colour plane and the 30x30 label" irreducible-floor claim above (§Irreducible-floor) — RoiAlign+Einsum removes BOTH.
Generalizes to other data-dependent-translation / sprite-stamp tasks currently paying for a full 30×30 label + shift
matrix. See [[neurogolf-urad-7225-bundle-vein]] (mechs 1-3), [[neurogolf-bilinear-einsum-lever]].