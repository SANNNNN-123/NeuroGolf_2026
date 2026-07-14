# task198 — 83302e8f

**Rule:** size×size cell grid, each cell minisize×minisize px, cells separated by 1-px
`color` lines (pitch p=minisize+1, actual_size=size·p−1≤29). Input: black canvas + color
grid-lines, some line pixels punched back to black ("permeable points"). Output: green
canvas, same lines; every permeable point → YELLOW(4); a cell interior → YELLOW(4) if ANY
of its 4 walls has a gap, else GREEN(3). DEPTH-1 (each gap marks exactly the 2 cells it
separates) — NOT a transitive flood. (Memory's "task198 = flood wall" referred to our OLD
flood net; the actual ARC task is closed-form & separable.)

**Current (deployed):** 15.17 pts, ext:kojimar7113 (crowd MaxPool×8 net, f32→f16→bool entry
triplication), mem≈18600.
**Target tier:** A/B closed-form (separable cell-mark via selector MatMuls + Gather upsample).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| pre | existing custom (fp16 everywhere) | B | 30460 | 135 | 14.67 | — | below P |
| v2 | uint8 carriers + uint8 Equal output | B | 25206 | 136 | 14.86 | — | below P |
| v3 | shared isblack16, separable gap downsample | B | 21786 | 140 | 15.00 | — | below P |
| v6 | double-Gather uint8 upsample (kills fp16 1800 upsample plane) | B | 19487 | 143 | 15.12 | 200/200 | **best, still <P** |
| v9/11 | sentinel-interiorL to drop gridL/ingrid | B | 19701/19821 | — | 15.10 | — | net-worse (online needs 3 planes; line spans orthogonal off-grid axis) |

## Best achieved
15.115 @ mem 19487 params 143 — adopted? N (do-not-adopt). Beats prior 15.17? **NO** (−0.05).

## Irreducible-floor analysis
Three structural plane groups, all required for an EXACT net:
1. **Entry blackness plane = 3600B fp32** (Slice ch0). Required because the OUTPUT marks every
   permeable point (black-on-a-line, arbitrary (r,c)) YELLOW → a full 2-D `isblack∧online`
   plane is unavoidable. Conv on fp32 input yields fp32 (probed: declaring fp16 output →
   ORT type error); Cast(input→fp16) is 18000B. So one 3600B fp32 entry is the floor.
2. **Cell-mark gap pipeline ≈ 4320B**: isblack16 fp16 (1800, shared) + 6×[1,1,S,30] fp16
   selectors/matmuls (2520). fp32 variant (use ch0 directly, no isblack16) costs MORE
   (6×840 fp32 small > 1800+6×420). The gap "any-gap-in-cell" aggregation genuinely needs
   the cell-space reduction (separable double-MatMul Rsel@isblack16@Csel).
3. **Compose = 6 full planes ≈ 5400B**: online_b, ingrid_b, interiorL, lineL, gridL, L
   (all uint8/bool 900). Cannot drop below 6: `online=(rline|cline)&ingrid` needs 3 full
   planes because a line on an in-grid row STILL spans off-grid columns (per-axis in-grid
   folding into rline/cline is INCORRECT — verified it leaks colour into off-grid). The
   sentinel-interiorL trick (off-grid→99 via padded Gather) lets the L-gate drop but the
   correct `online` then needs onv+onh+or = 3 planes → wash.
Total ≈ 19.5K → 15.12, structurally ~0.05 BELOW the kojimar MaxPool net (15.17) and +0.35
below the +0.3 win threshold (needs mem ≤ 13772 ⇒ would have to delete the fp32 entry AND
halve both the cell-mark and compose — not possible for an exact net).

## OPEN ANGLES (re-attack backlog)
- Gather-downsample the gaps at WALL columns only (wall_cols=(i+1)·p−1 via runtime ramp) to
  shrink the 30-wide gap intermediates to S-wide — saves maybe ~1KB of small planes, NOT
  enough to clear +0.3.
- Match kojimar's localized-MaxPool-fill formulation (avoids the cell-space round-trip) — but
  that net is ALREADY 15.17; replicating it can't beat it by +0.3.
- The +0.3 gap is structural: an exact closed-form net for this rule floors ~15.1-15.2.

## INSIGHT (transferable)
⭐ **Gather-upsample beats MatMul-upsample for a uint8 cell→pixel expansion**: a cell-space
value plane interiorCell[S,S] upsamples to pixel space as `Gather(Gather(cell,Ridx,ax2),Cidx,ax3)`
with int32 index vectors — Gather PRESERVES uint8, so the [1,1,30,30] result is 900B and the
only full plane, vs the double-MatMul which forces a 1800B fp16 plane + a 900B uint8 cast
(task198 v3→v6: 21786→19487, +0.12). Clip indices in fp16 (Clip rejects int) then Cast int32.
⭐ **uint8 Equal/Where/Gather all run under ORT_DISABLE_ALL** (probed) — declare the final
colour-index carrier uint8 (900B) feeding `Equal(L_u8, chan_u8)→bool output` (free); 2× cheaper
than the fp16-Equal carrier.
⭐ **Per-axis in-grid folding into line masks is INCORRECT when lines span the orthogonal axis**:
`online=(rline&ringrid)|(cline&cingrid)` leaks colour at (in-grid-line-row, off-grid-col)
because rline alone is true there. A correct in-grid line gate needs the full cross-axis AND
(3 planes). Watch this on any line/grid task with an in-grid mask.
⭐ Memory's "task198 = flood wall (infeasible)" was a FALSE label tied to our old flood net —
the real ARC task is closed-form depth-1 wall-gap marking, fully separable. Re-triage
"flood/connectivity wall" labels against the GENERATOR, not the deployed net's op signature.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.000)
**Mechanism (op-census diff):** Zero-masking recast from **uint8-`Mul`** (`zero_u8`) to **fp16-`Where`** (`where_mask_zero_f16`): Cast 11→9, Mul 6→3, Where 1→4. −6B.
**Old→new:** mem 13434→13428, params 138→138.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task198_pre_s10.onnx`; source `public_candidates/bobmyers7186/task198.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 12806 -> 12298 (+0.040)
Mechanism: value_info Slice crop + CumSum.
Gate (fresh_verify, inc/cand fail on 1500-2000): 0/0 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 

## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 12298 -> 11412 (+0.075)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].