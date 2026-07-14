# task106 — 46442a0e (C4 rotational symmetrization of a small grid)

**Rule:** INPUT is a size×size grid (size∈{2,3}) of nonzero colours (1..9) at the TOP-LEFT.
OUTPUT is 2size×2size = the C4 (4-fold rotation) orbit of the input about the N×N centre
(N=2size): each cell (r,c)→{(r,c),(c,N-1-r),(N-1-r,N-1-c),(N-1-c,r)}. Equivalently OUTPUT =
OR over {G, rot90, rot180, rot270} where G = input placed top-left of an N×N grid; rot90 ccw =
F_N @ Gᵀ (F_N = N×N anti-identity). All colours nonzero ⇒ active region is exactly the top-left
6×6 (≤3 size). The four quadrants are disjoint (size×size each) so OR == max == sum.

**Current (prior stored):** 17.94 pts, ext:kojimar6275, mem 1039, params 125.
**Target tier:** A/B (data-dependent rotation). The STORED net is already an optimal closed-form
**constant GatherElements**: Slice→3×3 [1,10,3,3], ArgMax→colour-index, flatten to a 10-elem value
vector (9 cells + appended bg-0), then ONE GatherElements with a [1,1,36] permutation index that
realises the entire C4 symmetrization — TWO precomputed perm tables (size=2 / size=3) selected by a
size flag (the 144B int32 Where). Equal→one-hot [1,10,6,6] bool→Pad→output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf Conv on 6×6 slice; rot90 = F_N@Tᵀ ×3 MatMul, Max-OR; Equal→Pad bool | A | 12690 | 66 | 15.55 | — | works, heavy |
| 2 | sentinel-bg, single-channel uint8 Pad→Equal | A | 3654 | 57 | 16.78 | — | trim |
| 3 | 6×6 uint8 one-hot → Pad as FREE uint8 output | A | 3438 | 57 | 16.84 | 200/200 | best (mine, old) |
| 4 | DIRECT uint8-one-hot gather: slice 3×3 f32→u8→flat[1,10,9], Where-select [6,6] perm idx, Gather axis=2 → [1,10,6,6] direct → Pad | A | 1085 | 99 | 17.92 | 500/500 | best, ties stored |

## 2026-06-19 re-probe (uint8 whole-pipeline angle)
NEW best net `src/custom/task106.py`: **17.92 @ mem 1085 / params 99, fresh 500/500**. Improves the
distribution massively over my old MatMul attempts (16.84 → 17.92, +1.08) by following the task152
uint8-one-hot lever: NO ArgMax, NO colour-index plane, NO Equal-expansion — gather the uint8 one-hot
*directly*. Pipeline: Slice input→[1,10,3,3] f32 (the ONLY fp32 plane), Cast→uint8, Reshape→[1,10,9],
size-flag Where over two [6,6] int32 perm tables → idx, ONE Gather axis=2 with the [6,6] index emits
[1,10,6,6] DIRECTLY (output=data[:axis]+indices.shape, so no reshape plane), Pad→free uint8 output.
Background output cells (size=2's outer ring) route to 3×3 source cell 8 (always empty for size=2) →
no zero-column needed. But it only TIES the stored 17.94 (Δ≈−0.02), does NOT beat by +0.3.

## ⛔ +0.3 INFEASIBLE — hard floor proof (864B > 862B budget)
Target 18.24 requires mem+params ≤ ~862. Three planes are each individually irreducible for ANY
copy/permutation encoding and SUM to 864 in mem ALONE (before cast/reshape/flag overhead + params):
1. **fp32 entry [1,10,3,3] = 360B**: Slice preserves fp32; ALL 10 channels needed (one-hot copy);
   3×3 is the min source extent (size≤3). Casting the full input first = 9000B. Irreducible.
2. **uint8 one-hot output plane [1,10,6,6] = 360B**: Pad needs a 4D input; the 10-ch one-hot at the
   6×6 working canvas (min for size=3) is 360B even in uint8. Gathering straight to [1,10,30,30] (free
   output, no Pad) needs a [30,30] int32 index = 3600B — far worse. Irreducible.
3. **int32 Gather index = 144B**: ORT Gather REJECTS int8/int16/uint8 indices (re-tested 2026-06-19,
   ShapeInferenceError); int32 is the floor; a [6,6] index = 36×4 = 144B. The size flag is MANDATORY —
   the two C4 perm tables genuinely differ (size=2 fills a 4×4 orbit, size=3 a 6×6 orbit; no fixed perm
   or N-arithmetic unifies them — diff matrix has no clean structure), so the index is a RUNTIME Where
   output (a counted intermediate), not a static init.
360+360+144 = 864 > 862 ⇒ no copy/gather encoding can reach +0.3. The stored 17.94 is at floor.

## Irreducible-floor analysis
The STORED net is at floor. Two ~360B planes dominate and are both structural:
(1) entry 10-channel slice [1,10,3,3] fp32 = 360B (ArgMax needs the multi-channel float; Slice
preserves fp32; 3×3 is the minimum that covers size=3); (2) the one-hot expansion [1,10,6,6] bool
= 360B (the 10-ch reduce/expand floor at the 6×6 working canvas). Plus the GatherElements index
machinery (~150B) + 125 params (two 36-elem perm tables + selector). Total 1164. To beat by +0.3
needs ≤862B; the two 720B floor planes alone forbid it. The ONLY loose piece is the size-select
Where (144B int32 [1,1,36] + ~36 params); a hypothetical single merged perm would save ~180,
landing ~985 → 17.99 (+0.05), still far short of +0.3. The two perm tables genuinely differ
(size=2 output is 4×4 with bg fill, size=3 is full 6×6) so they cannot be merged into one gather.

## OPEN ANGLES (re-attack backlog)
- Merge the two C4 perm tables into ONE size-agnostic gather (eliminate the size-flag Where). The
  obstacle: size=2 vs size=3 output extents differ (4×4 vs 6×6), so a fixed perm over the value
  vector cannot route both — would need a size-parametric index ARITHMETIC (e.g. clip/offset by N)
  that is itself ≥ the Where it replaces. Likely net-neutral, untried in detail. Even if free: +0.05.
- Shrink the [1,10,3,3] entry: impossible below 360B (ArgMax needs ≥ all colour channels over the
  3×3 region; fp32-forced by Slice).

## INSIGHT (transferable)
⭐ A fixed-position SMALL-grid C4/Cn symmetrization where input & output are pure colour COPIES is
NOT a MatMul-rotation task — it is a **constant GatherElements**: flatten the K×K input to a value
vector (+1 appended bg slot), precompute the [1,1,(2K)²] permutation index that maps each output
cell to its source cell under the rotation orbit, and emit the whole symmetrized grid in ONE gather
(NO rotation planes, NO transpose chain). For a variable grid size, select among per-size perm
tables with a size flag. This beats the F_N@Tᵀ MatMul approach by ~1.1 pts because it removes all
6 rotation intermediates. The MatMul/reverse-transpose idiom (task027/112) is for DATA-DEPENDENT-
centre rotations on the full canvas; a FIXED top-left small grid collapses to a constant gather.

## S10 (2026-07-03) — kojimar 7185.95 teacher ADOPTED (+0.328)
**Mechanism swap — fold the Slice+Conv colour-read into ONE input-contracted Einsum.** The rest of the
pipeline is byte-identical (Cast→Reshape→3×Gather→Where→Cast→Cast→ScatterElements, same idx2/idx3 [6,6]
int32 C4 perm tables). Only the FRONT changes: OLD did `Slice`(input→[1,10,3,3] fp32, initializers
patch_starts/ends) then `Conv` with color_w[1,9,1,1] to read the colour index over the 3×3 patch. NEW does
ONE `Einsum` `bchw,c,hr,wk->brk` contracting the FREE input directly against c_1[10] (colour weights
0..9) and c_2[30,3] used TWICE (as `hr` row-selector and `wk` col-selector, each picking the 3 top-left
rows/cols out of 30) → emits the [1,3,3] colour-index patch directly. **The counted [1,10,3,3] fp32 Slice
intermediate (360B) is eliminated** — the Einsum contracts the free input straight to a [1,3,3] result.
Cost: c_2[30,3]=90 floats added as a STATIC initializer. **The +81 params trade:** new params 174 vs old 93;
the front-end initializers {color_w 9, patch_starts 4, patch_ends 4, flat_shape 3, probe_idx 1 = 21 elems}
are replaced by {c_1 10, c_2 90, c_5 1, c_7 1 = 102 elems}, net +81, essentially the c_2[30,3] projection
table — i.e. the teacher BOUGHT an 81-param static selector matrix to DELETE a 324B counted working plane
(mem 776→452 −324, params 93→174 +81, sum −243). pts 18.233→**18.561 (+0.328)**.
**Gates:** bundled fail=0; fresh arc-gen 2×2000, 2000/2000 valid, inc_fail=0 cand_fail=0; no TopK/uint8
offenders (Gather/ScatterElements idx already int32); NON-CACHED, orchestrator-reverified. Backup
reports/retired_networks/task106_pre_s10.onnx. Provenance public_candidates/kojimar7185_95/overrides/task106.onnx.
⭐ TRANSFERABLE: a `Slice`(fixed sub-window)→`Conv`(colour/index read) front-end materializes a counted
[1,C,k,k] fp32 plane; replace it with ONE `Einsum bchw,c,hr,wk->brk` that contracts the FREE input against
a static [C] colour weight + a static [G,k] row/col SELECTOR matrix, so only the tiny result is counted and
the selector lives in ln-cheap params. Trade a counted working plane for static params whenever mem−cost >
params−cost (here −324 vs +81). Selection criterion: any net that Slices a fixed grid window then reads it
with a 1×1/small Conv or MatMul (grep for Slice→Conv or Slice→ArgMax on the input). Direct instance of the
Einsum-vs-FREE-input lever (neurogolf-einsum-vs-free-input-lever) — feed struct_scan these Slice→Conv fronts.
