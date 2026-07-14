# task048 — ARC-AGI 239be575

**Rule:** A width×height (each 5..8) black canvas holds scattered cyan pixels plus
TWO solid 2×2 red boxes (non-overlapping, gap≥1). The output is a **1×1** grid:
cyan iff a 4-connected path of non-background (cyan OR red) cells links the two
red boxes (reaches a red cell of the other box), else black. The generator rejects
instances where 4- vs 8-connectivity disagree, so the predicate is unambiguous.
This is a genuine 4-connectivity / flood-fill predicate over a variable-size noisy
grid — no closed-form/separable escape (path existence is inherently iterative).

**Current:** 16.415 pts, `ext:biohack_new`, mem 5283, params 66
**Target tier:** detection (bounded flood) — connectivity is not collapsible to
copy/separable/count; the only freedom is the dtype/round/canvas of the flood.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | baseline flood Conv→Greater→Where, 12 rounds, 8×8 | det | 5283 | 66 | 16.415 | 200/200 | reference |
| 2 | fold Greater+Where → `Min(passable,count)` (drop bool plane/round) | det | 4515 | 66 | 16.570 | 200/200 | +0.155 |
| 3 | + fuse final reach∩red into last `Min(twos16,count)` (drop 1 plane) | det | 4387 | 66 | 16.599 | 200/200 | **+0.18 MARGINAL** |
| 4 | passable from ch0 (`bg==0`) instead of red+cyan | det | 4323 | 67 | 0 (fail 16) | — | WRONG: off-grid cells have ch0=0 → falsely passable; red+cyan is the correct passable signal |

## Best achieved
16.599 @ mem 4387 params 66 — adopted in src/custom/task048.py. Beats prior 16.415
by **+0.18 → MARGINAL (< +0.3)**. Behaviorally **identical** to the deployed
baseline: 0 disagreements over 20 000 fresh instances (both nets implement the same
12-round flood and even fail the same ~0.05% extreme-long-path cases).

## Irreducible-floor analysis
Dominant memory = the 24 fp16 [1,1,8,8] flood planes (12 Conv `count` + 12 `Min`
reach), 128 B each = 3072 B. Each is irreducible:
- **Canvas**: generator bounds the grid to ≤8×8, already cropped; 8 is the max.
- **dtype**: the propagation step is a *cross* (4-neighbour) dilation, which forbids
  the cheaper `MaxPool` (rectangular = 8-neighbour = wrong connectivity, lets the
  path jump a non-passable gap). So a float `Conv` is mandatory ⇒ fp16 is the floor.
- **rounds**: empirically (100 000 samples) the max dilations needed to reach the
  other box is 14, with R=12 already failing 0.035 % of cases → P(fresh-200 all
  pass)=93 %. Dropping to R=11 → P=83 %, R=10 → P=68 %. **Reducing rounds fails the
  isolated fresh-200 generalization gate**, so R=12 (the baseline's choice) is the
  minimum honest round count. The baseline itself is borderline on fresh-200.
Setup ≈ 1315 B (two mandatory f32 channel slices 512 B + their fp16 casts + the
argmax/onehot seed + passable). The +0.3 threshold needs mem ≤ 3891 (cut 496 more);
the only fat left is in the irreducible flood. ⇒ structurally capped at MARGINAL.

## OPEN ANGLES (re-attack backlog)
- Component-merge formulation: seed ALL red, flood once, test whether the two
  boxes land in the same label — but labeling needs a 64-iter propagation of a
  per-cell id (far more planes than the BFS). Not promising.
- Reachability via a single [64,64] adjacency power (A^k) — [64,64] fp16 = 8192 B
  per MatMul, strictly worse than the 8×8 BFS.
- Trim the two f32 slices to one: any single-channel passable signal that excludes
  off-grid cells (ch2+ch8) requires both channels; ch0 alone mis-marks off-grid.
  ~64 B at best, insufficient for +0.3.

## INSIGHT (transferable)
⭐ **Min folds Greater+Where in a bounded-flood/BFS** (saves a bool plane/round):
`reach' = Min(passable∈{0,1}, count)` == `Where(count>0, passable, 0)` and stays
bounded in {0,1} (no fp16 overflow), since Min(1,int≥0)=(int≥1) and Min(0,·)=0.
⭐ **Fuse the final reach∩target into the last round's gate**: replacing the last
`Min(passable,count)` with `Min(target_mask,count)` makes the final propagation
output already-masked to the target colour (target⊆passable), deleting the trailing
product plane. ⭐ fp16 `Min` DOES run under ORT_DISABLE_ALL (confirms task377 over
the older "fp16 Min/Max crashes" warning). ⭐ **Connectivity is symmetric**: seeding
from EITHER box (here the row-major-first red cell via ArgMax) gives the same
"are they connected" answer, so you don't need to identify the generator's specific
box0. ⭐ **Re-probe verdict: a flood/connectivity predicate over a variable-noisy
grid with a long path-length tail is a TRUE memory wall** — the deployed baseline is
already near its honest floor; only ~+0.18 of dtype/op-fold golf is available and
reducing rounds trades directly against the fresh-200 generalization gate.
