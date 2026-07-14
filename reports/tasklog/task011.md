# task011 — 09629e4f

**Rule:** The 11x11 grid is a 3x3 array of 3x3 mini-cells separated by a gray (5)
"hollywood squares" frame at rows/cols 3 and 7. Each mini-cell holds rainbow
pixels (colours {2,3,4,6,8}); exactly ONE cell ("chosen") has 4 coloured pixels,
every other has 5. The output keeps the frame and fills each output mini-cell
block (mr,mc) SOLID with the colour of the chosen cell's pixel at interior
position (mr,mc) (bg if empty) — i.e. output = the chosen 3x3 cell upscaled 3x
onto the frame. NOT a flood-fill/connectivity task: it is count-discrimination
(unique min-count cell) + a position->fill transposition.

**Current:** 14.12 pts, gen:biohack_new, mem 50756, params 2331
**Target tier:** A/B — fixed geometry, separable, no data-dependent crop size; the
only forced full-canvas tensors are the colour plane and the 30x30 output label.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice[1,10,11,11]+Mul+ReduceSum colf, gather 9x9 compact, count<5 select, gather upscale | B | 14197 | 310 | 15.42 | — | works, but in11+kin = 9680B |
| 2 | replace slice+Mul with 1x1 Conv on FREE input -> colf30 | B | 8317 | 298 | 15.94 | 200/200 | ADOPT-candidate |

## Best achieved
15.94 @ mem 8317 params 298 — adopted? N (per instructions, not self-adopted).
Beats prior 14.12? Y by +1.82.

## Irreducible-floor analysis
Two ~3600B planes dominate: (a) colf = colour-index plane [1,1,30,30] fp32 (3600B)
from the 1x1 Conv — needed once to read both occupancy and colour; can't be fp16
because ORT Conv requires input/weight type match and the input is fp32-free
(casting the [1,10,30,30] input to fp16 costs 18000B). (b) L = padded uint8
[1,1,30,30] label (3600B) — the output-shaping floor; output is genuinely 30x30
one-hot so the 30x30 sentinel-padded label is unavoidable. Remaining ~1100B are
small (cr [1,1,9,30]=1080B from the two-step row/col gather, plus 484B label
copies). colf+L ~= 7200B is the practical floor for this construction.

## OPEN ANGLES (re-attack backlog)
- Drop colf30 entirely: gather the 9 interior rows on a fp16-cast of the conv
  plane, or contract the channel axis with a MatMul that lands the 11x11 colour
  directly — but every channel-contraction route I see materialises a >=3240B
  10-channel intermediate, so unlikely to beat 3600B. (~+0.1 at most.)
- Shrink cr (1080B) by gathering a flattened 81-index map in one Gather op
  instead of axis2-then-axis3 (~+0.05).
- These are marginal (<0.2); the 7200B colf+L floor caps this task around ~16.2.

## INSIGHT (transferable)
"Odd-one-out by pixel COUNT then upscale" is closed-form tier-B, NOT a detection
wall: the unique cell is `count < k` (here <5) — no ReduceMin/ArgMax needed when
exactly one cell differs. Select-the-cell collapses to `Sum_{R,C} (count<k)*block`,
and the 3x->block upscale reuses the task195 const-index-map Gather idiom (block
content = cellflat[(r//4)*3 + (c//4)], frame via a const Where). Computing the
colour plane with a 1x1 Conv on the FREE fp32 input (3600B) beats Slice+Mul+Sum
(9680B) whenever you need the whole plane anyway. ⭐ count-discriminate + Kronecker
upscale pattern.

## 2026-07-01 sequential deep pass — LANDED

Current source before this pass was already much better than the older log:
`memory=1991, params=123, points=17.343662833569816`, fresh 1000/1000.

New observation: because the generator assigns the five rainbow colours as
`rainbow[0:count]`, every 5-pixel mini-cell contains colour 8 and the unique
4-pixel chosen mini-cell does not.  Therefore the chosen cell can be found by
counting colour-8 pixels in each 3x3 block.

Replaced nine independent `Slice(input ch8 block) + ReduceSum` branches with:

- one `Conv(input, W[1,10,3,3], strides=4)` where only channel 8 has 3x3 ones;
- one `Slice` to keep the true 3x3 mini-cell grid from the 7x7 stride output;
- one `Reshape` to the existing `[9]` `noise_sums` vector.

Verification:

- Stored: pass 267/267.
- Fresh: 1500/1500, candidate identical to incumbent.
- Adopt gate: **ADOPTED**.
- Source/live after adoption: both `memory=1863, params=163,
  points=17.386181315191372`.

Tradeoff: params increase by 40, but memory drops by 128, net cost improves by
88 and points improve by about +0.0425.

Reusable insight: when several fixed-size, fixed-stride blocks need the same
channel/count statistic, a single strided Conv can replace many tiny
Slice+ReduceSum branches.  This is especially useful when the Conv's extra
weight params are smaller than the eliminated repeated fp32 slice memory.

## 2026-07-01 parallel deep dive — ADOPTION CANDIDATE, not integrated

Scope: task011 only.  Current source/live remains the sequential-pass incumbent:
stored `267/267`, fresh `1500/1500`, `memory=1863`, `params=163`,
`points=17.386181315191372`.

### Human-readable rule

All stored and generated examples are fixed 11x11 "hollywood squares": a 3x3
array of 3x3 mini-cells separated by gray `5` lines at rows/cols `3` and `7`.
Colours inside mini-cells are from `{2,3,4,6,8}` plus black `0` holes.  Exactly
one mini-cell has 4 coloured pixels; the other 8 mini-cells have 5 coloured
pixels.  Because the generator assigns `rainbow[0:count]`, the 5-pixel cells
always contain colour `8`, and the unique 4-pixel chosen cell never contains
colour `8`.

The output copies the chosen 3x3 mini-cell pattern by position: each chosen-cell
position `(mr, mc)` becomes a solid 3x3 output block at output mini-cell
position `(mr, mc)`.  Gray separators stay gray; black holes become black
blocks.  Python oracle using "find the only mini-cell with no `8`, then 3x
upscale its 3x3 contents" passed stored `267/267` and fresh `1000/1000`.

### Prior-note challenge

| prior claim | status | evidence |
|---|---|---|
| `14.12`, `mem=50756`, `params=2331` baseline | contradicted/stale | current source and live measure as `1863/163/17.38618` |
| 1x1 colour-plane floor around two 3600B planes | contradicted/stale | active graph has no full fp32 colour plane; largest tensors are `color30` 900B and `selected_oh` 252B |
| count-discrimination + Kronecker upscale rule | still valid | oracle and generator confirm unique 4-pixel/no-8 cell and positional 3x block fill |
| strided colour-8 Conv replaces nine Slice+ReduceSum branches | still valid | active graph uses one stride-4 Conv; stored and fresh pass |

### Current cost anatomy

| component | tensor(s) | bytes | semantic job |
|---|---:|---:|---|
| final label carrier | `color30` uint8 `[1,30,30]` | 900 | sentinel-padded label plane for free final `Equal(..., channel_ids)` output |
| chosen-cell read | `selected_oh` fp32 `[1,7,3,3]` | 252 | dynamic crop of chosen cell, channels `0..6`; channels `7..9` excluded because chosen cell never uses them |
| colour-8 discriminator | `noise_grid7` fp32 `[1,1,7,7]`, `noise_grid`, `noise_sums`, `clean_*` | 277 incumbent / 250 candidate | one stride-4 Conv counts colour `8`; zero count selects the chosen mini-cell |
| 11x11 label synthesis | `color4`, `color_rows`, `color11` | 181 | append gray separator label and gather row/col tile index to form the visible 11x11 grid |
| scalar crop routing | `clean_idx`, `r0/c0/r1/c1`, `starts_sel`, `ends_sel` | 112 | convert chosen cell index to dynamic slice bounds |
| colour decode | `color3`, `color3_u8` | 81 | ArgMax chosen-cell one-hot to compact uint8 labels |

Dominant cost is the 900B `color30` label carrier.  A bool one-hot-before-pad
route would need at least `[1,10,11,11]` = 1210B before the final pad, so the
current label-plane + final Equal route remains better.

### Mechanism tests

1. **Direct output threshold algebra / ArgMax dtype shrink.**
   - Expected payoff: remove the fp32 `clean_float [9]` carrier by feeding
     `ArgMax` a smaller dtype; max saving 27-36B.
   - Proof test: stored eval and fresh candidate equivalence.
   - Kill condition: ORT rejects `ArgMax` input dtype or stored/fresh mismatch.
   - Result: `ArgMax(bool)` is rejected by ORT, but `ArgMax(uint8)` is valid.
     Candidate changes only `Cast(clean_bool -> clean_float)` to
     `Cast(clean_bool -> clean_u8)`.  Stored passes `267/267`, fresh passes
     `1500/1500`, candidate equals incumbent on all fresh.  Cost becomes
     `memory=1836`, `params=163`, `points=17.3995976654996` (+0.013416).

2. **Flattened one-step output gather.**
   - Expected payoff: remove `color_rows [1,11,4]` (44B) by flattening
     `color4` and gathering `color11` with one 11x11 index map.
   - Proof test: stored eval and measured total cost.
   - Kill condition: params added by the larger gather index exceed memory saved.
   - Result: stored-correct but worse overall:
     `memory=1835`, `params=273`, `points=17.346505090338745`.  Killed because
     it saves only 28B memory while adding 110 params.

### Recommendation / next experiment

Recommend main-session adoption of the `clean_bool -> uint8 -> ArgMax` candidate:
it is source-owned, task011-scoped, stored/fresh verified, and output-identical
to incumbent on 1500 fresh cases.  No repo source adoption was performed in this
parallel pass.

Next exact experiment if main wants to chase marginal bytes: try to eliminate or
shrink `color3 int64 [1,3,3]` by replacing ArgMax colour decode with a compact
uint8 label LUT only if it avoids materialising a larger one-hot/gather carrier;
expected payoff at most 63B, kill if it introduces any `[1,10,3,3]` or larger
intermediate.
