# task270 — ae3edfdc

## 2026-07-01 task-only deep dive — current live/source already beats old scalar note

**Status:** verified no-adopt, useful failure.  The prior notes below are stale in
their score framing: `src/custom/task270.py` is now an exact source reconstruction
of the current live graph, and both source and `networks/task270.onnx` measure:

| artifact | stored pass | mem | params | pts | fresh |
|---|---:|---:|---:|---:|---:|
| source `task270.build` | 266/266 | 2742 | 327 | 16.970892945380264 | — |
| `networks/task270.onnx` | 266/266 | 2742 | 327 | 16.970892945380264 | 200/200 |

### Human rule, verified

All stored examples are 15x15 with colours `{0,1,2,3,7}`.  There are two
independent flowers:

- centre colour `2`, petal colour `3`;
- centre colour `1`, petal colour `7`.

Each centre is a single pixel.  Same-colour petals, if present, lie on the same
row or same column as their centre, at distance at least 2.  Output keeps the
centre and pulls every existing petal back to the adjacent cell in the same
direction.  Missing directions remain background.

Python oracle over all stored examples:

```text
oracle fails 0 []
axis violations 0 []
distance min/max 2 12
```

This makes the semantic rule **verified** for stored data.  Fresh generator
availability was also verified (`gen270` from `/tmp/arc-gen/tasks/task_ae3edfdc.py`).

### Current mechanism

The current graph is **not** the old 8235B scalar matrix-rebuild candidate.  It
is a lower-cost packed 1-D profile graph:

1. `Einsum(input, chan_w)` builds row/column packed colour profiles.
2. `Slice`/`BitwiseAnd` extracts centre and petal profiles for colours
   `1,2,3,7`.
3. `CumSum`, `ArgMax`, `GatherElements`, and scalar `Sub`s compute the four ray
   flags for each flower.
4. `Pad`s place row/column centre and adjacent-neighbour basis vectors.
5. One final float16 `Einsum` emits the full thresholded 10-channel output
   directly.  No `[15,15]` or `[30,30]` label plane is materialized.

### Cost anatomy

| component | tensors / params | bytes or params | semantic job |
|---|---:|---:|---|
| final row/column basis stacks | `row_basis`, `col_basis` `[7,30]` fp16 | 840B | carry board, centres, and four adjacent rows/cols into the output-only final `Einsum` |
| 1-D packed profiles | `row_pack`, `col_pack` `[1,30]` fp32 | 240B | reduce 10 input channels into row/column colour bitmasks |
| shifted centre/petal basis pads | twelve `[1,30]`/`[1,15]` fp16/int16 intermediates | ~900B | align centre and adjacent cells without full 2-D planes |
| prefix/flag scalars | `CumSum`, `ArgMax`, `GatherElements`, `ReduceSum`, scalar `Sub`s | ~200B | decide up/down/left/right petal existence for each flower |
| final selector params | `coef` 110, `row_sel` 77, `col_sel` 77, `board` 30, `chan_w` 10, small scalars | 327 params | route each candidate rank-1 placement to colour channels, including channel-0 background |

Dominant live memory is the two basis stacks (840B), not directional MatMuls or
full-canvas carriers.  The graph is already tier A/S-adjacent: the only
full-canvas tensor is the official output, which is free under the scorer.

### Prior notes challenged

- **Contradicted/stale:** "Current prior adopted 14.85" and "best achieved
  15.97 @ mem 8235 params 146" are no longer current.  The source and live
  artifact both verify at 16.9709 @ mem 2742 params 327.
- **Still valid conceptually:** the task is a 12-scalar / 1-D-profile
  reconstruction problem, not a full directional plane pipeline.
- **Superseded:** the old irreducible-floor analysis about ~34 fp16 15x15 planes
  is not a current floor.  The live graph has no 15x15 working plane and no
  directional MatMul stack.
- **Still useful:** uint8/full-label carrier notes remain useful historically,
  but they are worse than the current direct-output `Einsum` route.

### Mechanism tests

1. **Scalar pull-back / 12-scalar rebuild (NEAR_18 claim).**
   - Expected payoff: delete directional MatMuls and rebuild from centres plus
     direction flags.
   - Proof test: compare current source/live score and check whether the claimed
     scalar route still removes a live dominant cost.
   - Result: concept validated historically, but not adoptable now.  Current
     graph already implements a cheaper 1-D profile/direct-output variant and
     beats the claimed `8235+146` total by a wide margin (`2742+327`).
   - Kill condition hit: an old scalar label route necessarily pays at least a
     `[15,15]` label carrier plus `[30,30]` pad carrier (>=1350B before profile
     and selector costs), while current direct-output basis stacks are 840B and
     avoid the label carrier entirely.

2. **Integer dtype shrink for the final direct-output algebra.**
   - Expected payoff: halve `row_basis`/`col_basis` and pad-chain memory if the
     final `Einsum` accepted int8/uint8/int16.
   - Proof test: construct minimal ORT `Einsum` kernels for int8, uint8, int16,
     float16, and float32 under opset 18.
   - Result: int8/uint8/int16 all fail with `NOT_IMPLEMENTED`; float16 and
     float32 work.
   - Kill condition hit: ORT has no CPU kernel for integer `Einsum`, so the live
     final algebra cannot be dtype-shrunk without replacing the output primitive.

### Next exact experiment

Look for a non-`Einsum` final emitter that preserves output-only materialization
but consumes smaller integer/bool basis tensors.  A scatter route is unlikely to
win unless it can avoid a full zero-data input, because a full `[1,10,30,30]`
zero carrier would dominate current memory.

**Rule:** Fixed 15x15 grid, background 0, two "flowers". Flower 0 = centre colour 2 with
petals colour 3; flower 1 = centre colour 1 with petals colour 7. Each flower's centre is a
single pixel; in each of the 4 orthogonal directions a petal MAY exist, placed somewhere along
that ray at distance >= 2 from the centre (it "flew off"), at most one petal per ray. The OUTPUT
keeps both centres in place and moves every existing petal to the cell immediately ADJACENT to
its centre in that direction. Because flower 0 is the only source of colour-3 pixels and flower 1
the only source of colour-7, "does a petal exist in direction d from this centre" reduces to "is
there a petal-colour pixel anywhere along that ray" — closed-form, no flood-fill, no shape
correspondence.

**Current (prior adopted):** 14.85 pts.
**Target tier:** A — separable directional reconstruction; output colours are a FIXED known set
(1,2,3,7) so slice+place, not a Conv colour-index plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 6 | **SCALAR rebuild: 12-scalar pull-back, 1-D profile flags + single small MatMul L=RS@CW** | A* | **8235** | **146** | **15.97** | **200/200** | **NEW BEST (+1.11)** |
| 1 | CumSum prefix-OR presence + MatMul shifts, fp32 | A | 52200 | 961 | 14.12 | — | below P |
| 2 | strict-triangular MatMul presence (skip Greater/Cast: <=1 petal/ray so gated exclusive count is {0,1}) + MatMul shifts, fp16, Sum-fold | A | 21600 | 1858 | 14.94 | — | MARGINAL |
| 3 | stacked 2-flower channel batch + grouped Conv shift | A | 32850 | 1109 | 14.57 | — | worse (multi-ch planes bloat mem) |
| 4 | #2 + matrix dedup (all 8 matrices are Aup/Sup or their transpose) | A | 20700 | 958 | 15.02 | — | MARGINAL |
| 5 | #4 + uint8 label entry plane (Cast L->uint8, Pad sentinel 99, uint8 Equal) | A | 20025 | 958 | **15.05** | 500/500 | MARGINAL (+0.20) |

## Best achieved
**15.97 @ mem 8235 params 146** (attempt #6) — beats prior 14.85 by **+1.11**, fresh 200/200,
eval 204/204 exact. CLEARS the +0.3 bar decisively. (Earlier 15.049 @ mem 20025 was MARGINAL.)

## ⭐ THE WIN (attempt #6) — scalar pull-back, not directional MatMuls
The shelving note ("MARGINAL ~0.1 short") came from over-modelling the task as a directional
prefix/shift problem (~34 fp16 15x15 planes). The real structure: the ENTIRE output is a function
of 12 scalars — two centres (r,c) + 8 direction flags — so it is a COUNT/SCALAR->FIXED-PATTERN
rebuild, NOT a plane pipeline.
- Centres = position-weighted ReduceSum of the centre-colour profile (single pixel -> exact).
- Each direction flag is a pure 1-D-PROFILE test, NO 2-D plane: rowprof=ReduceSum(petal_ch,axis=col),
  up = any rowprof at rows<r, dn = rows>r; colprof for lf/rt. Exact because vertical petals sit at
  col==c, row!=r (land in rowprof at rows!=r) while horizontal petals sit at row==r (touch only
  rowprof[r]) -- the two never collide.
- 10 candidate cells (2 centres + 8 petals) are each a rank-1 placement, so the WHOLE label plane is
  ONE small matrix product: RS[r,k]=Equal(rampR[r],row_k) [15,10], CW[k,c]=(colour_k*flag_k)*
  Equal(col_k,rampC[c]) [10,15], L=RS@CW [15,15] (cells disjoint -> exact). No per-cell 15x15 plane.
- Route to free output: Cast L->uint8, Pad to 30x30 with sentinel 99 (off-grid stays all-zero),
  Equal(L,arange[1,10,1,1]) -> bool output.
Dominant intermediates now: two fp32 row/col profiles (1200B each) + the 900B uint8 Pad carrier.

## Irreducible-floor analysis (superseded — kept for history)

## Irreducible-floor analysis
~34 fp16 [1,1,15,15] working planes (450B each) dominate. The per-flower pipeline needs 13 planes:
4 directional-presence MatMuls (Aup@P above, AupT@P below, P@AupT left, P@Aup right) — these
contract DIFFERENT axes/sides so they cannot be fused into one op; 4 centre-gated planes; 4
one-step shift MatMuls to the neighbour; 1 Sum. Removing the gate-then-shift pair by gating with a
shifted centre instead (petUp = shiftUp(C) (.) (Aup@P)) is plane-count-neutral (still 4 presence +
4 shifted-centres + 4 products). Stacking the two flowers on the channel axis is mem-neutral
(planes double in size, halve in count) and adds fp32 Concat planes, so it is strictly worse here.
Input prep is 4 fp32 channel Slices (3600B, Slice preserves fp32) + 4 fp16 casts (1800B); a 1x1
colour-index Conv would instead pay a 30x30 fp32 plane (3600B) so it doesn't help. The uint8 label
entry plane (900B) is already minimal. Net: mem floor ~20KB -> ~15.05, ~0.1 short of the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- A single fused op that yields all four directional presences at once (they need 4 distinct
  triangular contractions) would cut ~4 planes (~1.8KB) and likely clear +0.3 — none found in opset 11/13.
- Encode the 4 direction flags additively at the centre into ONE plane and expand with a per-bit
  Conv: blocked because one scalar value can't be split back into 4 directional taps by a linear conv.
- A bounded directional Conv instead of the full-length triangular MatMul (petal distance is bounded
  by grid size 15, no tighter generator bound) would not shrink params/planes meaningfully.

## INSIGHT (transferable)
- ⭐ When a directional prefix/suffix indicator is read ONLY at a single gated pixel and at most one
  hit exists along the ray, the strictly-triangular MatMul output IS already {0,1} at that pixel —
  drop the Greater+Cast (saves 2 planes/direction).
- ⭐ For axis-aligned row/col shifts, ALL four one-step shift matrices and ALL four strict-triangular
  presence matrices collapse to ONE base matrix + its transpose (Sdn=Sup.T, Sleft=Sup.T, Sright=Sup;
  Adn=Bl=Aup.T, Br=Aup). Store base+transpose as two inits (params, no runtime Transpose planes) —
  halves the matrix-param budget at zero mem cost.
- ⭐ uint8 label entry plane: Cast the fp16 colour-index plane to uint8 (900B vs 1800B fp16 at 30x30),
  Pad with an out-of-range sentinel (e.g. 99), and Equal(uint8, uint8 arange) — ORT supports uint8
  Pad and uint8 Equal under ORT_DISABLE_ALL; off-grid sentinel keeps the harness's all-zero off-grid
  target satisfied.
