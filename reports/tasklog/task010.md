# task010 — 08ed6ac7

**Rule:** 9x9 grid. Four gray (colour 5) bars hang from the bottom at columns
{1,3,5,7} (column = order[bar]*2+1, order a permutation of 0..3), each with a
distinct height in 1..9. The OUTPUT keeps the identical bar shapes but recolours
each bar by its height RANK: tallest -> colour 1, 2nd -> 2, ..., shortest -> 4;
background stays 0.
**Current:** 16.48 pts (public label-map), beaten.
**Target tier:** B (count-parametric label map) — output colour couples columns
(rank needs a global pairwise comparison) and the bar shape couples r&c
(`r+h[c]>=9`), so it is neither single-Conv (S) nor row⊗col separable (A). But
the entire output is reconstructible from a [9] column-height vector, so the
label map is the only full plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum(input,[2]) heights + count-rank + count-parametric fill, fp32 | B | 3885 | 50 | 16.72 | 200/200 | marginal (+0.24) |
| 2 | replace row-sum with one no-pad Conv (W[1,10,30,1], ch5) -> 120B | B | 2685 | 349 | 16.98 | 200/200 | +0.50 |
| 3 | + cast all [9,9] working planes to fp16 (ints<=9 exact) | B | 2109 | 349 | 17.19 | 200/200 | ADOPT (+0.71) |

## Best achieved
17.19 @ mem 2109 params 349 — beats prior 16.48 by +0.71. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate is the 30x30 uint8 label map L (900B) materialised by the
Pad before the final `Equal(L, arange)`. The output is 30x30 so a per-cell
colour-index carrier is required; the active region is only 9x9 but Pad to 30x30
to off-grid sentinel 10 is unavoidable for the BOOL output (Pad rejects bool, so
the small-canvas carrier cannot itself be the bool output). Second cost is the
300-element Conv weight (params, cheap in ln). No fp16 helps here: dtype tricks
don't shrink the 30x30 carrier and ORT upcasts; uint8 900B is already the floor.

## OPEN ANGLES (re-attack backlog)
- Trim the 300-param Conv: any single-op channel-5 row-sum with fewer elements
  (e.g. exploiting that only rows 0..8 are ever nonzero) would save ~250 params
  (~+0.1) but every alternative (Slice/Gather channel 5) materialises a 3600B
  full plane — net loss. Conv is the cheapest known.
- A separable route is blocked: fill `r+h[c]>=9` couples r&c, so any separable
  formulation still needs a 30x30 fill plane = the same 900B.

## INSIGHT (transferable)
"Bottom-anchored solid bars recoloured by height-RANK" is closed-form tier-B with
ZERO occupancy plane: per-column height = a no-pad Conv row-sum on channel-k
([1,10,30,1] kernel -> [1,1,1,W], 120B, dodging the [1,10,1,W] ReduceSum
intermediate); rank = pairwise-Greater over the tiny height vector (count
function, no sort/argmax); and the bar SHAPE rebuilds from height alone via
`r+h[c]>=9` (no per-cell gray mask read). Cast the height vector to fp16 once and
run all the tiny [K,K] rank/fill planes in fp16 (ints<=9 exact) to halve them.

## 2026-07-01 sequential deep pass

Current source has advanced beyond the older 2109B label-map solution:

- **memory 730, params 340, points 18.024586072544047**
- fresh recheck: **1000/1000 pass**
- dominant memory: `Pad -> I [1,10,1,30]` = 300B, gray bar slice
  `[1,1,9,4]` = 144B, pairwise rank matrix = 64B.
- dominant params: `k [1,10,30,1]` row-threshold initializer = 300 params.

Important probe: replace dense `k` with a sparse initializer.  The tensor has
300 entries but only 145 nonzero values, so this would save ~155 params if
legal.  Result: not landable.  After working around the local sanitizer's sparse
rename issue, ORT can run stored examples, but scorer shape inference rejects:

`ShapeInferenceError ... Less ... unsupported type: sparse_tensor(uint8)`

Also considered constructing `k` with `Range`/`Where`/`Concat`: params drop, but
the required 30/300-element intermediate tensors add more memory than the 300
dense params save.

Conclusion: no adoptable improvement.  The current graph already avoids the old
30x30 label carrier; the remaining cost is mostly the dense final threshold
table, and sparse initializers are not accepted for `Less` by the scorer.

## 2026-07-01 parallel task-agent deep dive

Scope: task010 only.  Existing notes were treated as hypotheses and rechecked
against `data/task010.json`, `src/custom/task010.py`, `networks/task010.onnx`,
manifest state, stored eval, and fresh generation.

### Human rule, verified

Stored examples are 9x9.  Inputs contain only 0 and gray 5.  Four solid vertical
gray bars occupy odd columns 1, 3, 5, and 7, each bottom-anchored and with a
distinct height.  Output keeps exactly the same occupied cells and recolors each
bar by descending height rank: tallest -> 1, next -> 2, next -> 3, shortest -> 4.
Background remains 0.  The visible test case has heights `[8,3,7,5]` at columns
`[1,3,5,7]`, so ranks are `[1,4,2,3]`.

Python oracle:

```text
height[c] = count(input[:, c] == 5) for c in [1,3,5,7]
rank[i] = 1 + count(height[j] > height[i])
output[input[:, c_i] == 5, c_i] = rank[i]
```

Oracle verification: train 2/2, stored test 1/1, fresh generator 300/300.
Confidence: verified for current generator samples; assumes generator keeps
distinct heights and fixed odd columns, which is supported by stored/fresh data.

### Current source/live state

Manifest row for task 10: `memory 730`, `params 340`, `points 18.024586072544047`,
`method custom:task010`.

Task-only verification:

```text
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py 010
-> {'ok': True, 'pass': 265, 'fail': 0, 'memory': 730, 'params': 340,
    'points': 18.024586072544047, 'error': None}

PYTHONPATH=. .venv/bin/python src/harness.py networks/task010.onnx 10
-> ok true, pass 265, fail 0, memory 730, params 340,
   points 18.024586072544047

PYTHONPATH=. .venv/bin/python reports/scripts/fresh_verify.py 010
-> fresh_instances=1500/1500, incumbent fail = 0
```

### Cost anatomy

| tensor/init | shape | dtype | cost | semantic job |
|---|---:|---|---:|---|
| `I` | `[1,10,1,30]` | uint8 | 300B | padded per-channel/per-column threshold values for final broadcasted `Less` |
| `k` initializer | `[1,10,30,1]` | uint8 | 300 params | row threshold table: ascending for background, descending for colors 1..4, off-grid guards |
| `l` | `[1,1,9,4]` | float32 | 144B | gray-channel crop for the four odd bar columns before height sum |
| `r` | `[1,1,4,4]` | int32 | 64B | pairwise greater-than rank matrix after bool cast |
| `H` | `[1,5,1,9]` | uint8 | 45B | combined background threshold row plus four rank/color rows before padding |
| `B` | `[1,4,1,9]` | uint8 | 36B | rank-colored bar-height values placed on odd columns |
| small rank/height tensors | scalar to `[1,1,4]`/`[1,4,1,4]` | mixed | 141B total | casts, unsqueezes, splits, and 4-way rank selection |

Dominant floor is now a broadcast-threshold output carrier, not the old 30x30
label plane.  Final `Less(k, I)` is graph output, so the full 30x30 bool output
is not charged as an intermediate.  Any alternative that computes a small bool
image and then pads/concats it would materialize the bool image as an
intermediate and likely loses despite smaller constants.

### Prior-note challenge

- Still valid: semantic rule, distinct bottom bars, rank-by-height recoloring,
  no occupancy plane needed, sparse `k` would be attractive if legal.
- Contradicted/outdated: the "Current 16.48", "Best achieved 17.19", and
  2109B/900B label-map floor sections describe older graphs, not the current
  incumbent.  Current source/live is 18.0246 with 730B/340 params.
- Contradicted for the current graph: "Conv is the cheapest known" for row-sum.
  It was useful versus older label-map builds, but current `Slice+ReduceSum`
  is better overall.
- Unverified in this pass: the old "Pad rejects bool" claim.  It was not needed
  for the two selected probes because a bool-pad route has a worse lower bound
  under the scorer even before legality: the pre-pad bool image becomes a large
  counted intermediate.

### Mechanism probes

1. Direct height extraction by no-pad Conv

- Family: input one-hot direct routing / bounded crop.
- Expected payoff: remove `l [1,1,9,4]` 144B and replace with `convh
  [1,1,1,30]` 120B plus a small slice/squeeze, saving only about 8 scored bytes
  after shape alignment.
- Kill condition: params added by a `[1,10,30,1]` Conv kernel exceed the byte
  saving.
- Proof result: stored pass, but worse score.

```text
in-memory candidate:
{'ok': True, 'pass': 265, 'fail': 0, 'memory': 722, 'params': 632,
 'points': 17.78918154652778, 'error': None}
```

No adopt: `memory+params` rises from 1070 to 1354.

2. Sparse final threshold table

- Family: direct output threshold algebra / param compression.
- Expected payoff: dense `k` has 300 params but only 145 nonzero entries, so a
  legal sparse initializer could save about 155 params.
- Kill condition: official checker/shape inference or harness rejects sparse
  tensor use as `Less` input.
- Proof result: not landable.

```text
dense elems 300, sparse nnz 145
checker full_check error:
  ShapeInferenceError (op_type:Less): unsupported type: sparse_tensor(uint8)
harness eval error:
  Invalid model. Node input 'safe_name_34' is not a graph input, initializer,
  or output of a previous node.
```

The harness error is the local sanitizer's sparse-initializer rename gap; the
more important blocker is ONNX full-check rejecting sparse uint8 for `Less`.

### Next experiment

No source adoption recommended from this pass.  The remaining credible path is
to eliminate or shrink the 300B `I` carrier without materializing any bool
pre-output.  A proof must beat the incumbent lower bound `memory+params = 1070`;
generated/factorized `k` is not enough unless the final op can still broadcast
directly to the graph output with fewer than 300 extra counted bytes.
