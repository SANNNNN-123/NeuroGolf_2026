# task146 — 662c240a

**Rule:** Input is a 9x3 grid = three 3x3 colour blocks stacked vertically (block i = input rows 3i..3i+2, cols 0..2). Exactly two blocks are symmetric along the main diagonal (block[r][c]==block[c][r]); exactly ONE is asymmetric. The 3x3 output is the asymmetric block. Colours are sampled from 1..9 (background 0 never appears inside the grid); output cells outside the 3x3 grid are unset (all-zero).
**Current:** 16.86 pts (prior), method n/a
**Target tier:** B (label-map + Equal) pushed below floor via tiny working canvas — the active region is only 9x3 in / 3x3 out, so no full 30x30 fp32 plane is ever needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop[1,10,9,3]→Conv colf→within-block transpose→ArgMax asym→Gather rows→uint8 Pad→Equal | B+ | 2868 | 57 | 17.02 | 267/267 stored | passed but channel-0 bg pad bug first (sentinel 255 fix) |
| 2 | fp16 colf + transpose chain; ArgMin equal-count (drop Sub) | B+ | 2634 | 56 | 17.10 | — | |
| 3 | fp16 bool→count chain (Reshape bool first, Cast fp16) | B+ | 2493 | 56 | 17.16 | — | |
| 4 | crop channels 1..9 only (ch0 bg never set in grid) | B+ | 2385 | 55 | 17.20 | 200/200 | ADOPTED |

## Best achieved
17.20 @ mem 2385 params 55 — beats prior 16.86 by +0.34. Fresh isolated 200/200.

## Irreducible-floor analysis
Two dominant intermediates: cropped input `[1,9,9,3]` fp32 = 972B (entry — 9 colour channels over the 9x3 active region; can't go narrower since the per-instance colour set is an arbitrary 6-subset of 1..9, and Slice keeps fp32) and the label pad `[1,1,30,30]` uint8 = 900B (the output one-hot is `Equal(label, arange)` and the label MUST be full-canvas to broadcast to the [1,10,30,30] output; uint8 is the floor for a colour-index label). Everything else is <120B (fp16 transpose/compare chain + scalar index arithmetic).

## OPEN ANGLES (re-attack backlog)
- Kill the 900B label pad: route the 3x3 box one-hot directly into the FREE output without a full-canvas label. Padding a [1,10,3,3] bool one-hot to [1,10,30,30] needs Pad (rejects bool) or uint8 9000B — both worse. A separable rowcond⊗colcond can't represent the arbitrary 3x3 colour content. No cheaper construction found.
- Kill the 972B crop: contract the channel axis via MatMul straight off the FREE input to dodge materializing 9 channels in the cropped region — but a spatial crop is still needed first, so MatMul(input,...) over [30,30] would re-materialize a full plane. Net not obviously cheaper.

## INSIGHT (transferable)
⭐ "Output is one of K fixed sub-blocks selected by a per-block symmetry/equality predicate" is closed-form tier-B, NOT a detection wall: a within-BLOCK transpose is a Reshape→Transpose(swap inner axes)→Reshape on the tiny colour plane (no per-cell coupling matrix), block selection = ArgMin of per-block EQUAL-cell count (symmetric block = count 9, the unique asymmetric one is strictly lower — no Sub, no threshold). Two structural levers stacked to break the 16.8 floor: (1) the active region is tiny (9x3 in / 3x3 out) so Slice the FREE input to [1,9,9,3] (drop ch0 background which is never set inside the grid) before any Conv — never touch a 30x30 fp32 plane; (2) pad the colour-index label with a sentinel (255) NOT 0, because the harness expects cells OUTSIDE the native output grid to be all-zero (unset), not background-channel-0=1.

## 2026-07-01 — latent direct-output algebra micro-win

Class probe for `compiler_direct_output_algebra`.

The current source had already superseded the old label-map route: it computes
two symmetry predicates, dynamically slices the selected 3x3 one-hot block
directly from the free input, and pads that block to the free output.  Current
pre-edit baseline:

- `measure_task.py 146`: stored/arc-gen `267/267`, `memory=388`,
  `params=107`, `points=18.79544223743131`.

The class-level question was whether this latent direct-output task still had
any carrier to delete.  The 3x3 selected block (`selected [1,9,3,3]`, 324B) is
load-bearing: replacing it with three candidate blocks gated by symmetry flags
would materialize more block data, and replacing the dynamic slice with dense
30-wide selectors would add more params than it saves.

Small source-owned win found: the Conv symmetry checksum was computing all
three block checks, but block0's check is never used.  Because exactly one block
is asymmetric, the selected row is determined only by whether block1 and block2
are symmetric:

```text
b1_sym and b2_sym -> select block0
not b1_sym and b2_sym -> select block1
not b2_sym -> select block2
```

Changing the checksum Conv from `pads=[0,0,-21,0]` to `pads=[-3,0,-21,0]`
skips block0 and emits only the two needed checks.  The following `Split` now
has only `b1_check` and `b2_check`.

Verification:

```text
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py 146
{'ok': True, 'pass': 267, 'fail': 0, 'memory': 380, 'params': 107,
 'points': 18.81173587691741, 'error': None}

PYTHONPATH=. .venv/bin/python reports/scripts/fresh_verify.py 146
task146 arc=662c240a fresh_instances=1500/1500
  incumbent fail = 0

PYTHONPATH=. .venv/bin/python -m src.adopt 146
ADOPTED: task146 real 18.80 -> 18.81 (generalizing)
```

Result: adopted `memory 388 -> 380`, params unchanged, `+0.0162936394861` pts.

Transferable lesson: latent direct-output tasks may already avoid the obvious
label carrier; the remaining class-level optimization is often to avoid
computing unused selector/predicate scores.  For "select one of K blocks" tasks,
check whether all K predicates are actually needed.  With an exactly-one-outlier
promise, K-1 predicates may determine the selected block.
