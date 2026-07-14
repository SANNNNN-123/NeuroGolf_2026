# task354 — ddf7fa4f

## Human-readable rule

**Confidence: verified on stored examples and fresh generator.**

Input/output are 10x10 visible grids embedded in the 30x30 one-hot scorer shape.
Row 0 contains three colored header/light pixels, in fixed column bands:
left columns 0-2, middle columns 4-5, right columns 7-9. Row 1 is blank.
Rows 2-9 contain up to three gray (`5`) rectangular boxes separated by black
background (`0`).

For each gray box, find the single header color whose column lies inside that
box's horizontal span. Recolor the whole gray rectangle to that header color.
Keep row-0 header pixels unchanged; all other visible cells remain background.

Stored examples:

- train0: header colors at columns 2/5/9 are `2/6/8`; left gray boxes spanning
  cols 1-2 become `2`, middle box spanning cols 3-6 becomes `6`, right box
  spanning cols 7-9 becomes `8`.
- train1: header colors `1/4/7`; a top wide box spanning cols 0-3 becomes `1`,
  bottom box spanning cols 3-6 becomes `4`, right box spanning cols 7-9 becomes
  `7`.
- edge case: boxes can share columns at different rows, so a single column-color
  propagation is invalid. The row-local contiguous gray run is the useful unit.

Fresh proof for the row-crop assumption: 1500/1500 generated instances had
`row0_gray=0`, `row1_nonzero=0`, and no gray in rows 0-1.

## Current state

Before this deep dive, `src/custom/task354.py` and `networks/task354.onnx`
matched the current live exact graph:

| artifact | pass/fail | memory | params | points |
|---|---:|---:|---:|---:|
| source pre-edit / live network | 266/0 | 2998 | 79 | 16.968289624677958 |

This source already contained the NEAR_18 "CumSum reset/run-boundary" idea,
implemented manually as `Split` + per-column `Add` over the non-gray mask:

1. Slice gray channel 5 over the 10x10 visible grid.
2. `gap = gray_channel == 0`, where true means non-gray/background boundary.
3. Build row-local segment IDs by cumulatively adding `gap` along columns.
4. Replace non-gray cells with sentinel `11`, leaving each contiguous gray run
   with a stable segment ID.
5. Find the segment ID under each header-light column band and recolor matching
   segments with the corresponding header color.
6. Pad a uint8 label plane to 30x30 with sentinel `11`, then final `Equal` with
   channel IDs produces scorer output.

Stored eval and fresh verification before this edit:

```text
measure_task.py 354 -> {'ok': True, 'pass': 266, 'fail': 0, 'memory': 2998, 'params': 79, 'points': 16.968289624677958, 'error': None}
fresh_verify.py 354 -> task354 arc=ddf7fa4f fresh_instances=1500/1500; incumbent fail = 0
```

## Candidate adopted in source only

I implemented a task354-scoped source-owned candidate in `src/custom/task354.py`.
It does **not** update `networks/task354.onnx`, `reports/manifest.json`, or any
global adoption artifact.

Change: crop gray segmentation to rows 2-9, because rows 0-1 are structural
header/blank rows. Reassemble visible label as:

```text
Concat(top_label row0, zero row1, rect_label rows2-9) -> label10
Pad(label10, sentinel=11) -> label30
Equal(label30, channel_ids) -> output
```

Verification:

```text
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py 354
{'ok': True, 'pass': 266, 'fail': 0, 'memory': 2674, 'params': 77, 'points': 17.080280239075424, 'error': None}

PYTHONPATH=. .venv/bin/python reports/scripts/fresh_verify.py 354
task354 arc=ddf7fa4f fresh_instances=1500/1500
  incumbent fail = 0
```

Payoff vs live/manifest incumbent: memory `2998 -> 2674` (-324), params
`79 -> 77` (-2), points `16.968289624677958 -> 17.080280239075424`
(+0.111990614397466).

## Cost anatomy after candidate

| component | tensors | bytes | semantic job | reducibility |
|---|---:|---:|---|---|
| final padded label | `label30` | 900 | expand 10x10 label to 30x30 with sentinel so final `Equal` emits all-false outside visible area | current dominant floor unless a cheaper direct 30x30 output construction is found |
| gray input slice | `gray_f` | 320 | read gray channel over rows 2-9, cols 0-9 | crop already applied; could only shrink if generator proves fewer rows/cols |
| header color reads | `left_top_f`, `mid_top_f`, `right_top_f`, scores/casts | 352 | collapse row-0 one-hot colors in the three fixed bands | possible tiny param/mem golf, not a near-18 lever |
| row-local segment stack | `gap`, split columns, casts/adds, `seg_id`, `seg_gray` | ~464 | give each contiguous gray run a row-local ID reset at non-gray boundaries | manual CumSum costs small per-column tensors; built-in `CumSum` may save only tens of bytes, not hundreds |
| segment selection and recolor | seed segment slices, seed reductions, same-segment masks, `right_fill`, `left_rect`, `rect_label` | ~548 | select the gray run under each header band and paint it with the band color | necessary for rows where boxes share columns but have different colors |
| visible label carrier | `label10` + `top_label` | 110 | combine header row, blank row, and recolored rectangles before final pad | small |

Total measured intermediate memory: 2674 bytes. Params: 77 initializer elements.

## Prior notes challenged

Still valid:

- Semantic rule: row-0 colored lights recolor gray rectangles by horizontal span.
- Pure column propagation is invalid because boxes can share columns at different
  rows with different colors.
- A row-local run/segment mechanism is the right abstraction.

Contradicted or stale:

- "Current stored before me: 15.26, mem 16860, params 147" is obsolete. Current
  manifest/live before this edit was `16.9683, mem 2998, params 79`.
- "Best achieved 16.75 @ mem 3780 params 57" is stale. The live graph had already
  moved past the MaxPool fill chain.
- "2-op run-fill via CumSum reset untried" is contradicted by the current graph:
  the incumbent uses reset-on-boundary segment IDs, manually expressed with
  split/add columns. The idea is verified, but the expected `~17.1` target was
  slightly optimistic for the uncropped 10-row implementation.
- "Dominant intermediate = horizontal run-fill chain, 8 fp16 planes" is no longer
  true. The dominant tensor is now `label30` at 900 bytes.
- "Crop fill to 8-row gray band is worse" was true for the older MaxPool-chain
  attempt but is false for the segment-ID graph. Row-cropping is a stored/fresh
  win here.

## Mechanism tests

### 1. CumSum reset / run-boundary segment IDs

- Expected payoff: replace fp16 MaxPool spread chain with row-local gray-run IDs;
  old estimate was roughly `mem 3780 -> ~2580` and `points ~17.1`.
- Proof test: current source/live graph before this edit used manual cumulative
  segment IDs and passed stored `266/0` plus fresh `1500/1500`.
- Kill condition: any stored/fresh failure on boxes with adjacent/unit-gap runs
  or shared columns.
- Result: verified mechanism, but actual uncropped cost was `mem 2998, params 79,
  points 16.9683`, not a full near-18 jump. The remaining floor moved to
  `label30` plus cropped segment-selection planes.

### 2. Bounded crop before scan/segment

- Expected payoff: remove rows 0-1 from all gray segmentation/selection tensors;
  expected memory drop about 300 bytes.
- Proof test: Python generator invariant check for rows 0-1, stored eval, and
  fresh eval.
- Kill condition: generator emits any gray in rows 0-1, row1 nonzero content, or
  candidate fails stored/fresh.
- Result: passes. Source candidate is `mem 2674, params 77, points 17.080280239075424`;
  fresh `1500/1500` with no row0/row1 invariant violations.

## Next experiment

Try replacing the manual `Split` + nine cast/add steps with ONNX `CumSum`
(`exclusive=1`) over the cropped `gap` mask, then add a scalar one. Expected
payoff is modest, likely 50-100 bytes and a few params, because the crop already
shrunk the segment stack. Kill it if ORT rejects uint8/bool CumSum or if the
extra cast/CumSum tensors exceed the removed split/add stack.

No reusable registry patch proposed from this agent beyond the existing lesson:
row-crop any run/scan mechanism after proving generator header/blank rows.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 2751 -> 2692 (+0.022)
Mechanism: Einsum + value_info Slice. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].