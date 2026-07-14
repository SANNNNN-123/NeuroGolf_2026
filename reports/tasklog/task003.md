# task003 — 017c7c7b

**Rule:** A height=6, width=3 grid holds a vertically-PERIODIC blue (1) stencil with
period `steps`∈{2,3}; for steps==2 the column pattern may flip L-R once per period
(`flip`). Output is height=9, width=3: same stencil recoloured RED (2) and extended to
9 rows by the same offset+flip schedule. Off the 9×3 grid all channels are 0.
Output rows 0–5 == input rows 0–5. The continuation rows reuse rows the input already
shows, so NO flip handling is needed: steps3 (offsets 0,3,6) → out6,7,8 = in0,in1,in2;
steps2 (offsets 0,2,4,6,8, flip toggling) → out6,7,8 = in2,in3,in0. Only one scalar
needed: is3 = shift-by-3 matches (in3==in0,in4==in1,in5==in2).

**Current:** 17.92 pts, ext:kojimar6275, mem 1172, params 18
**Target tier:** S-ish (spatial copy + 1 scalar) — output is a pure recolour+periodic-copy
of input cells; the only data-dependent choice is steps2 vs steps3.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | uint8 copy block, period2-match detection | A | 745 | 33 | — | — | WRONG (flip breaks period-2 match) |
| 2 | shift-by-3 detection + flip-row select | A | 731 | 33 | — | — | WRONG (out6/flip parity wrong) |
| 3 | corrected extension table (out6,7,8 = in0/1/2 or in2/3/0) | A | 720 | 33 | 18.38 | 200/200 | PASS |
| 4 | fully-uint8 copy (bg/zeros via Equal not Sub) | A | 633 | 34 | 18.50 | 200/200 | ADOPT |

## Best achieved
18.50 @ mem 633 params 34 — beats prior 17.92 by **+0.58**. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate = the pre-Pad [1,10,9,3] uint8 block (270B) — floor for "pad a
10-channel 9×3 one-hot block to the 30×30 output"; the 7 always-zero channels still
count but the channel dim must be 10 to match the output. Next is the fp32 ch1/6×3
slice (72B) — Slice inherits the fp32 input dtype, 18 elems × 4B, irreducible entry.
Everything downstream is uint8 (one-hot {0,1}, scored out>0). Detection runs on tiny
fp16 6×3 row slices (~6B each).

## OPEN ANGLES (re-attack backlog)
- Pad the small block is already the cheapest output route; building a colour-index
  plane + Equal-to-arange is WORSE here (the padded full-canvas [1,1,30,30] plane = 900B
  > 270B). No obvious sub-633 angle remains; the 270B 10-ch block is the wall.

## INSIGHT (transferable)
⭐ For a periodic-EXTENSION task, do NOT model the flip/offset parity abstractly — the
generator's continuation rows are usually IDENTICAL to rows the input already contains
(same offset-parity & flip-state), so the extension collapses to a fixed lookup table of
input rows selected by ONE period scalar (here shift-by-3 match). Detect the period by an
exact shift-match (Sub→sq→ReduceSum==0), NOT by a period-2 tiling match — the latter is
broken by per-period column flips even though the period IS 2.
⭐ UINT8 WHOLE-PIPELINE for a one-hot copy/recolour: build bg = Equal(red,0) and a zeros
plane = Equal(red, <unused value>) since uint8 Sub/Mul are rejected but uint8 Equal/Where/
Concat/Pad/Slice all run under ORT_DISABLE_ALL — keeps the whole copy block at itemsize 1.

## 2026-07-01 task001-insight pass

Rechecked the current source/live graph after later micro-golf.  Current status
is better than the older 633-memory log entry:

- source/live: **memory 281, params 21, pass 265/265, points 19.28957298262513**.
- dominant intermediates: `valid3 [1,3,9,3] uint8 = 81`, `one6_f` input slice
  `72`, three 9x3 one-channel planes `27` each.

Applied the task001 lessons:

- Direct final-output routing would avoid `valid3`, but needs 9x3-to-30x30
  spatial selectors or a full-canvas carrier.  That is larger than the current
  81-byte pre-Pad block.
- Colour domain is already minimal: only channels 0 and 2 are semantically live,
  but channel 1 must be represented as an explicit zero plane before `Pad`
  because `Pad` cannot insert an internal channel gap.
- A bool one-hot variant was tested (`Equal/Not/Concat/Pad` with bool output).
  It passes stored examples but is worse: **memory 308, params 22**.  Bool and
  uint8 both cost one byte per element, and the bool route adds conversion
  intermediates.

Conclusion: no adoptable task001-style improvement found for task003.  The
current `uint8 small-block -> Pad(output)` route is the right tradeoff because
the small carrier is only 81 bytes; replacing it with direct 30x30 routing costs
more in params or memory.

## 2026-07-01 deep generator/scorer recheck

Generator `/tmp/arc-gen/tasks/task_017c7c7b.py` is input-deterministic: `steps`,
`rows/cols`, and optional flip fully determine both input and output, and the
visible six rows contain enough information to choose the 9-row continuation.
Incumbent verification: **1000/1000 fresh, 0 fail**.

Rechecked possible scorer edges:

- `Split(one6)` instead of four row `Gather` indices removes 4 params but adds
  two extra row outputs.  Candidate passes stored but is worse:
  **memory 287, params 17, points 19.282972** versus current
  **memory 281, params 21, points 19.289572**.
- Tried old-opset attribute encoding where `Slice.starts/ends` and `Pad.pads`
  are node attributes rather than counted initializers.  This could remove 16
  params in theory, but opset 9 type support breaks the cheap route:
  `Pad(bool)` invalid, `Equal(uint8)` invalid, and `Sub(uint8)` invalid.  Mixed
  bool/uint8 variants need extra casts and are worse than current.

Conclusion: task003 is genuinely near its local floor.  The current graph's 21
params are mostly unavoidable structural constants for crop/pad and row selection;
attempts to move those constants into op attributes lose the modern uint8/bool op
support that makes the 281-byte memory path cheap.
