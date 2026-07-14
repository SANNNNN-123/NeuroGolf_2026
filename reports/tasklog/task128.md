# task128 — 5521c0d9

**Rule:** A 15x15 grid (top-left on the 30x30 canvas) holds up to 3 solid axis-aligned
rectangles, each BOTTOM-ANCHORED (rows [size-tall, size)) with colour in {1,2,4}, never
overlapping horizontally. Output re-stamps each box UP by exactly its own height:
`output[r-tall][c] = colour`, i.e. the box is copied directly on top of its own footprint
(output rows [size-2h, size-h)); the original footprint becomes background. The op is purely
PER-COLUMN (width-1) and PER-CHANNEL (each colour shifts by its own run height) -> depthwise.

**Current:** 18.60 pts, dwconv59x1+b (group=10, [10,1,59,1]), mem 0, params 600
**Target tier:** A (single depthwise Conv; output is the FREE graph output, mem 0; only params shrink)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | trim kernel to box-only 15-tap (pt1,pb13) | A | 0 | 160 | 19.93 | — | WRONG: ignored ch0 (harness target sets ch0=1 on bg) |
| 2 | joint box+bg depthwise conv L=29 pt15 pb13 | A | 0 | 300 | 19.30 | 200/200, 300/300 | ADOPTED candidate |

## Best achieved
19.30 @ mem 0 params 300 — beats prior 18.60 by +0.70 (>= +0.3). Fresh isolated 300/300.

## Irreducible-floor analysis
Not at floor for the public net. The public net used a 59-tap symmetric kernel (params 600)
purely from brute-force fitting. The problem is per-column linearly separable in occupancy:
- COLOUR channels (run -> stamp-up-by-height) need a kernel reaching DOWN 13 rows (pb=13).
- BACKGROUND channel 0 (bg-after-transform, scored because the harness one-hot target sets
  ch0=1 on every in-grid bg cell) needs to reach UP 15 rows (pt=15) to know if a cell is now
  covered by the relocated box. A single ONNX Conv shares ONE (L, pads); the joint minimum that
  separates BOTH is L=29, pads(top=15, bottom=13) -> params 300. Below L=29 (joint) one of the
  two sub-problems is no longer sign-separable by a single linear filter (perceptron-verified:
  box needs pb>=13, bg needs pt>=15).

## OPEN ANGLES (re-attack backlog)
- Cross-channel ch0 = NOT(any box channel) via a group=1 conv or post-op would let the colour
  kernels shrink to 15 taps (params ~160) — but any extra op materialises a non-free plane (mem
  cost) and a group=1 conv needs a 10-in kernel; net likely worse than 300-param depthwise.
- Could a non-Conv per-column op (e.g. shifted-Gather by a recovered run-height scalar) beat
  300 params? Run height per column is data-dependent (a [1,1,1,30] vector), Gather-by-index
  would need an int index plane — likely costs more mem than the 0-mem conv. Not pursued.

## INSIGHT (transferable)
⭐ When a per-channel/per-column shift-by-local-run-length is linearly separable in occupancy,
a learned/brute-forced full-span depthwise Conv (here 59 taps) can be TRIMMED to the minimal
linearly-separating kernel via a margin-perceptron over the small set of canonical column cases
(here h=0..7). Params shrink with the kernel height and the output stays mem-0.
⭐ The harness one-hot TARGET sets channel 0 = 1 on every background cell (color 0), so the
background channel is SCORED and must be reconstructed too — you cannot just suppress ch0 with a
negative bias. A single ONNX Conv shares ONE (kernel_height, pads) across all channels, so the
adopted kernel size is the JOINT minimum over the foreground AND background sub-problems, not the
foreground minimum alone (this is what bumped 160 -> 300 params here).
