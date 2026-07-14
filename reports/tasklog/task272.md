# task272 — aedd82e4

**Rule:** A small grid (width,height each random 3..5, placed at top-left rows 0..h-1, cols
0..w-1) has random red pixels (color 2) on a black (0) background. The OUTPUT recolours every
red pixel that has NO orthogonal (4-neighbour) red neighbour to blue (color 1); red pixels with
at least one orthogonal red neighbour stay red. Off-grid cells count as empty. So: "recolour
isolated red pixels to blue" — a genuine 4-neighbour neighbourhood op.

**Current:** 18.87 pts, mem-0 single group-Conv (W[10,5,3,3], group=2, B[10]), mem 0, params 460.
**Target tier:** detection/floor — the highest admissible single-op is the mem-0 Conv already in place.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-canvas Slice(red)+plus-Conv+Where(iso,blue,input) | detection | 9900 | 29 | 15.80 | n/a | worse |
| 2 | 5×5 active-region + bool iso + fp16 Pad→cond + Where | detection | 3025 | 39 | 16.97 | n/a | worse |
| - | current mem-0 group-Conv (baseline) | floor | 0 | 460 | 18.87 | 200/200 | best |

## Best achieved
18.87 (baseline unchanged) — adopted? N (cannot beat). Beats prior 18.87? N.

## Irreducible-floor analysis
The output one-hot requires ch0=1 on EVERY black cell across the full 30×30 canvas (scorer compares
`out>0` against the full [10,30,30] target; black decodes as ch0). Any per-cell routing of the
recolour therefore needs at least one full 30×30 intermediate (a Where condition, an Equal index
plane, or a neighbour-sum Conv). The minimum cost of one full 30×30 plane is a bool plane = 900B
(calculate_memory takes itemsize from the DECLARED value_info dtype, line 133-135 — bool=1B — and
the trace only sets the shape, so a bool full plane really is 900B, not 3600B). 900B alone gives
score 25−ln(930)≈18.16 < 18.87, and the +0.3 target (19.17) needs mem+params ≤ e^(25−19.17) ≈ 340.
So NO full-plane decomposition can win. The only winning structure is mem-0 (no intermediate at
all), i.e. a single Conv whose output IS the graph output. The neighbour logic (center_red and the
orthogonal-red sum) forces a 3×3 kernel; the three needed output channels (0=black, 1=blue, 2=red)
all depend on input channel 2 (red), so the Conv must be group≤2 — group=2 (in/group=5) is the
coarsest grouping in which a contiguous output group {0..4} shares an input group {0..4} that
contains channel 2. That fixes the weight shape at [10,5,3,3] = 450 elements, which is irreducible
(params count ELEMENTS; fp16/zeros don't help) and already > 340. Hence the rule is at hard floor.

## OPEN ANGLES (re-attack backlog)
- None viable. The +0.3 target requires total < 340; one bool full plane is already 900, and the
  smallest mem-0 conv is 450. The two budgets cannot be reconciled for a 4-neighbour op whose
  output must paint ch0 over the whole canvas. Dropping the conv bias (−10 params → 450) is
  impossible because the bias carries the load-bearing >0 thresholds (−10.5/−1/+1).

## INSIGHT (transferable)
⭐ MEM-0 group-Conv recolour-by-neighbourhood is a HARD floor when (a) the op is a genuine
4-neighbour predicate (3×3 kernel forced) AND (b) the background channel-0 must be emitted across
the full canvas (so any decomposition pays a ≥900B full plane) AND (c) the active output channels
{black,blue,red} all read one input colour channel, forcing group=2 / W[10,5,3,3]=450. Confirmed
numerically: Where-route 15.80, 5×5-active+bool-pad route 16.97 — both far below the 18.87 mem-0
baseline. ⭐ Verified scorer fact: a bool [1,1,30,30] intermediate counts as 900B (itemsize from
declared dtype), NOT 3600B — but 900B still busts any "+0.3" budget that needs <340.
