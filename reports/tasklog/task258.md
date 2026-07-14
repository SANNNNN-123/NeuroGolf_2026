# task258 — a699fb00

**Rule:** Horizontal segments of odd `length` at row r, start col c. EVEN offsets are BLUE in both input and output; ODD offsets are RED in the output only. So input holds only blue pixels at c,c+2,c+4,...; output = input blue + RED at every background cell whose immediate left AND right neighbours are both blue (the one-cell gaps). Colours bg=0, blue=1, red=2. Active grid is in-canvas.
**Current:** 19.92 pts, single Conv W[10,5,1,3]+B[10] group=2, mem 0, params 160
**Target tier:** S (spatial closed-form, mem 0) — already there; question is only param count.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | minimal single Conv (reproduce floor) | S | 0 | 160 | 19.92 | 200/200 | exact floor, ties existing |

## Best achieved
19.92 @ mem 0 params 160 — adopted? N (ties existing, does not beat by +0.3). Does NOT beat prior 19.92.

## Irreducible-floor analysis
mem is already 0 (the Conv's [1,10,30,30] output IS the graph output; input/output tensors are free). Only lever is conv params. Three hard constraints make params=160 irreducible for a mem-0 single-Conv emission:
1. **Bg channel required.** Harness scores (output>0) per channel and demands exact one-hot equality, so ch0 (background) must be >0 on every bg cell — it is a non-copy channel that subtracts blue-neighbour mass to turn OFF the red-gap cells (W[0]: ch0 centre +0.04, ch1 left/right −0.02, bias −0.01).
2. **Cross-channel blue→red.** Red (out ch2) must read the BLUE input channel (in ch1). This forces ch1 and ch2 into the same Conv group, so `group ≤ 2` (in-dim = 10/group = 5). group=5/10 (in-dim 2/1) split blue and red into different groups → rule unrepresentable. Feasible groups checked: g=1→300 params, g=2→150, g=5/10→infeasible. So 150 W elements is the group floor.
3. **1×3 kernel.** Red footprint spans offsets {−1,+1} (width 3); a 1×2 kernel can read only one neighbour. Kernel is minimal.
W = 10·5·1·3 = 150 + bias 10 = 160 → score 25−ln(160) = 19.92. Any decomposition that emits <10 channels then Pads, or routes via Where/Equal one-hot, materialises a ≥2700B (bool 3-ch) / ≥3600B (fp32 index) full-canvas intermediate, which alone scores ≤17.1 — strictly worse than the 19.92 mem-0 net. So no path beats +0.3.

## OPEN ANGLES (re-attack backlog)
- None structural. The only sub-160 single-Conv would need group>2 with blue and red co-grouped, which ONNX even-split grouping forbids. A learned non-Conv mem-0 emission would still need a full-canvas op whose only free landing spot is the output — but producing 10 distinct channels (bg+blue+red+7 zeros) from blue/bg with a single output-emitting op other than Conv is not available (Where/Equal need a precomputed mask plane = mem).

## INSIGHT (transferable)
⭐ MEM-0 SINGLE-CONV-AT-FLOOR generalizes beyond cross-channel neighbourhood ops to **gap-fill / between-neighbours stamp** rules: when (a) the bg channel is a required subtractive channel (scored out>0), (b) one output colour reads a DIFFERENT input colour (forcing group≤2, in-dim=5), and (c) the footprint forces k≥3, the params floor is 10·5·k = 150(+bias). Decomposition always pays a ≥2700B intermediate that scores below the mem-0 net. BAIL INFEASIBLE. Quick discriminator before attempting: is the output expressible as ONE Conv over the 10-ch input whose output IS the graph output? If yes and group can't drop below 2, it's at floor.
