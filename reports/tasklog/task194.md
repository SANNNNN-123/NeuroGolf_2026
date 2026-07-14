# task194 — 7fe24cdd

**Rule:** Fixed size=3 grid with 5..9 coloured pixels -> 2*size=6 output. Each
coloured pixel (r,c,v) is stamped at its 4-cell C4 rotation orbit about the 6x6
centre: out[r][c], out[2s-c-1][r], out[c][2s-r-1], out[2s-r-1][2s-c-1] = v.
The 4 orbit cells are always distinct, so it is a pure FIXED coordinate scatter:
each of the 36 output cells reads exactly ONE input cell (full coverage, no
conflicts). No value plane, no flood/argmax, all instances fixed 3x3->6x6.
**Current:** 17.67 pts, GridSample(nearest)+Pad import (ext:wguesdon6304), mem 1440, params 81
**Target tier:** S (pure spatial copy) — and it IS realisable at the copy floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | GridSample(nearest) [1,6,6,2] grid + Pad (prior verdict) | S-ish | 1440 | 81 | 17.673 | 200/200 | ties import, "at floor" verdict (WRONG) |
| 2 | Slice3x3->Cast u8->Reshape flat9->Gather(36)->Reshape6x6->Pad | S | 1260 | 58 | 17.816 | 34/34 | works but +0.15 (two 360B u8 planes) |
| 3 | Slice3x3->Cast u8->Reshape flat9->Gather([6,6] idx)->Pad | S | 900 | 54 | 18.139 | 200/200 | ADOPT-WORTHY (+0.47) |

## Best achieved
18.139 @ mem 900 params 54 — adopted? N (build-only, do not adopt). Beats prior 17.673? Y (+0.466).

## Irreducible-floor analysis
Invert the C4 scatter to a per-output-cell SOURCE map src[6][6] = r*3+c. The whole
output is then ONE Gather of the 9 flat input cells: a [6,6] index tensor along the
flattened spatial axis yields [1,10,6,6] directly (no final Reshape — that removed
the 4th attempt's extra 360B uint8 plane, the key step from 1260->900).
Dominant intermediates after that:
- fp32 input Slice [1,10,3,3] = 360B — UNSHRINKABLE: input dtype is fp32; all 10
  one-hot colour channels are possible (colours random 0-9). Slice-then-Cast is the
  only cheap entry (casting the full input is 9000B).
- uint8 Gather output [1,10,6,6] = 360B — the 6x6 one-hot block, irreducible.
- cheap helpers: g33u u8 90B, flat u8 90B.
900 total. This is the SAME 360B copy floor as the separable D2-mirror sibling
(task152, mem 990) — the non-separable rotation is absorbed by the flat 2-D-indexed
Gather instead of two axis Gathers, and it actually beats 152 by 90B (no per-axis
intermediate). The GridSample is NOT required and NOT at floor.

## OPEN ANGLES (re-attack backlog)
- Removing the `flat` Reshape via GatherND on [1,10,3,3] would save only 90B (the
  flat plane) and risks GatherND batch-axis complications; net ~ +0.05, not worth it.
- The two 360B planes (fp32 slice + uint8 6x6) are the genuine floor; no opset-11
  op produces a sub-fp32 sample of an fp32 input, so 360B entry is hard.

## S12 (2026-07-03) — attribute-form Slice/Pad param shave REJECTED
Tried moving the tiny `Slice`/`Pad` tensors into older-opset attributes to save
about 10 params.  `Slice` attributes are fine, but attribute-form `Pad` in
opset 9 rejects bool and uint8 inputs under ORT/scorer:
`Type 'tensor(uint8)' ... Pad ... is invalid`; bool fails shape inference too.
Current opset-18 `Pad` accepts the bool block but requires pads as an input
tensor.  No adoptable small-param shave found.

## INSIGHT (transferable)
⭐ A FIXED full-coverage coordinate scatter where EACH output cell reads exactly ONE
input cell (C4/Cn rotation, arbitrary permutation, non-separable remap) is NOT a
GridSample-floor task. Invert it to a per-output-cell SOURCE-index map and emit the
whole output as ONE Gather: a MULTI-DIMENSIONAL index tensor (shape = output spatial
block) on the flattened-spatial axis produces the output block in a single op, with
NO final Reshape. Cast to uint8 + uint8 free output. Floor = (fp32 input slice) +
(uint8 output block), same as a separable copy but it also handles NON-separable
geometry (rotation couples r,c) that two axis-Gathers cannot. This RETIRES the
"GridSample small-block = at floor" verdict for any fixed-geometry single-source
scatter. Prior tasklog "at-floor" claim was a stale assumption that GridSample was
the only realisation.
