# task060 — mem0 horizontal Conv params probe

Current source score: 18.082294 @ mem 0 params 1010. The live graph is a single
horizontal Conv with kernel shape `[10,10,1,10]` and pads `[0,5,0,4]`.

## 2026-06-30 mem0 params sweep

Tried refitting a smaller single horizontal Conv over stored examples for every
kernel width 1..9 and valid left padding. No exact separator was found. The current
width-10 single-Conv form appears necessary if the graph remains mem0.

The usual sparse-kernel escape does not help under this scorer: params count dense
initializer elements, and replacing the Conv by sliced/sparse semantic planes would
pay at least one full-canvas intermediate, usually more than the 1010 params saved.
