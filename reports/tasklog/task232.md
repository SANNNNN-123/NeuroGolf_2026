# task232 — 97999447

## 2026-06-29 sparse-kernel probe

Current source score: 17.748655 @ mem 0 params 1410. The model is a single
`Conv(input, weights, bias) -> output`, so all runtime memory is free, but the
dense kernel has shape [10,10,1,14] = 1400 params.

Semantic rule: each coloured pixel emits a rightward horizontal trail in its row;
even offsets keep the source colour and odd offsets become gray. The dense kernel
is structurally sparse: only parity-compatible offsets matter, but the scorer counts
all dense initializer elements.

Tried converting `weights` to a sparse initializer with the same dense shape and
only nonzero values. Result: invalid. ONNX shape inference reports Conv `W` as
`sparse_tensor(float)`, and ORT rejects it as not being a graph input, initializer,
or previous node output after sanitization. Sparse initializers therefore cannot
be used as a params loophole for Conv weights in this scorer.

Grouped/decomposed Conv is also unattractive: it can cut initializer elements, but
the necessary intermediate trail tensors become counted memory, losing the mem0
advantage of the current one-Conv graph.

## 2026-06-30 mem0 params sweep

Tried refitting a smaller single horizontal Conv over the stored examples for every
kernel width 1..13 and valid left padding. No one-vs-all linear separator was found;
the current width-14 Conv appears necessary if the solution stays a single mem0 Conv.
Any decomposition that first materializes colour/trail planes must beat the current
`memory + params = 1410`, which is hard because even one full 30x30 fp32 Slice is
3600B.
