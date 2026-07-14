# task297 — bd4472b8

## 2026-06-29 final-expansion screen

Current source score: 17.763661 @ mem 1350 params 39.

The graph builds a compact [1,9,14,6] bool one-hot tensor (`compact`, 756 B) and
pads it directly to the free graph output.  This beats the alternative label-map
route: a [1,1,30,30] uint8 label alone would cost 900 B before the final `Equal`.

No rewrite adopted.  This is a useful counterexample to the usual label-map rule:
when the active footprint is very small and only non-black colour channels are
needed, compact one-hot + final Pad can be cheaper than full-canvas label + Equal.
