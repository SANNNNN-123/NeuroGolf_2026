# task019 — 10fcaaa3

## 2026-06-29 single-Conv probe

Current source score: 17.063340 @ mem 2709 params 89.

Rule: copy sparse coloured points from an HxW input into four quadrants of a
2H x 2W output and draw cyan diagonal neighbours around each copied point, with
coloured points overwriting cyan.

The tempting 20+ mechanism is a single Conv: point colour copies and cyan diagonal
stencils are both local/linear, and cyan overwrite could in principle be handled
with negative centre weights.  Ran `reports/scripts/conv_fit.py 19`; result:

- k=1 failed, channel 0 not separable on 300 fresh examples
- k=3 failed, channel 0 not separable on 300 fresh examples
- k=5 failed, channel 0 not separable on 300 fresh examples

No rewrite adopted.  The blocker is the black/background channel: the output
footprint is 2H x 2W, while H and W are data-dependent and must be recovered from
the input rectangle.  Colour/cyan are local, but the footprint mask is not a fixed
translation-invariant stencil.
