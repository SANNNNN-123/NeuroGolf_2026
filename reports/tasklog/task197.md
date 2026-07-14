# task197 — 82819916

## 2026-06-29 colour-kernel screen

Current source score: 17.135964 @ mem 2586 params 16.

Rule: infer a binary pattern from the first coloured row, then complete later rows
using each row's two observed colours.

The graph recovers two per-row colour sequences (`row_color0/1`) and uses
`ConvInteger` with runtime colour kernels to write directly to output.  Dominant
costs are the two [1,10,14,1] f32 slices (560 B each) and their u8/transposed
kernel forms.

No rewrite adopted.  A float Conv variant would make the runtime colour kernel
4x larger, and the 14-row maximum is required by the generator height bound.
