# task160 — 6c434453

## 2026-06-29 mechanism screen

Rule: identify 3x3 sprite types in a 10x10 top-left canvas; convert box sprites
to red plus sprites and pass other blue sprites through.

Current source score: 17.804063 @ mem 1164 params 170. Dominant tensors are
`fg` [1,1,10,10] fp32 = 400 B and `features_u8` [1,4,10,10] = 400 B, followed by
three 10x10/8x8 uint8 feature planes. The final 1x1 `QLinearConv` writes directly
to graph output, avoiding counted 10-channel intermediates.

No rewrite adopted. Replacing the feature concat with separate output branches
would create counted 10-channel planes, and casting the full input before slicing
would be much larger than the current single-channel f32 slice.
