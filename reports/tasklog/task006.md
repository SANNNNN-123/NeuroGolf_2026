# task006 — binary overlap through separator

## 2026-06-30 deep audit

Visible rule: the input is a 3x7 grid made of a left 3x3 binary blue mask, a
gray separator column, and a right 3x3 binary blue mask.  The output is a 3x3
red mask at cells where both left and right masks contain blue; otherwise the
cell is black.

Current graph is already extremely compact:

- slice channel-1 left 3x3 and right 3x3;
- cast both to bool;
- `And`;
- `Where` between a 3-channel black/red mini-output;
- pad the 3x3x3 mini-output to the official 10x30x30 output.

Current score is `19.969562078607566` with `memory=126`, `params=27`, total
cost `153`.  Crossing 20 points needs total cost at most about `148`, so only 5
cost units must be removed.

Tried a legal-looking ONNX representation change: encode `Slice` and `Pad`
indices as opset-9 attributes instead of counted initializer tensors.  This
would have dropped enough params to pass 20 if legal.  Probe:
`reports/scripts/task006_opset9_attr_probe.py`.

Result: rejected by ORT/scorer because opset-9 `Pad` does not accept
`tensor(uint8)` input:

`Type Error: Type 'tensor(uint8)' of input parameter (...) of operator (Pad) is invalid.`

Float Pad would be legal but expands the 3-channel 3x3 mini-output from 27B to
108B, more than wiping out the saved params.  Bool/Concat constructions also
replace initializer cost with extra 3x3 bool planes and are worse than the
current `Where`.

Conclusion: current task006 is a true boundary case.  The remaining 5-unit gap is
not semantic; it is the legal `uint8 Pad` representation cost.  A future route
would need either a counted-param-free `uint8` pad equivalent or a way to emit the
3x3 black/red one-hot directly to the official output without materializing extra
planes.  The tested opset-attribute route is not legal.

## 2026-07-01 sequential deep pass

Fresh recheck: **1000/1000 pass**.

Reconsidered the remaining 5-cost gap with task001-style direct-output thinking:

- A single wider slice of the blue channel could reduce Slice index params, but
  it materializes a `[1,1,3,7]` float slab plus split outputs.  This is larger
  than the two precise `[1,1,3,3]` float slices.
- Replacing `Cast+And` with arithmetic (`Mul` or summed threshold) keeps a
  float `[1,1,3,3]` product/sum plane, which is worse than two bool casts plus a
  bool `And`.
- A direct full output `Conv`/`Einsum` would need either dense coordinate weights
  or a full output-coordinate selector.  For this tiny 3x3 task, the existing
  3-channel uint8 `Where` plus `Pad` remains cheaper.

Conclusion unchanged: no adoptable improvement found.  Current cost is
`memory=126, params=27` and the blocker is ONNX representation overhead, not the
semantic rule.
