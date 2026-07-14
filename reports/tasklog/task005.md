# task005 — 045e512c

## 2026-06-29 mechanism screen

Rule: recover a 3x3-ish sprite from the centre stamp and first directional hints,
then repeat the full sprite every 4 cells along two or three sampled directions.

Current source score: 16.163335 @ mem 6392 params 490.

Dominant tensors from source eval:

- `colors21_f32` [1,1,15,15] fp32 = 900 B
- `color30` [1,1,30,30] uint8 = 900 B
- `ray_total` [1,1,23,23] uint8 = 529 B
- `color21` [1,1,21,21] uint8 = 441 B
- `colors21` [1,1,15,15] uint8 = 225 B

The current graph is already a semantic ray compiler: it extracts a colour-indexed
15x15 view, infers the 3x3 template, builds an 11x11 block grid of directional ray
seeds, then stamps with a final 3x3 `QLinearConv`. `onnxsim` gave no score change
on task005. The tempting full-canvas `color30` label plane is not a free win:
switching to a 10-channel bool assembly before pad would be larger, and the final
`Equal` output is already free.

Open only with a new proof about generator bounds. Reducing the 23x23 ray lattice
or dropping a direction bank is rare-fail prone because the generator samples two or
three directions and allows centre positions 6..12 on a 21x21 canvas.

## 2026-06-30 quick mechanism check

User-facing rule read: a 21x21 canvas contains a central 3x3-ish sprite/template
and two or three coloured directional hints.  The output copies the recovered
template every 4 cells along the hinted directions, preserving each hint colour.

Tried folding the final `QLinearConv(ray_total)->color21` plus
`Pad(color21, value=255)->color30` into one asymmetric-padded `QLinearConv` that
emits `color30` directly.  This would have removed the 441B `color21` plane and
measured `mem=5951`, but it fails all stored examples: convolution padding emits
zero, while the 21x21-to-30x30 off-grid area must be sentinel 255 so final
`Equal(color30, palette)` produces all-false outside the true canvas.  A zero
pad incorrectly turns off-grid into background colour 0.

Conclusion: the final scalar `Pad(value=255)` is semantically real unless we add
a separate validity mask, which is larger than the 441B saved.

## 2026-07-01 re-adjudication (borrowed-net pass)

Independent re-measure: mem=6392 params=490 pts=16.163. Per-tensor: 5 tensors
>=100B sum 2995 (colors21_f32 900 fp32 read, color30 900 carrier, ray_total 529,
color21 441, colors21 225); remaining 3397B spread over 183 sub-100B ray-assembly
slices/blocks (each tiny, bespoke).

Floor proof: the 900B fp32 Conv read is the detection floor (15x15 sampled colour
map x4); the 900B color30 is the BACKGROUND-CHANNEL CARRIER FLOOR — output is a
genuine 2-D multi-colour stamp field (not separable rects), so the final
Equal needs a [1,1,30,30] index plane with ch0=1 across the in-grid rect (no
signed-Einsum/strip route exists). Together 1800B are irreducible; the remaining
~4100B is the hand-written 8-direction ray assembler. Prior QLinearConv+Pad
fusion (−441B) already shown to break the 255-sentinel pad. 

NEW: incumbent FAILS 24/800 fresh arc-gen instances (~3%) — it is itself an
imperfect re-fit, not a faithful generaliser. A cheaper replacement would have to
be both cheaper AND strictly more correct than a 200-node bespoke assembler;
infeasible. VERDICT: FLOOR (no cheaper equivalent landable).

## 2026-07-01 task001-insight pass

Rechecked with the task001 strategy: defer colour one-hot expansion to the final
free `output`, avoid full-canvas carriers unless they are scalar, and look for
ray/template intermediates that can be fused away.

Current source/live:

- **memory 6392, params 490, pass 266/266, points 16.163335413642574**.

Measured dominant intermediates:

- `colors21_f32 [1,1,15,15] fp32`: **900**.  This is the colour-index detection
  entry plane from the one-hot input.
- `color30 [1,1,30,30] uint8`: **900**.  This is the scalar colour/sentinel
  carrier before final `Equal(color_values) -> output`.
- `ray_total [1,1,23,23] uint8`: **529**.
- `color21 [1,1,21,21] uint8`: **441**.
- `colors21 [1,1,15,15] uint8`: **225**.
- The remaining memory is spread across many 7x7, 3x3, and row-concat ray
  assembly slices.

Task001-style conclusions:

- Colour handling is already in the right form: keep a scalar colour index until
  the final `Equal`, which writes the free one-hot output.  Expanding to
  10-channel bool before padding would be much larger than `color30`.
- The tempting `QLinearConv(ray_total) -> color30` fusion was already tested and
  fails because convolution padding emits zero, while off-grid must be sentinel
  255 before final `Equal`; replacing that sentinel with a validity mask costs
  more than the saved `color21` plane.
- The memory is not dominated by a single removable accidental carrier.  It is a
  mix of one scalar full-canvas carrier plus a bespoke ray assembler.  A real
  improvement likely needs a stronger generator-bound proof that shrinks the
  23x23 ray lattice or removes a direction family, not just output fusion.

Conclusion: no new adoptable improvement from the task001 insight pass.

## 2026-07-01 deep fresh-failure routing analysis

Instrumented the arc-gen generator to return sprite/removal/direction metadata and
classified 30k fresh samples against the incumbent.

Fresh failure is not concentrated in one sprite-removal family.  It is mostly a
ray-routing miss:

- 30k sample: **1009/30000 fail = 3.36%**.
- Individual direction failure signal: expected `south (1,0)` pixels dominate
  the missing foreground set; apparent "extra" pixels are mostly the background
  channel becoming true where the foreground channel was missed.
- Concrete trace on a failing stored-style fresh instance showed
  `marker_match_gated_s` is already correct: the south marker is present in the
  7x7 marker map.  The failure occurs when fixed `Slice` windows copy that marker
  into the 23x23 `ray_total` lattice; for one example the marker at `[5,3]` was
  outside all south slice windows, so the later `QLinearConv` had no south seed.

Probe: add an in-memory south correction by padding `marker_match_gated_s` into
23x23 seed planes for the k=1/k=2/k=3 south repeats, then `Max` with
`ray_total` before the existing stamp convolution.

- Stored: pass 266/266.
- Cost: **mem 8529, params 518, points 15.8898** versus incumbent
  mem 6392, params 490, points 16.1633.
- Fresh 2000: incumbent fail 59, candidate fail 25.
- Fresh 8000 on the south-corrected candidate: fail 101; remaining misses are
  now dominated by `southwest (1,-1)`.

Conclusion: the current task005 graph is an approximate bespoke ray compiler.
The likely exact repair is to add similar padded seed correction banks for the
diagonal/axial directions, but that increases memory by multiple 23x23 planes
and is not a score improvement.  No adoptable optimization found.  The reusable
lesson is to trace marker-to-lattice routing separately from marker detection:
if a task uses fixed block routing, fresh misses may come from slice-window
coverage rather than semantic detection.

## 2026-07-03 center-crop / 2x2 direction-probe check

User hypothesis: since the source sprite is a 3x3 shape with a few missing
cells, recover the center from the middle only and detect directions by checking
the eight 2x2 regions adjacent to the center sprite, instead of building the
shared 15x15 colour map.

Generator-bound results:

- Center top-left is always 6..12, so the center sprite itself is always within
  rows/cols 6..14.  A 9x9 center crop is enough for shape recovery.
- Direction probes need offsets about -3..+5 from the center top-left across
  all possible center positions, so the absolute envelope is rows/cols 3..17.
  That is exactly the incumbent `colors21` 15x15 sampled colour map.
- For every generator-valid missing-cell mask and valid direction, there is a
  fixed 2x2 probe inside the first directional sprite that intersects the
  visible hint.  The hypothesis is semantically valid for direction detection.

Cost check:

- Incumbent shared 15x15 colour map: Conv weight [1,10,6,6] = 360 params,
  `colors21_f32` 900B + `colors21` 225B = 1125B.
- Center-only 9x9 colour map would reduce memory to 405B, but still costs
  640 params with the needed dilated sampling and does not include direction
  hints.
- Direct 8-direction 2x2 probe maps with dense Conv would need output only
  8x7x7, but ONNX counts the dense [8,10,24,24] kernel: 46080 params, fatal.
- Building 2x2 probes from the incumbent 15x15 map adds about 392B and extra
  Slice/Max nodes unless it deletes later routing; it does not.

Conclusion: the 2x2-probe idea is a good semantic simplification for direction
recognition, but it is not an adoptable score optimization in ONNX.  The
incumbent 15x15 dilated colour map is already the cheap way to share all center
and direction samples; the remaining cost is still direction-to-ray routing, not
direction detection.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 6801 -> 6763 (+0.006); gate inc/cand=48/48 (safe). See [[neurogolf-urad-7225-bundle-vein]].