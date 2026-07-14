# task321 — cf98881b

**Rule:** Input is a 14-col x 4-row grid (size=4 -> width 3*size+2=14, height=4). Three 4x4 panels lie left-to-right, separated by single red(2) columns at cols 4 and 9: panel0 = cols 0..3 (colour 4, idx0), panel1 = cols 5..8 (colour 9, idx1), panel2 = cols 10..13 (colour 1, idx2). Each panel holds sparse coloured pixels. The output (4x4) overlays the three panels with FIXED priority panel0 > panel1 > panel2 (generator writes idx2 first, idx1, then idx0 -> lowest idx wins): `output[r][c] = 4 if panel0 has a pixel, else 9 if panel1 has, else 1 if panel2 has, else 0`. Colours are always the fixed triple (4,9,1).
**Current (prior stored):** ~16.40 pts, tier A
**Target tier:** B (label plane + final Equal). NOT tier S: background channel 0 = NOT(union of folded colour channels) is nonlinear; a single Conv can't route it. NOT clean tier A separable: the fold is a per-cell 3-way priority overlay, not a row⊗col rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 1x1 Conv idxmap[1,1,30,30] -> slice 3 panels -> priority Where fold -> Pad -> Equal (task372 idiom) | B | 4804 | 44 | 16.51 | 200/200 | correct but only +0.11 (idxmap f32 30x30 = 3600 B dominates) |
| 2 | channel-slice: colours FIXED so panel presence = single input channel sliced to its 4x4 region; priority Where on [1,1,4,4] -> Pad -> Equal | B | 1188 | 45 | 17.88 | 200/200 | WIN (+1.48). No 30x30 colour plane at all |

## Best achieved
17.88 @ mem 1188 params 45 — adopted? N (build-only). Beats prior ~16.40? **Y, +1.48**.

## Irreducible-floor analysis
Dominant intermediate: `Lp` (uint8 [1,1,30,30] = 900 B) — the padded label plane. It must be 30x30 to broadcast against the 10 colour channels in the final `Equal(Lp, arange[0..9])` that writes the free [1,10,30,30] BOOL output. Pad rejects bool, so the pad happens on uint8 before the Equal; 900 B is the floor for a single full-grid label plane. Everything else is tiny: three [1,1,4,4] f32 input slices (64 B each = 192), three Greater bools (16 B each = 48), three Where uint8 (16 B each = 48). The 30x30 output grid is fixed by the benchmark, so the 900 B plane is irreducible.

## OPEN ANGLES (re-attack backlog)
- Could the Pad be avoided by Equal-ing the 4x4 label then scattering into a 30x30 bool? Pad rejects bool and there is no cheap bool-scatter; the uint8 Pad route is already minimal.
- The three Greater ops could fold to one if the three channels were sliced into a single [1,3,4,4] tensor, but a non-contiguous channel set (4,9,1) can't be sliced in one Slice; a Gather(axis=1,[4,9,1]) then per-channel split adds ops without net byte savings. Not worth it.

## INSIGHT (transferable)
⭐ For a FOLD/overlay task whose panel/band colours are FIXED constants, skip the 1x1 colour-index Conv entirely: each panel's presence mask is a SINGLE input channel sliced to its region straight from the FREE one-hot input (no colour recovery needed), and the fixed output colour is just a [1,1,1,1] constant in a priority Where chain. This avoids the 3600 B [1,1,30,30] idxmap plane that the Conv-collapse idiom (task372) forces — here it cut mem 4804->1188 (+1.37 pts). Use Conv-collapse only when output colours COPY arbitrary input colours; when they're a known fixed set, channel-slice + constant Where is far leaner.
