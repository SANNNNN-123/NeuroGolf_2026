# task315 — cce03e0d (self-referential fractal placement)

**Rule:** Input is a fixed 3x3 grid with colours in {0,1,2} (red==2). The 9x9
output is a fractal: block (r,c) (output rows 3r..3r+2, cols 3c..3c+2) holds a
full copy of the input grid IFF input[r][c]==2 (red); every other block is all
background. Equivalently `output[u,v] = input[u%3, v%3] if input[u//3,v//3]==2
else 0`. The input always occupies the top-left 3x3 of the canvas. Colours 1,2
only.
**Current:** prior 16.77. This session: **17.44 pts, custom kron label-map
(colour-index + Kronecker index maps + Equal), mem 1710, params 202.**
**Target tier:** B (label map + final Equal). Tier S/A blocked: output cell value
is `S[u%3,v%3]` GATED by `S[u//3,v//3]==2` — the kron 2-factor (macro,micro)
index map is not a single row⊗col separable rectangle, and output preserves
arbitrary input colours (1/2) so a fixed permute/copy (tier S) can't route. B is
highest admissible.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf via Mul(full slice [1,10,3,3])+ReduceSum; fp32 9x9 gathers; uint8 Pad | B | 2664 | 210 | 17.04 | 200/200 | correct but +0.27 only (MARGINAL) |
| 2 | colf via 1x1 Conv (drop Mul plane); cast Sflat fp16 (gathers 324->162) | B | 1998 | 210 | 17.30 | 200/200 | +0.53 |
| 3 | slice only channels 1..2 ([1,2,3,3]=72B), Conv kernel [1,2] | B | **1710** | **202** | **17.44** | **400/400** | BEST |

## Best achieved
**17.44 @ mem 1710 params 202 — fresh 400/400 (isolated temp-net).** Beats prior
16.77 by **+0.67**. Adopted? N (build-only per brief).

## Irreducible-floor analysis
Dominant intermediate: the **900 B uint8 Pad** (the 30x30 label map L feeding the
final Equal). uint8 is the smallest dtype and the Equal must span the full 30x30
output footprint; ORT Pad rejects bool so the 9x9 bool Equal (810 B) can't be
padded directly — 900 is the canonical label-map floor (same as task195). The
rest is tiny: in33 [1,2,3,3] fp32 72, two fp16 9x9 gathers 162 each, 81-byte
bool/uint8 9x9 maps, [1,1,3,3] colf 36.

## OPEN ANGLES (re-attack backlog)
- Drop the 900 L-pad: assemble the 30x30 output without a uint8 carrier (Concat /
  ScatterND of 10 channels from the 9x9 bool) — but per task195 this costs MORE
  than 900; no clean sub-900 final found.
- in33 72 B fp32 slice is already minimal for a 2-channel read; casting to fp16
  would ADD bytes (Slice emits fp32 first). Net neutral.

## INSIGHT (transferable)
⭐ A "self-referential fractal" (place a copy of the whole grid into each cell
that meets a per-cell predicate) is the **task195 kron idiom with a GATE**: the
block selector `Smac=Sflat[macro]` becomes a boolean mask via the predicate
(`==red`) instead of an `AND` with the same sprite, and `Smic=Sflat[micro]`
carries the actual input colour. ⭐ When only 2 fg colours appear, slice ONLY
those channels ([1,2,3,3]=72 B) and fold the colour ramp into a 2-weight Conv
kernel — strictly cheaper than slicing all 10 channels + a full colour-ramp Conv.
