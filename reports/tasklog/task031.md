# task031 — 1cf80156

**Rule:** Fixed width=12, height in {10,11,12}. The grid holds ONE connected blob
of 8..12 pixels in a SINGLE foreground colour on a background of 0 (corner cell
input[0][0] is bg). The output is the tight bounding box of the blob, cropped to
the top-left corner of a fresh grid — background 0 fills the holes inside the bbox;
everything outside the HxW box is all-channels-off. Measured 0/20000: exactly one
non-zero colour present, bg=0, output bbox <= 9x9.
**Current (public):** 15.17 pts, gen:vyank6322, mem 18558, params 58
**Target tier:** detection→B (variable-size crop translated to origin → needs a
Gather-shift + final Pad; not separable/single-conv). But identification is trivial
(single fg colour, no noise, no min-span argmin needed), so it lands well below the
detection floor at a single fp32 colour-plane entry.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf=Σk·input_k → occupancy profiles → (min_row,min_col,H,W) scalars → 9×9 window Gather → boxmask → Pad(10) → Equal | B | 7273 | 123 | **16.09** | 200/200 | WIN, beats 15.17 by +0.92 |

## Best achieved
**16.09 @ mem 7273 params 123 — fresh 200/200, stored 266/266.** Beats public 15.17
by **+0.92** (≥0.3 ✓). Adopted? **N** (main adopts via `python -m src.adopt 31`).

## Irreducible-floor analysis
Dominant intermediate: the **3600 B fp32 `colf`** = `sum_k k·input_k` (1×1 Conv),
the per-cell colour-index plane. Irreducible because the 9×9 crop WINDOW position
(min_row,min_col) is data-dependent, so the full 30×30 plane must materialise before
the window Gather; fp32 is forced (ORT ReduceMax/Conv output fp32, can't pre-narrow
without paying an extra cast). The remaining ~3.6KB is the 9×30 fp32 `Vr` row-window
(1080 B), the two 1-D occupancy ReduceMax profiles (120 B each), and tiny scalars.
The fp16-downstream trick does NOT help here: downstream full-canvas planes are
already cropped to 9×30 / 9×9, so casting colf→fp16 would only ADD 1800 B without
removing the 3600 B fp32 entry.

## OPEN ANGLES (re-attack backlog)
- Tier A/S blocked: output is a data-dependent translate of a variable-size bbox
  (extent + offset depend on input content), so no single fixed Conv/permute/
  separable row⊗col mask produces it — the Gather-shift is mandatory.
- Could shave the 1080 B `Vr` by Gather'ing cols before rows (9×9 directly) — but
  axis-2 Gather on a 30-wide plane first is symmetric; sub-0.1 pt, not worth it.

## INSIGHT (transferable)
A single-colour, single-blob "crop to bbox" task is the EASY twin of task036: with
no noise and one fg colour you SKIP the entire min-span argmin / blob-colour-select
machinery — just `colf=Σk·input_k`, recover (min_row,min_col,H,W) from 1-D occupancy
profiles, Gather a small fixed-size window (size = generator's max-bbox bound, here
9×9), gate with (r<H)∧(c<W), Pad with sentinel 10, final Equal→bool. Floor = the one
3600 B fp32 colour plane forced by the data-dependent crop position.

## S9 (2026-07-03) — kojimar 7184.85 teacher (overrides/) ADOPTED (+0.761)
Log-space bbox extents: free-input Einsum weighted by powers → Log/Div/Floor decode
min/max row/col as scalars (no 30×30 profile planes); Slice crop; ConvInteger epilogue
(u8 feed + u8 weight — designed dtypes, safe; NO TopK). mem 1426→578, params 46→110.
Gates: stored fail=0 (re-checked); fresh 6000 uncached: both 0. base_submission variant
= our own mechanism, ignore. Backup reports/retired_networks/task031_pre_s9.onnx.
