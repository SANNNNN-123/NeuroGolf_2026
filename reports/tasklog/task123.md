# task123 — 539a4f51

**Rule:** `colors` = list of L colours (L∈{4,5}, every colour 1..9). INPUT (5×5):
`grid[r][c]=colors[max(r,c)]` for `max(r,c)<L` else 0 → the input's column 0 (and
main diagonal) over rows 0..4 is exactly `colors[0..L-1]` then 0s; `L=#(nonzero)`.
OUTPUT (10×10): `output[r][c]=colors[max(r,c) % L]`; all output cells are 1..9 (no
background inside the 10×10 footprint), and the 30×30 canvas outside the 10×10 is
all-channels-off.
**Current:** 16.68 pts (public label-map floor), method label-plane.
**Target tier:** B-broken — output value depends only on the scalar `max(r,c)`, so the
whole label is a const-index Gather of a tiny length-10 band-colour vector; the only
30×30 intermediate is the final uint8 label being Equal'd into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 5×5 colf diag + mod-plane + Gather, pad 0 | B | 3849 | 148 | fail | — | off-grid bug (pad 0 lit ch0; footprint is exactly 10×10) |
| 2 | sentinel pad 10, fmod=1 Mod | B | 3849 | 148 | 16.71 | — | passes but only +0.03 |
| 3 | col-0 slice [1,10,5,1] not 5×5; bidx int32 | B | 2469 | 143 | 17.13 | — | +0.45 |
| 4 | band-colour vector (Mod on len-10, not plane); const maxidx Gather | B | 1889 | 157 | 17.38 | — | +0.70 |
| 5 | fp16 then uint8 band-colour Gather (uint8 lab plane) | B | 1499 | 157 | **17.59** | 200/200 | **ADOPT-candidate +0.91** |

## Best achieved
17.59 @ mem 1499 params 157 — adopted? N (build-agent; report only). Beats prior 16.68? **Y (+0.91)**.

## Irreducible-floor analysis
Dominant intermediate = the 30×30 uint8 label plane fed to `Equal` (900 B). It is
irreducible: the output value_info is fixed [1,10,30,30], so the per-cell colour-index
label MUST be materialised at 30×30 before the 10-ch one-hot expansion is routed into
the FREE bool output; uint8 is already the minimum dtype ORT `Equal` accepts. Everything
else is tiny — the colour recovery (col-0 slice 200 B), the length-10 band vector, and a
10×10 uint8 label gather (100 B). No working 30×30 plane other than the final label.

## OPEN ANGLES (re-attack backlog)
- Shave the 200 B col-0 slice: contract the 10 channels with a per-channel MatMul over a
  [1,10,5,1] weight instead of Slice→Conv (marginal, ~150 B).
- Build the 30×30 label by a single Gather of band-colours with a const 30×30 max-index
  plane (off-grid index → a sentinel slot) — folds Pad + Reshape into one Gather, may
  trim ~100 B but keeps the 900 B floor.
- 2026-06-29 sparse-initializer probe: replacing the dense 30×30 `canvas_index`
  with a sparse initializer and moving the sentinel to default index 0 is invalid.
  ONNX shape inference reports Gather indices as `sparse_tensor(int32)`, and ORT
  rejects the model because sparse initializers are not dense tensor operands.

## INSIGHT (transferable)
⭐ When the output colour at (r,c) depends ONLY on a scalar function of the coordinates
(here `max(r,c)`), the whole label collapses to a CONST-INDEX Gather of a tiny per-band
colour VECTOR — do the data-dependent `mod L` on the length-K band vector (10 elems), NOT
on the full 10×10/30×30 index plane, killing every full-size int/fp index plane. Recover
the band-colour palette from a single input line (column 0 = `colors[0..L-1]`), not a 2-D
region. Footprint guard: when the output grid is exactly K×K with no interior background,
Pad the label with a sentinel ≥10 (NOT 0) so off-footprint cells are all-channels-off, not
channel-0-on (a pad-0 silently lights background and fails every fresh instance).
