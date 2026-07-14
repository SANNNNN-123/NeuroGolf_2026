# task326 — d10ecb37

**Rule:** Input is a width×height grid (width,height ∈ {4,6,8,10,12}, both multiples of `size`=4 or 6) of arbitrary colours 0-9. Output is the 2×2 grid = the TOP-LEFT 2×2 corner of the input: `output[r][c] = input[r][c]` for r,c ∈ {0,1}. Pure spatial COPY of the 4 corner cells (Tier-S shape, but the one-hot carries all 10 channels).
**Current:** 19.92 pts, Slice([1,10,2,2]) + Pad → 30×30, mem 160, params 0 (inline attrs).
**Target tier:** S (copy) — but the 4 corner cells × 10 colour channels force a 2×2×10 fp32 intermediate.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | current Slice+Pad fp32 | S | 160 | 0 | 19.92 | 200/200 | AT FLOOR |
| 2 | Slice→Cast(uint8)→Pad | S | 200 | 15 | 19.63 | — | WORSE (fp32 slice 160 + uint8 40 both summed) |
| 3 | Cast(uint8)→Slice→Pad | S | 9040 | 15 | — | — | WORSE (full-input uint8 cast = 9000B) |

## Best achieved
19.92 @ mem 160 params 0 — current public net is optimal. Beats prior? N (cannot).

## Irreducible-floor analysis
Need mem+params ≤ 119 to beat 19.92 by +0.3. The single intermediate is the cropped
corner `[1,10,2,2]` = 10·2·2·4 = **160B fp32**. It cannot shrink:
- Channels can't be dropped: corner colours are arbitrary 0-9, all 10 one-hot channels needed.
- Spatial is already minimal (2×2 = the exact output footprint).
- itemsize (uint8) is the only remaining lever, but NO ONNX op crops AND narrows dtype in
  one node. Slice/Gather/Pad preserve dtype (fp32); Cast narrows but can't crop. So uint8
  requires EITHER a preceding fp32 slice (160 fp32 + 40 uint8 = 200B, summed → WORSE,
  empirically measured) OR a full-input Cast (9040B → much worse, measured).
160B is the structural floor; this matches task152's finding that the fp32 entry slice is
unavoidable and is summed alongside any uint8 working planes.

## OPEN ANGLES
- None viable. Collapsing channels (Σk·input_k Conv) to a [1,1,2,2]=16B plane still needs
  the fp32 corner first (Conv over full input = 3600B, or Slice = 160B), so it cannot get
  below 160. The free output / Equal-broadcast lever doesn't help because the bottleneck is
  the ENTRY crop, not the output expansion.

## INSIGHT (transferable)
⭐ "Copy a tiny K×K×10 corner" tasks are at the fp32-entry-slice floor (K²·10·4 B) whenever
NO transform follows — the uint8 lever only pays off when a transform (gather/mirror/tile)
ADDS uint8 working planes that would otherwise be fp32; with a bare crop+pad the Slice→Cast
adds a plane instead of replacing one (160→200). For 2×2×10 the floor is 160B ≈ 19.92.
