# task351 — dc0a314f

**Rule:** A 16x16 grid is built from an 8x8 quadrant mirrored both H and V, so it is fully D2-symmetric: value(r,c)==value(15-r,c)==value(r,15-c)==value(15-r,15-c). A 5x5 cutout at (row,col) is erased to GREEN(3) (green appears nowhere else). The 5x5 output is the original cutout content, reconstructed from the intact double mirror: output[i][j]=grid[15-row-i][15-col-j]. Pure SPATIAL COPY. random generate() clamps row,col∈{0..3}; stored stress examples reach row=5,col=9 (source spans rows 6..15, cols 2..15 — NOT a fixed corner quadrant).

**Current (prior):** 15.81 pts, two-Gather one-hot copy + Pad→output, mem 9752, params 16
**Target tier:** S (pure spatial copy of input cells)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 8x8-corner slice + uint8 copy | S | 4282 | 30 | 0.0 | — | FAIL: stored row=5,col=9 breaks corner assumption (262/265) |
| 2 | full-input 2-Gather one-hot, uint8 Vs, green16 recovery | S | 8514 | 24 | 15.95 | — | ok but Vr [1,10,5,30] fp32 6000B dominates |
| 3 | + green_sz=12, fp32 Vs (no cast) | S | 7784 | 24 | 16.04 | — | best 10-ch copy; still 6000B Vr |
| 4 | source-region pre-Slice [6:16,2:16] | S | 10184 | — | 15.77 | — | WORSE: slice plane ADDS to gather plane |
| 5 | opset-12 GatherND batch_dims=2 | S | 8464 | 71 | 15.95 | — | WORSE: [1,10,25,2] int64 index = 4000B |
| 6 | **colour-index plane + 1-ch gather + one-hot-on-5x5-block→FREE output** | S | 5392 | 44 | 16.40 | — | WIN |
| 7 | + Equal on fp32 (drop int32 cast), green[7,11] | S | **5292** | 44 | **16.42** | 200/200 | **ADOPTED** |

## Best achieved
16.4178 @ mem 5292 params 44 — adopted? Y. Beats prior 15.81 by +0.605 (≥+0.3 ✓). Fresh 200/200 isolated; stored 265/265; opset-11 full_check OK.

## Irreducible-floor analysis
Dominant intermediate: **colf [1,1,30,30] fp32 = 3600B** — the single colour-index entry plane (Σk·input_k via 1×1 Conv). Conv output is fp32 (input is fp32; casting input to fp16 = 18000B), and the gather must read from a full-canvas plane, so 3600B is the entry floor (the standard "pay one 3600B fp32 entry" pattern). Everything downstream is tiny: Vr [1,1,5,30]=600B, Vs [1,1,5,5]=100B, oh/oh8 [1,10,5,5]=250B each, green recovery [1,1,7,11]≈300B. The 10-channel expansion is routed into the FREE output (Equal→bool→uint8→Pad), so NO [1,10,*] full plane ever materialises.

## OPEN ANGLES (re-attack backlog)
- Shrink the 3600B colf entry: would need the colour-index on only the 16x16 active grid, but slicing input first costs more (10240B). No cheap route found.
- oh (bool) + oh8 (uint8) are two 250B planes because Pad rejects bool; if a future ORT/opset lets Pad take bool, drop the Cast (−250B → ~16.45).
- green recovery [7,11] has small margin (first-col max 9 → cols 0:10 needed); [6,10] is exact (mem 5316→ wait, smaller, 16.41) but zero-margin. Kept [7,11] for safety.

## INSIGHT (transferable)
⭐ A "pure spatial COPY of a 10-channel one-hot" is NOT best done by gathering the 10-channel input (forces a [1,10,W,H] fp32 plane = 10×cost). Collapse to a SINGLE colour-index plane first (1×1 Conv Σk·input_k, 3600B once), do the data-dependent Gathers on the 1-CHANNEL index (6-10× cheaper), then expand back to one-hot ONLY on the small gathered block via Equal(block, arange_ch[1,10,1,1])→bool and Pad that tiny uint8 block into the FREE output. This is the count→one-hot-into-free-output lever applied to a COPY task. Also: Equal accepts fp32 operands under ORT_DISABLE_ALL (integer-exact), so the int32 cast before Equal is unnecessary. And: a "pure corner-quadrant copy" assumption can be FALSE even when the random generator clamps coordinates small — the STORED stress examples (evaluate) use much larger hand-coded offsets, so always check the stored examples' coordinate range, not just the generator's randint bounds.

## S8 (2026-07-02) — rect-recipe conversion ADOPTED, div 0
free-input einsum marker locate (25·(15−row) scalar), green12 plane dropped; 1632→1015, +0.487. Fresh: agent uncached 2500 div0 + my uncached 400 div0.

## S12 (2026-07-03) — current floor rechecked
Current source/live is the S8 moment-locate model, scoring 18.077 at mem 956 /
params 59.  Dominant cost is now `answer_patch[1,9,5,5]` fp32 = 900B from the
reversed `Slice` of the free fp32 input.  Casting after the slice would add a
second 225B bool/uint8 patch while keeping the 900B entry tensor, so it is
worse.  Slicing/casting the input before coordinate recovery would materialize a
large input plane.  No new candidate built; this is a direct input-copy floor
unless a primitive can sample fp32 input into a smaller dtype.
