# task240 — 9d9215db (4-fold mirror a diagonal bitmap + dotted-square rings)

**Rule:** 19×19 grid. Input holds a small (length 2-4) diagonal of colored pixels
at odd coords (2r+1,2c+1), optionally a "nextdoor" color one step along the
diagonal, and the grid may be h/v-flipped. Output is fully determined per
chebyshev ring d∈{1,3,5,7} (distance from border) by two colors: Cd (the 4 ring
corners, where ymin==xmin==d) and Kd (all other odd cells of ring d = dotted
square; 0 keeps them bg). Because flips+mirroring are 4-fold symmetric, Cd/Kd
are read off canonical positions corner d=(d,d), ring d=(d,d+2) by folding (max
over the 4 mirror cells). Verified exact on all 266 stored + fresh 500/500.
**Current (prior):** 16.06 pts, custom:task240, mem 6670, params 957
**Target tier:** B (per-cell deterministic label-map → Equal→bool output). The
group assignment depends on min(ymin,xmin) AND ymin==xmin, genuinely 2-D coupled
(not row⊗col separable), so a label-carrier is required — but the carrier and
the colour-read plane can both be shrunk far below the 30×30 floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| prior | full 30×30 fp32 Conv plane + 900-int groupid gather | B | 6670 | 957 | 16.06 | — | baseline |
| 1 | STRIDE-2 Conv (1×1 ramp, pads tl=1) → 16×16 fp32 colour plane (only odd coords); 11×11 group plane; separable row/col carrier; bg ch0 bug (off-grid = ch0 instead of all-zero) | B | 4442 | 347 | 0.0 | — | FAIL (pad region) |
| 2 | + sentinel colour 10 for off-grid (sentinel row/col idx 10 in rowmap; gcvec[10]=10) | B | 4162 | 213 | 16.62 | — | exact |
| 3 | + gather carrier directly on spatial axes 2,3 (subcol [1,1,11,11]) → no [1,1,30,30] reshape-duplicate plane | B | **3262** | **209** | **16.85** | 200/200, 500/500 | **FINAL** |

## Best achieved
**16.848 pts @ mem 3262, params 209 — 266/266 stored, fresh 500/500.** Adopted? N
(orchestrator gates). Beats prior 16.06? **Y, +0.79.** GENERALIZES.

## Irreducible-floor analysis
Dominant intermediates: (1) the strided colour plane `cidxf` [1,1,16,16] fp32 =
1024B — Conv must output float and 16×16 is the smallest dense grid a stride-2
1×1 Conv yields from a 30-wide input (can't shrink via negative pad; slicing the
10-ch input first is far costlier). (2) the carrier `outidx` [1,1,30,30] u8 =
900B — the Equal carrier must span the full 30×30 output (Equal output itself is
FREE as a declared BOOL output). Both are at floor. The rest (byrow 330, two u8
flatten steps 256+256, tiny gather/cast tensors) is small.

## OPEN ANGLES (re-attack backlog)
- byrow [1,1,30,11] (330B) is the row-gather intermediate of the separable
  carrier; gathering cols-first is symmetric (no gain). Folding both 1-D gathers
  into one would need a 900-int 2-D index map (back to the prior params cost).
- The two u8 flatten tensors (cidx8 256 + cflat 256) get a flat u8 to gather the
  32 mirror cells; gathering from fp32 directly needs a 1024B fp32 reshape
  (worse). ~512B likely irreducible for the cheap-flatten path.

## INSIGHT (transferable)
⭐ **STRIDE-2 Conv to read only odd-coord cells.** When a colour-index plane is
only ever sampled at odd (or any fixed-stride) coordinates, a stride-2 1×1
`Conv(input, ramp)` with `pads=[1,1,0,0]` collapses the 10-ch read into a
16×16 fp32 plane (1024B) — `out[i,j]=input[2i-1,2j-1]` — instead of the full
30×30 (3600B), a 2576B saving with NO extra ops. ⭐ **Separable carrier via two
spatial-axis Gathers.** A 4-fold-symmetric / coarse-resolution per-cell label
map can be rebuilt as `Gather(Gather(subcol[1,1,K,K], rowmap[30], axis=2),
colmap[30], axis=3)` straight to [1,1,30,30] — replaces a 900-int 30×30 index
init with two 30-int maps (60 params) AND, by gathering on axes 2/3 directly,
the result is already [1,1,30,30] so NO reshape-duplicate plane is materialised.
Off-grid cells get an out-of-band sentinel index (→ sentinel colour 10) so the
final `Equal(carrier, ramp)` leaves them all-zero (the 30×30 pad region must be
all-zero, NOT ch0=1 — convert_to_numpy only one-hots the true HxW region).
