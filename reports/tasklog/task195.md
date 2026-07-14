# task195 — 80af3007 (fractal self-tiling of a 3x3 sprite)

**Rule:** A 3x3 binary sprite S (`common.conway_sprite`, 3..6 on-cells, always
covers every sprite row AND column) is upscaled 3x — each on-cell becomes a 3x3
gray(5) block — and dropped at a random (rowoffset,coloffset) on a ~16-19 wide
canvas. Only colour 5 ever appears. The OUTPUT is always a 9x9 grid equal to the
Kronecker product `kron(S,S)*5`:  `output[3i+r,3j+c]=5 iff S[i,j] AND S[r,c]`.
The output one-hot covers ONLY the top-left 9x9 footprint (verified: ch0 sum 56 +
ch5 sum 25 = 81 cells); cells outside the 9x9 are all-channels-off.
**Current:** prior 15.44 (public ext:kojimar6275). This session: **16.86 pts,
custom label-map (sprite recovery + kron + Equal), mem 3172, params 243.**
**Target tier:** B (label map + final Equal). Tier S/A blocked: the output cell
value is `S[u//3,v//3] AND S[u%3,v%3]` — needs the full data-dependent 3x3 sprite
recovered from a variable-offset upscaled copy (offset → a Gather, not a fixed
conv/permute; the 2-factor kron index map is not a single row⊗col separable
rectangle). B is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp16 occ Conv [1,1,30,30] + reduce profiles + 2x Gather S + Sa⊗Sb Mul + 30x30 L-pad Equal | B | 8557 | 266 | 0 | — | FAIL: Sa[u]·Sb[v] is wrong; kron is a 2-factor index map, not outer product of one flat vec |
| 2 | fix kron via 2 const index maps macro=(u//3)*3+(v//3), micro=(u%3)*3+(v%3) → Gather Sflat ×2 → Mul | B | 8557 | 266 | 15.91 | — | correct; fp32 occf(3600)+fp16 occ(1800) both counted = waste |
| 3 | drop fp16 cast, use fp32 occ Conv directly | B | 6799 | 266 | 16.14 | — | removed redundant 1800 |
| 4 | bool kron (And not Mul; Greater→bool Sflat, And) | B | 5998 | 266 | 16.26 | — | [9,9] planes bool not fp32 |
| 5 | replace Conv occ with Slice ch5 to small 18x18 canvas (sprite always within rows0..15 cols0..17) | B | 3334 | 244 | 16.82 | — | 3600→1296 plane: biggest win |
| 6 | tighten canvas to 16x18 (verified 30k fresh: last row<=15 col<=17) | B | **3172** | 243 | **16.86** | **400/400** | BEST |

## Best achieved
**16.86 @ mem 3172 params 243 — fresh 400/400 (isolated, temp-net).** Beats prior
15.44 by **+1.42**. Adopted? N (build-only per brief; main adopts via
`python -m src.adopt 195`).

## Irreducible-floor analysis
Two intermediates dominate, total ~2052 of 3172:
- **occ [1,1,16,18] fp32 = 1152** — the channel-5 occupancy slice. Slice preserves
  the input's fp32 dtype, and the sprite can sit anywhere within the 16x18 active
  region (offset>=1, canvas<=19 wide), so this plane can't shrink further without
  losing a reachable cell. Casting to fp16 would ADD 576 cumulatively (Slice still
  emits fp32 first) — net worse, not better.
- **L [1,1,30,30] uint8 = 900** — the Pad output that drives the final Equal. The
  Equal must span the full 30x30 output footprint, so the label map is 900 at uint8
  (already the smallest dtype). This is the canonical label-map floor.
Everything else is <=216 B (1-D profiles [1,1,16,1]/[1,1,1,18], the [1,1,3,18]
row-gather, 3x3 sprite, [9,9] bool/uint8 kron+label).

## OPEN ANGLES (re-attack backlog)
- **Drop the 900 L-pad** by doing Equal at 9x9 ([1,10,9,9] bool=810) then placing
  it at the top-left of the 30x30 output — but ORT **Pad rejects bool** (verified
  INVALID_GRAPH), and Concat/ScatterND assembly of 10 channels costs more than 900.
  A uint8 Pad path keeps the 900. No clean sub-900 final found.
- **Drop the 1152 occ** by deriving r0/c0 from two 1-D Convs (no 2D plane) and
  gathering S straight from `input` — but the area-gather of a single channel still
  forces one [1,1,~16,~18] (or [1,10,3,30]=3600) plane; the 1-channel slice IS the
  cheapest read. Net neutral at best.
- Tier-A long shot: kron is separable only via the 2-factor (macro,micro) index
  pair, not a single row⊗col outer product, so a clean Tier-A row⊗col And does not
  apply here. (This is exactly what broke attempt #1.)

## 2026-06-30 task001 threshold-product transfer probe

The live/source graph is now ahead of the older note above:
`18.174539963744692`, `mem=882`, `params=39`, method `ext:franksunp7166_65`.
It uses a tiny recovered 3x3 gray mask, a bool 9x9 kron, and a
`out6ch [1,6,9,9]` carrier before `Pad`.

Probed task001's direct factorized product in
`reports/scripts/task195_threshold_product_probe.py`.

- ch5-only direct product: invalid, because a fixed background constant cannot
  be formed from a binary occupancy sprite inside one `Einsum`.
- ch0/ch5 one-hot product: exact stored `265/265`, but **worse**:
  `17.921658420442327`, `mem=775`, `params=411`.

Conclusion: task195 is a semantic sibling of task001, but not a scoring sibling.
The current `out6ch` carrier is cheaper than paying dense coordinate/channel
selector params for direct-output factorization.  Revisit only if sparse
`Einsum` initializers become scoreable.

## INSIGHT (transferable)
⭐⭐ **ARC "fractal self-tiling" (a shape rendered with copies of itself) = the
Kronecker product `kron(S,S)`. kron is NOT the outer product of one flattened
vector — `kron(S,S)[u,v] = Sflat[(u//3)*3+(v//3)] · Sflat[(u%3)*3+(v%3)]`, a
2-factor product over TWO constant integer index maps. Build it as two Gathers of
the recovered Sflat by precomputed [9,9] macro/micro maps, then And/Mul.** The
naive Sa[u]·Sb[v] outer product is wrong and silently passes shape inference.
⭐ **Recover a variable-offset upscaled sprite offset-free via the bounding box:**
when the generator guarantees the sprite covers every row/col (conway_sprite does),
r0/c0 = first occupied row/col (ReduceMin of `present ? idx : BIG`), then Gather the
3 macro-rows (r0+[0,3,6]) and cols (c0+[0,3,6]) to read S — no offset parameter
needed, no full-resolution image.
⭐ **GEOMETRY BOUNDS BEAT DTYPE TRICKS again** (3600→1152 by slicing the colour
channel to the 16x18 active region; fp16 on the same plane would have ADDED bytes).
Compute the exact active extent from the generator (30k fresh: last row<=15,
col<=17) and Slice straight to a small fp32 canvas instead of a full 30x30 Conv.
