# task058 — 28e73c20

**Rule:** INPUT is an all-background square grid of side `size`∈[5,20]; there is
NOTHING spatial to detect. OUTPUT is a deterministic green (colour 3) inward
rectangular involute spiral that is a pure function of `size`: starts at (0,0)
going right, lays the top + right edges, then winds inward with stride 2.
Closed form (exact, all sizes 5..20): with r,c index, e=N-1-r, f=N-1-c,
ring layer L=min(r,c,e,f) — `green = (L even) XOR (r==c+1 AND c==L)` OR the
even-N termination cell `(2r==N AND 2c==N-2)` gated by `N%4==2`, then AND in-grid.
The `r==c+1 AND c==L` term is the single per-ring break/connector (gap on even
rings, connector on odd rings), one cell below the diagonal on the left edge.
N is recovered as `ReduceMax(ReduceSum(input_ch0, axis=3))`.
**Current (public):** 15.31 pts
**Target tier:** B — output is a genuinely 2-D, size-dependent ring pattern; the
ring layer `L=min(r,c,e,f)` couples r&c non-separably, forcing at least one full
working plane; no tier-A separable route exists.

## Attempts (this session — BROKE the prior 15.10 "below baseline" verdict)
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | closed form @30×30 fp32 | B | 37341 | 83 | 14.47 | — | correct, below baseline |
| 2 | crop to 20×20 working canvas (N≤20) | B | 20301 | 78 | 15.08 | — | below baseline |
| 3 | + fp16 label/Pad/Equal pipeline | B | 16901 | 78 | 15.26 | — | below baseline |
| 4 | + **fp16 geometry (ramps/Min/Mod/Equal)** | B | 12815 | 78 | 15.54 | — | +0.22 |
| 5 | + variadic Min (one plane vs three) | B | 11215 | 78 | 15.67 | — | +0.36 |
| 6 | + separable special (r==c+1, drop L+1 plane) | B | 10455 | 78 | 15.74 | — | +0.42 |
| 7 | + 1-D-gated center-fix Ands | B | 10075 | 78 | 15.77 | 200/200 | **ADOPT** |

## Best achieved
15.77 @ mem 10075 params 78 — adopted? Y (src/custom/task058.py only; manifest /
networks / pipeline UNTOUCHED). Beats prior public 15.31? **Y, by +0.46.**
Isolated fresh: 200/200 (separate process, generator loaded by file path) AND
per-size 96/96 across all sizes 5..20.

## Irreducible-floor analysis
Dominant intermediate: the 30×30 fp16 padded label plane `lab` (900 elems × 2B =
1800B) that carries the colour index into the FREE bool output via
`Equal(lab, arange)`. All other working planes live on the 20×20 crop (400 elems:
fp16 = 800B, bool = 400B). One full-canvas index plane is the structural minimum
for a per-cell colour map. Not at a hard floor — see open angles.

## OPEN ANGLES (re-attack backlog)
- Route ch0 (in-grid non-green) and ch3 (green) bool planes directly into the
  output, dropping the shared 30×30 carrier; Pad rejects bool so use a uint8
  carrier (1B vs 2B fp16, ~−900B → ~+0.1).
- 20×20 crop is the true minimum (gen size hits 20); no tighter crop.

## S12 (2026-07-03) — sparse-initializer bitpack probe REJECTED
Current live/source is no longer the old fp16 closed-form pipeline; it is a
5-node uint64 bitpack model:
`ReduceL2 -> Cast -> Sub -> Gather(row_bits_table) -> BitwiseAnd(column_bits)`,
scoring 18.060 at mem 252 / params 781.  The remaining cost is mostly dense
initializers: `row_bits_table[16,1,30,1]` has 200 nonzero / 480 dense elems and
`column_bits[1,10,1,30]` has 60 nonzero / 300 dense elems.

Tried sparse initializers for both tables.  ORT can execute after fixed-point
sanitize naming, but scorer shape inference rejects sparse `Gather`:
`data tensor must have rank >= 1`.  Tried sparse `column_bits` only; scorer
rejects sparse `BitwiseAnd` input: `B typestr: T, has unsupported type:
sparse_tensor(uint64)`.  Therefore sparse initializer compression is not
adoptable for this bitpack family under the current harness.  Generating
`column_bits` dynamically would materialize a full channel/column mask
intermediate and is more expensive than the 300 dense params it saves.

## INSIGHT (transferable)
⭐ **The prior verdict here was wrong because of TWO stale dtype claims** — a prior
agent recorded "fp16 Min crashes under ORT_DISABLE_ALL" and "ORT upcasts Where to
fp32 via PrecisionFreeCast", concluded the floor was ~15.1 (below baseline), and
bailed. BOTH are FALSE on this ORT build: fp16 `Min`/`Where`/`Mod`/`Equal`/`Pad`
all run AND the fp16 planes count at HALF in the static trace. Re-running the whole
geometry in fp16 dropped bloat 37k→10k (14.47→15.77). **Always empirically re-test
a recorded "fp16 X crashes / upcasts" claim with a 5-line ORT probe before paying
fp32 — these crash claims drift across ORT versions and silently cost ~0.5+ pts.**
⭐ A "draw a deterministic shape whose ONLY free parameter is grid size N" task is
COUNT→FIXED-PATTERN: recover N via `ReduceMax(ReduceSum(input_ch0,[3]))`, then the
output is a closed-form per-cell predicate in (r,c,N). The ARC inward square spiral
closed form: `green = (min-ring-layer parity) XOR (per-ring connector at (L+1,L))`,
with a single `N%4==2` center patch cell. Crop the working canvas to the gen max
size (20) before the per-cell algebra — 2.25× cheaper than 30×30 — then Pad the
small label up. Variadic `Min(a,b,c,d)` is one plane vs three chained `Min`s; And
is NOT variadic (max 2 inputs), Min/Max/Sum are.
