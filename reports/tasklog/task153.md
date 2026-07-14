# task153 — 681b3aeb

**Rule:** A 10x10 grid holds TWO 3x3 windows at random non-overlapping positions
(|row gap|>2, |col gap|>2). Window `idx` is filled with colour `colors[idx]`
exactly at the cells where a 3x3 partition mask `idxs` equals `idx`. Group 1 is a
`continuous_creature` grown from (0,0) (4-connected, ALWAYS contains its window's
top-left corner, so its bbox-min == its window anchor); group 0 is the complement
(forced diagonally connected). Both colours come from `random_colors(2)` (distinct,
nonzero). The 3x3 OUTPUT is the superposition of the two windows: out[r][c] = colour
of whichever group covers (r,c) (the groups partition all 9 cells, so every output
cell is filled exactly once). Reconstruction (verified): place the CREATURE colour
with its bbox top-left at output (0,0) — it lands exactly on its cells — and fill the
rest with the OTHER colour. Identify the creature by congruence: for candidate colour
C, the complement region (3x3 minus C's bbox-TL pattern), re-normalised to its own
bbox-TL, equals the other colour's bbox-TL pattern iff C is the creature (when both
satisfy it, both placements give the identical output).

**Current:** 15.07 pts (public CumSum/TopK/OneHot net, 72 nodes), mem ~20.5k, params ~?
**Target tier:** A — separable per-colour bbox window extraction + closed-form
creature discriminator; output routed into the FREE bool one-hot. No flood-fill,
no global argmax over variable components.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox-TL window per colour + congruence select, fp32, int64 Pad | A | 20466 | 56 | 15.07 | 265/265 | correct but only ties P |
| 2 | 1x1 Conv colour-index + slice 10x10; uint8 sentinel Pad | A | 9703 | 56 | 15.81 | 265/265 | +0.74 |
| 3 | window/pad planes in fp16 | A | 8355 | 57 | 15.96 | 200/200 genverify, 5000/5000 inline | +0.89 ADOPT |

## Best achieved
15.96 @ mem 8355 params 57 — adopted? recommend **Y**. Beats prior 15.07 by **+0.89** (>=+0.3).

## Irreducible-floor analysis
Dominant intermediate = `colf30` [1,1,30,30] fp32 = 3600B: the 10->1 colour-index
collapse via a 1x1 Conv MUST output fp32 (the documented 3600B entry floor; ORT
ReduceSum/Conv won't emit a narrower colour-index plane). Next is the uint8 sentinel
label [1,1,30,30] = 900B (Pad rejects bool, uint8 is the floor for the Equal-onehot
carrier). Everything else lives on the tiny 10x10 / 13x13 / 3x3 active canvas in fp16
(<=338B each). Conv-on-full (3600+400 slice) beats slice-input-first (4000B for the
[1,10,10,10] slice + 400 conv). Hard to push below ~15.96 without removing the colf
entry entirely.

## OPEN ANGLES (re-attack backlog)
- Avoid the 3600B `colf30` entry: the two colours could be read as two channel-Slices
  (each [1,1,30,30] fp32 = 3600B too, no win) — but a Conv that outputs directly onto
  a 10x10 crop is blocked because Conv reads the full 30x30 input. A Gather-based 10x10
  crop of the INPUT first costs 4000B. No clear sub-3600 path found.
- Replace the congruence discriminator (compA renorm + Equal) with a cheaper scalar:
  tried (tl/br corner occupancy, 4-conn, pixel count) — none are exact; the shapes of
  creature vs complement are too symmetric, so the position-aware congruence test is
  required. Small cost anyway (3x3/6x6 planes).

## INSIGHT (transferable)
⭐ When an output is the OVERLAY of two data-dependently-placed sub-patterns that
PARTITION a fixed small grid, you don't need both placements: anchor ONE pattern
(the one with a recoverable anchor — here the creature, whose 4-connected growth from
(0,0) makes bbox-min == window anchor, 100%) and fill the rest with the other colour.
⭐ "Which of two complementary shapes is the canonical one" can be a closed-form
CONGRUENCE test: complement-of-A renormalised to its bbox-TL equals B's bbox-TL
pattern iff A is canonical — exact even when scalar shape features (count, corner
occupancy, connectivity) are symmetric and fail.
⭐ Edge-of-canvas bbox-TL window extraction: zero-pad the mask by (k-1) on the far
side BEFORE the Gather(arange+first_idx) so a bbox-min near the border still gathers k
rows/cols without harmful index-clip duplication. fp16 ArgMax/ReduceMax/Pad/Gather all
work under ORT_DISABLE_ALL (only fp16 Min/Max crash).

## S17 (2026-07-06) — dtype-overpay recast (bit-identical safe golf, +dtype_overpay_scan)
task153 presence_nonzero {0,1} → uint8 via Cast(pool_max_all→uint8) before Slice; ArgMax order-preserving. 795→778 (−17).
Gate: evaluate bundled fail=0 + **bit-identical outputs** over all train/test/arc-gen (verified). Safe for both tracks + private LB.
⭐ TRANSFERABLE: only ACTIVATION (node-output) dtype narrowing saves grader bytes — params counted by element-count (dtype-independent). Narrow the PRODUCER (upstream Cast/init dtype), never a post-Cast. Blocked when the plane is derived from / contracted with the free fp32 `input` (Einsum-vs-input, Slice/Conv of input, ScatterND updates vs fp32 data) → those force fp32. See [[neurogolf-fp16-count-plane-recast]].
