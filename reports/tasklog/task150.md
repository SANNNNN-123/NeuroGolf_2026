# task150 — 67a3c6ac

**Rule:** A square grid of side `size` (3..9), every cell coloured from {6,2,1,7}
(never colour 0), sits at the top-left corner of the 30×30 canvas; off-grid cells
are ALL-ZERO. The output is each row reversed (horizontal mirror):
`out[:,:,:,c] = in[:,:,:,size-1-c]` for c<size, all-zero for c>=size. Pure spatial
permutation along the column axis.
**Current:** 18.44 pts, public CumSum/Where index + Gather (mem 668, params 36)
**Target tier:** S — output is a pure column permutation of the input → ONE Gather
whose output IS the free graph output; only the int32 index vector materialises.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum col-profile → size → Where(in-grid, rev, 29) idx fp32 → Gather | S | 788 | 37 | 18.28 | 200/200 | below target |
| 2 | same, fp16 working vectors, reuse occ as in-grid mask | S | 454 | 36 | 18.81 | 200/200 | beats +0.37 |
| 3 | drop Where: negative idx wraps to off-grid zero cols (no clamp) | S | 394 | 35 | 18.94 | 200/200 | +0.50 |
| 4 | size = Sqrt(ReduceSum(input)) [=size²] → scalar, kills [30] col-profile | S | 192 | 31 | 19.59 | 200/200 | +1.15 |
| 5 | integer index arithmetic (int32 arange, no fp16 rev) | S | 136 | 31 | 19.88 | 500/500 | ADOPTED |

## Best achieved
19.88 @ mem 136 params 31 — adopted? Y. Beats prior 18.44? Y (+1.44).

## Irreducible-floor analysis
Dominant intermediate is the int32 column-index vector `idx` ([30] = 120B). Gather
indices reject uint8 (ORT) and int64 is wider, so int32×30 = 120B is the floor for
ANY Gather-based column permutation. The remaining ~16B are three scalars
(total/size/size-1). No full-grid (30×30) plane ever materialises — the result plane
is the FREE output, and `size` is recovered from a single scalar reduction.

## S3 re-attack (2026-06-30) — Range param-trick TESTED & HELD (grader-fragile)
Live incumbent is mem 128 / params 30 / **19.937 pts** (ReduceL2→Cast→Sub(arange[1..30])
→Gather; arange is a 30-elem int32 initializer = 30 params). The 30-param arange is the
only remaining cost beyond the 120B index + 8B scalars.

Angle: build the mirror indices with `Range(start=G-1, limit=G-31, delta=-1)` instead of
`Sub(G, arange)`. This is bit-identical (`[G-1..G-30]`) and drops the arange initializer:
**params 30→3, mem 128→136, total 158→139, 20.066 pts (+0.128)**. Verified 266/266 bundled
+ 1600/1600 random in-domain, 0 divergence — UNDER the grader's `ORT_DISABLE_ALL`.

REJECTED FOR LANDING: under `ORT_ENABLE_ALL` the optimizer collapses the dynamic Range to a
**0-width output on 1600/1600 inputs** (graph breaks entirely). If the Kaggle grader applies
any graph optimization, task150 scores ~0 (−15). Asymmetry +0.13 vs −15 is unacceptable; same
class as the uint8-TopK grader-killer (local≠grader). Held at `reports/candidates/task150_range_flip.py`
for an optional one-at-a-time Kaggle A/B probe (zip = submission.zip). **Incumbent KEPT.**
Conclusion stands: the incumbent (158) is optimal among ROBUST (optimization-safe) formulations;
the +0.90 min_stat headroom is the confound-2 variable-size-transform mirage.

## OPEN ANGLES (re-attack backlog)
- A negative-step Slice (steps=[-1], axis=3) reverses the WHOLE 30-col axis with 0
  params/0 mem, but lands the grid at the right edge (shifted by 30-size); undoing
  that shift is itself data-dependent and needs a Gather/Slice, so no net win.
- Strict floor for permutation-by-Gather is the 120B int32 index; only a 0-index
  closed-form reversal (none exists for a left-anchored variable-size block) beats it.

## INSIGHT (transferable)
⭐ Two reusable levers landed here:
1. SIZE FROM TOTAL-COUNT, NOT A PROFILE: for a fully-filled k×k grid of nonzero
   colours, `size = Sqrt(ReduceSum(input))` is a single 4B scalar (fp32 sqrt of a
   perfect square 9..81 is exact, truncating Cast→int32 is safe) — avoids the [30]
   column-occupancy profile entirely. Use whenever a scalar dimension equals √(pixel
   count) or (pixel count)/known-width.
2. NEGATIVE-INDEX WRAP AS A FREE OFF-GRID CLAMP: a reversed index `size-1-c` goes
   negative for c≥size; ONNX Gather wraps `idx+dim`, which for this left-anchored
   block lands every out-of-grid column on columns [size..29] — all off-grid/zero —
   so the output zero-fills with NO Where/clamp/fallback constant. Replaces a
   Where(mask, rev, fallback) (drops a bool mask + a const + an op).

## S9 (2026-07-03) — Range swap (task155 sibling) (+0.135) ADOPTED
30-elem arange init → Add/Sub scalars + Range(size-1, size-31, -1). mem 128→136,
params 30→2, total 158→138. Gates: stored fail=0; uncached 2000 fresh 0/0/0;
edge sizes G=1..30 (600, DISABLE_ALL) 0 mismatch. Known trade: ENABLE_ALL optimizer
collapses dynamic Range to 0-width — same profile as adopted task155, whose teacher
shipped in kojimar's LIVE 7184.85 LB submission (real-grader-proven). S3 rejection was
ENABLE_ALL-specific. Backup task150_pre_s9.onnx.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 9. Flagged `mirror_indices` int32 [30] = 120B is already a floor Gather index; its length must equal the 30-wide free output, so it can't shrink to 9. Cropping forces a counted [1,10,30,9] 10800B re-embed. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
