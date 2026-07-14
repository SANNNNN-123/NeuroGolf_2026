# task077 — 36fdfd69

**Rule:** The grid carries random `static`-colour noise pixels on a `0` background, plus
2–4 non-overlapping solid rectangles (height t∈{2,3}, width w∈{2..7}). Each rectangle is laid
over the noise: a cell that was empty(0) becomes `red(2)`, a cell that was `static` becomes
`yellow(4)` (legal-check guarantees every row & col of the rect has ≥1 red). The INPUT shows
that yellow re-cast back to `static` (yellow→static), so the input rectangle is a solid block of
`{red, static}`. The OUTPUT is the input with every **rectangle** `static` cell recoloured to
`yellow(4)`; noise `static` cells (and reds, background) are unchanged.
**Current:** 13.5 pts, public-net per-cell colour Conv (essentially reproduces input; cannot recolor).
**Target tier:** none admissible — see analysis. (entry hypothesis: Tier-S fixed recolor — REJECTED.)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | pure per-color recolor Where(input==static,yellow,input) | S | — | — | — | — | REJECTED: static maps to BOTH static (noise) and yellow (rect) in 50/50 instances → position-dependent, not a remap |
| 2 | bbox of 8-conn red components, recolor static inside | det | — | — | — | 395/500 bad | FAIL: separate rects' reds merge / a rect's reds are not connected |
| 3 | 8-conn nonzero component containing a red, recolor static | det | — | — | — | 500/500 bad | FAIL: noise static touches rect edges → component leaks across whole grid |
| 4 | row-run AND col-run through cell both contain red | det | — | — | — | 66/300 bad | FAIL (22%): noise static flush against a rect's reds gets false-positived |
| 5 | exists solid (all-nonzero) 2..3×2..7 window containing cell + a red | det | — | — | — | 150/150 bad | FAIL: accidental solid windows with nearby red |
| 6 | 2×2 solid block containing a red, recolor static in it | det | — | — | — | 296/300 bad | FAIL: pulls in red's own noise neighbours |
| 7 | interior (3×3 all-nonzero) static | det | — | — | — | 200/200 bad | FAIL: rects are 2–3 tall → 73% of flips are on the boundary, no interior |
| 8 | brute-force maximal *legal* solid rectangles (every row&col has red), recolor static | det | — | — | — | 115/150 bad | FAIL even in unrestricted Python: reconstruction is underdetermined |

## Best achieved
None. No encoding reaches exactness; identity (output=input) scores 0/200 (every instance has ≥1 flip,
avg 12.3 flips). Floor stays 13.5. No file written / not adopted.

## Irreducible-floor analysis
This is a **2D object-segmentation inverse problem**, not a closed-form transform. The recolor target
(rect-static → yellow) requires identifying the exact extent of each solid rectangle and confirming it is
a red-containing rectangle. The decisive obstacle: **noise `static` pixels sit flush against rectangle
edges** (the ≥2-cell spacing constraint applies only rect-to-rect, never rect-to-noise). This defeats every
separable / local / connected-component detector:
 - connectivity (red-comp bbox, nonzero-comp+red) leaks or splits because noise bridges rects and reds are
   not internally connected;
 - row/col runs and solid-window tests false-positive on noise adjacent to a rect's reds;
 - there is no rect interior to erode toward (height 2–3).
Crucially, attempt #8 shows that even a full Python brute force over maximal *legal* solid rectangles
fails 77% — the segmentation is genuinely ambiguous from the input alone, so it is impossible as a
fixed ONNX graph (which additionally bans Loop/Scan/NonZero/flood, ruling out any iterative region grow).

## OPEN ANGLES (re-attack backlog)
- None with a credible path to +0.3. The only thing that would help is iterative region-growing /
  labelling, which is banned (Loop/Scan/NonZero). If the op ban were ever lifted, a flood-fill from reds
  constrained to solid blocks might approach it — but even that is undercut by attempt #8's ambiguity.

## INSIGHT (transferable)
⭐ "Recolor" mislabels can hide true 2D segmentation. The fast disqualifier: check whether the source
colour maps to **multiple** targets *within a single grid* by position (here `static`→{static, yellow}
in 50/50). If yes, no fixed per-color remap / channel MatMul / per-channel Where can work, and you are in
detection territory. Then the deciding question is whether the discriminating object can be cut from
background noise by a *separable* (row⊗col / connected-component / local-window) rule — if noise abuts the
object (no guaranteed margin), all of those leak, and if even an unrestricted brute-force reconstruction is
ambiguous (>50% mismatch), the task is INFEASIBLE for a static ONNX graph regardless of memory budget.

# task077 (appended S8 2026-07-02) — WALK-EINSUM WIN: 14874 → 6331 (+0.854) ADOPTED
EXACT rule derived (old "underdetermined/heuristic" framing retired): every rect = bbox of its
red cells clustered by Chebyshev-≤2 connectivity (cross-rect red distance ≥3 ⇒ never bridges);
fill = static-coloured cell inside some cluster bbox. ONE 59-operand einsum: red plane read
straight from free `input` via channel selector e2[q] each step (NO counted red/t plane);
12-step penta-band walk with 4 checkpoint constraints (row(p0)≤r, row(p4)≥r, col(p8)≤c,
col(p12)≥c — one shared triangular T, subscripts swapped for ≥); stat[q] folds the static-at-
(r,c) factor in. Counted: walk 3600 + fill bool 900 = 4500 mem, 1831 params.
TRAP LOGGED: the 2 bundled original-ARC train examples VIOLATE the generator's row/col
red-visibility guarantee — a row/col-witness variant passed 20000/20000 fresh but failed
bundled. Checkpoint-bbox form covers both. Gates: stored 266/266; fresh 2500+1500 fail 0 div 0.

## S9 (2026-07-03) — native-crop lever: REJECTED, floor confirmed (measured)
Binding max 20×21 verified (gen + all 3 splits). Crop cand built & evaluated: 9167 vs
incumbent 6331 (+2836, fails mem gate). WHY CROP BACKFIRES on free-input walk einsums:
(1) fixed [1,10,30,30] output forces a 900B Pad of cropped fill; (2) static-at-(r,c)
read was FREE inside the einsum, cropped needs a counted 1680B+1100p conv plane;
(3) shared 30×30 T triangular splits into rectangular pairs (900→2460p). Best theoretical
config 10261 > 6331. ⭐ RULE: single-tap crop only wins when the 30×30 plane is a
COUNTED entry read; nets whose einsums read the free input in-op get WORSE. DO NOT re-probe.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k5(cost 6331): val gate fail. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
