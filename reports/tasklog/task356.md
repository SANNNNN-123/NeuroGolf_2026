# task356 — ded97339

**Rule:** Fixed 10×10 grid (size=10), placed top-left of the 30×30 canvas (rest background 0).
Random cyan(8) pixels scattered. In the OUTPUT, for every pair of cyan pixels sharing a ROW the cells
between them fill cyan; same for pairs sharing a COLUMN. Per row the closed span [min cyan col, max
cyan col] becomes cyan; same per column; single-pixel rows/cols fill only that one cell. A cell is
cyan iff it lies in some row-span OR some col-span. Both endpoints and the fill are the SAME colour
(cyan), so the span mask IS the output cyan mask — no "not-endpoint" gate needed. (Labeled
confirmed-infeasible BLANK note → FALSE-POSITIVE; closed-form, like its twin task350.)

**Current:** 16.48 pts, ext:kojimar6275, mem ≈5001
**Target tier:** A — closed-form per-row/per-col span fill via 4-directional prefix/suffix-OR
(non-separable, no flood-fill); 10-ch expansion routed into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4 fp16 MaxPool prefix/suffix-OR on 10×10 cyan slice → Mul → Greater → Where(cyan_oh,input) f32 out | A | 4000 | 29 | 16.70 | — | works, MARGINAL (+0.22) |
| 2 | replace fp32 Where path with uint8 colour-index plane L (0/8), Pad to 30×30, Equal(L,arange)→BOOL FREE output | A | 3100 | 31 | 16.95 | 200/200 | BEST, +0.47 |

## Best achieved
16.951 @ mem 3100 params 31 — adopted? N (per instructions). Beats prior 16.48? **Y by +0.47**.
Fresh ISOLATED 200/200 (subprocess, generator loaded by file path).

## Irreducible-floor analysis
The rule is genuinely NON-separable (each row has its own [min,max] col span, each col its own
[min,max] row span) so it requires FOUR full-canvas directional OR planes — the structural core. On the
FIXED 10×10 active grid every working plane is only 100 elems, ~6× smaller than task350's 26×24, which
is why this lands ~1.7 pts above task350's 15.27 floor despite identical structure. Memory breakdown:
- cyan_f32 fp32 slice = 400B (Slice preserves input fp32 — irreducible entry plane)
- C fp16 cast = 200B
- 4 MaxPool OR planes (fp16, 200 each) = 800B — the irreducible 4-direction core
- hprod, vprod, ssum (fp16) = 600B ; spanb bool = 100B ; L uint8 = 100B
- **L30 uint8 30×30 = 900B** — the dominant single plane; the one full-canvas index the Equal one-hot
  must broadcast against. Padded with sentinel 99 so off-grid cells match NO channel (all-zero column),
  matching convert_to_numpy which leaves off-grid cells all-zero (NOT background ch0=1).
Total 3100. The fp32-as-uint8 colour-index + Equal→BOOL route saved the 900B bool-cast carrier that a
Where path forces (task350 paid that tail); single-colour output makes the index plane trivially {0,8}.

## OPEN ANGLES (re-attack backlog)
- Shave the L30 900B: only possible if the FREE output could be built from a <30×30 cond — blocked, a
  non-separable 2-D mask needs one full 30×30 plane.
- 4→2 OR planes: derive suffix from prefix via per-line total — every analytic variant still needs 2
  full planes per axis (task350 confirmed no win). Tie at best.

## INSIGHT (transferable)
⭐ task350's directional-prefix/suffix-OR span-fill, re-applied on a **FIXED-SMALL active grid**, jumps
from MARGINAL (15.27 @ 26×24) to a clean +0.47 (16.95 @ 10×10) — the 4-direction floor scales with
active-region area, so the SAME structure that's "at floor" on a big variable canvas is a solid win on
a small fixed one. Re-check every gen-bounded span/fill task for its true active size before bailing.
⭐ Single-colour span/fill: skip the Where(colour_oh,input) tail entirely — build a uint8 colour-index
plane (0/colour), Pad with a 99 sentinel (off-grid → matches no channel), Equal(L,arange)→BOOL FREE
output. Drops the 900B bool-carrier the Where path forces. Sentinel pad is load-bearing: off-grid cells
must be all-zero one-hot (convert_to_numpy leaves them blank), NOT background ch0=1.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1300/17.82; crop_f 400B min fp32 entry, 4 MaxPool span planes irreducible (Mul not Add), ConvInteger needs {0,2} symmetric zp. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
