# task143 — 63613498

**Rule:** size=10 grid with a gray(5) L-marker at row3/col3. 3–5 monochrome
"creature" sprites (3–4 contiguous px, each fitting a 3×3 box anchored at (0,0))
at non-overlapping positions, DISTINCT colours (≠gray, ≠one special `bcolor`).
Sprite 0 is special: its raw creature shape is also drawn in `bcolor` ALWAYS in
the top-left 3×3 corner (rows0-2/cols0-2), and its PLACED copy is recoloured to
gray in the output. INPUT→OUTPUT differs ONLY at sprite-0's placed cells (colour
colors[0] → gray 5). Creatures are unique + colours distinct ⇒ sprite 0 is the
UNIQUE placed sprite whose shape matches the corner reference R.
**Current:** 15.12 pts, ext:kojimar6275, mem 19384, params 57 (very bloated;
labelled confirmed-infeasible with BLANK note → FALSE POSITIVE).
**Target tier:** detection→A — closed-form shape-match on a 10×10 active region.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice10×10 fp16, grouped-Conv corr per cand channel, select matched channel plane, Pad+Where into free output | A | 14980 | 142 | 15.376 | — | MARGINAL +0.25 |
| 2 | drop bg ch0 by slice (9ch, no Gather); kill corr_ok/match planes via corr−BIG·cornermask then ReduceMax==K | A | 11625 | 139 | 15.627 | 200/200 | ADOPT +0.50 |

## Best achieved
15.627 @ mem 11625 params 139 — beats prior 15.12 by **+0.50**. ReferenceEvaluator
and ORT(ORT_DISABLE_ALL) agree; isolated fresh 200/200.

## Irreducible-floor analysis
Entry Slice to the 9-channel 10×10 region is fp32 (Slice preserves dtype) = 3600B,
irreducible. The per-channel correlation planes `corr` and `corr2` are fp16
[1,9,10,10] = 1800B each — intrinsic to per-channel shape matching (sprites are
monochrome in distinct channels, so matching must be per-channel; a combined
foreground plane risks false R-matches across touching sprites, spacing=0). The
30×30 mask must be materialised twice (uint8 Pad 900 + bool Cast 900) because the
final Where needs a 30×30 bool condition and Pad rejects bool.

## OPEN ANGLES (re-attack backlog)
- Could collapse corr2 into corr by zeroing cand's corner pre-Conv, but that costs
  another [1,9,10,10] plane — no net win. corr2 is the cheapest corner-suppression.
- The 2×900 mask30 stage: if a future op let a 10×10 bool condition broadcast/pad
  to 30×30 in one step it'd save ~900B (→ ~15.7 pts).
- Entry 3600B fp32 is the dominant floor; only a non-fp32 input encoding could cut it.

## INSIGHT (transferable)
⭐ A "shape-correspondence / match-the-reference-sprite" task is NOT a detection
wall when (a) the active canvas is small (size bound from the generator → slice to
10×10) and (b) sprites are monochrome in DISTINCT colour channels: per-channel
grouped-Conv correlation with the reference kernel + a per-channel pixel-COUNT
gate (==K) uniquely picks the matching channel, and that channel's whole plane IS
the target mask (no dilation needed). Recover the reference kernel from a
generator-fixed corner region (here the bcolor ref is ALWAYS in the top-left 3×3).
Kill a self-match at a known fixed window via `corr − BIG·cornermask` then
ReduceMax==K — cheaper than building separate Equal/And bool match planes.
This re-triages a "confirmed-infeasible (blank note)" task to +0.50.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 838 -> 837 (+0.001); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].