# task029 — 1c786137

**Rule:** A `height x width` (10–25 each) grid of random static (3–4 colours).
One hollow rectangle ("zoom box") is drawn in a `zoom_color` that the static
never uses; its border occupies the perimeter of an axis-aligned bbox. The
output is the box INTERIOR (size `zoom_height x zoom_width`), i.e. the static
content strictly inside the ring, translated to the top-left. Out-of-region
output cells are all-background (all-channels-off, since the target grid is
exactly the small interior, zero-padded).
**Current (prior):** 15.206 pts, custom:task029, mem 17742, params 181.
**Target tier:** B — data-dependent variable-size crop WITH translation (the
interior must move to (0,0)), so not Tier-S/A; a Gather translate of a single
colour-id plane is the minimal admissible form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: per-ch bbox+ring detect, colorid Conv, 2x Gather crop, region-mask Where→L, Equal→output | B | 17742 | 181 | 15.21 | — | baseline |
| 1 | sentinel-PAD the uint8 id plane ([1,1,31,31], pad value 10) + send out-of-region gather idx to row/col 30 ⇒ gathered crop IS the label map: removes regionB + rd/cd/rokB/cokB + L-Where | B | 16937 | 129 | 15.255 | 200/200 | better |
| 2 | + compare fp32 counts rc/cc directly to fp32 bbox dims (drop rcf/ccf fp16 recasts); fp16 Equal for winner select (drop score_i/smax_i int32) | B | 15773 | 129 | 15.326 | 500/500 | best |

## Best achieved
**15.326 @ mem 15773 params 129 — fresh 500/500.** Beats prior 15.206 by **+0.12**.
Adopt-recommend: **N** (MARGINAL, < +0.3 threshold).

## 2026-06-19 re-golf vs kojimar7113 (deployed = ext:kojimar7113, P=15.7118, mem 10776)
The deployed crowd net (kojimar) is LEANER than our prior custom: it skips the
colour-id plane entirely and does the translate-crop via **one Einsum**
`bcrs,ir,js->bcij` straight into the FREE 10-ch output, with TWO 30x30 fp32
OneHot selectors (3600 each) as the only big carriers, plus per-channel row/col
counts (1200 each) for ring detection. Total 10776.
- Re-derived the EXACT kojimar breakdown: row_selector 3600 + col_selector 3600
  + row_counts 1200 + col_counts 1200 + tiny = 10776 (anchored via evaluate()).
- Built a count-detection + **colorid double-Gather** variant (reuse kojimar's
  lean count/ArgMax detection, swap Einsum for a sentinel-padded uint8 colorid
  crop). Measured: **mem 10883, pts 15.6994, fresh 200/200** — TIES, does not beat.
  Reason: the crop carrier costs ~7200 EITHER way: Einsum needs two fp32
  selectors (7200, pinned fp32 because Einsum must match the FREE fp32 10-ch
  input); colorid needs the f32 entry plane (3600, ReduceSum/Conv channel
  reduction is fp32-pinned) + u8 gather chain (colorid 900 + pad 961 + crop_r
  930 + L 900 = 3691) ⇒ 7291. The two approaches are within 1% of each other.

## VERDICT: MARGINAL — cannot beat +0.3 (need mem<=8008; floor ~9600)
Two structurally irreducible cost centres:
1. **Crop carrier ~7200 fp32.** A data-dependent translate+crop of a 10-ch grid
   into the free output is either (a) Einsum with two fp32 OneHot selectors
   (fp32 forced to match the free fp32 input; fp16 selectors need an fp16 input
   = 18000B cast, net loss — VERIFIED Einsum rejects mixed dtype), or (b) an
   fp32 colorid entry + uint8 gather chain. Both ~7200. The interior is small
   but its POSITION is data-dependent ⇒ no static Slice (symbolic-dim trap).
2. **Detection counts 2400 fp32.** Ring colour = unique channel with
   total==2*rowmax+2*colmax-4 AND max row/col occurring >=2x. Needs BOTH
   per-channel axis-count planes [1,10,30,1]+[1,10,1,30] (1200 each); ReduceSum
   output is fp32-pinned and both axes are required for the bbox + perimeter test.
Floor ~9600 -> best achievable ~15.82 (+0.11). Even an idealized mem-9000
(drop one count plane via a colorid ring-mask, save ~300) only reaches +0.18.
The deployed kojimar net is already at the practical floor; KEEP it.

## Irreducible-floor analysis
Dominant intermediates (measured via ORT profile trace, ORT_DISABLE_ALL):
- **colorid_f 3600 B** fp32 [1,1,30,30] — the 10→1 colour-index entry plane.
  IRREDUCIBLE per FLOOR_RESEARCH: the channel reduction must output fp32; the
  grid is up to 25×25 but Conv output is full-canvas and slicing the 10-ch input
  to 25×25 is a 25 000 B intermediate (net loss).
- **rc / cc 1200 B each** fp32 [1,10,30,1] / [1,10,1,30] — per-channel row/col
  counts. ReduceSum follows the fp32 input dtype; per-channel-per-position is
  300 elems = the floor for the ring discriminator (need BOTH axes for bbox +
  full-row/full-col test).
- **rlo/rhi/clo/chi 600 B each** fp16 [1,10,30,1] — masked index planes for the
  per-channel bbox min/max. Min AND max each need their own Where-sentinel plane
  per axis (4 total); ArgMax/ramp reformulations were measured to need the same
  or more 30-length planes.
- crop side: colorid 900 (u8, feeds Pad) + colorid_p 961 + crop_r 930 + L 900 —
  the unpadded `colorid` is the unavoidable materialised Pad input; the padded
  plane is the cheapest sentinel carrier (the alternative — route region via a
  [1,10,30,30] Equal And-chain — is a 9000 B intermediate).

## OPEN ANGLES (re-attack backlog)
- Eliminate per-channel detection by finding the ring's interior bbox purely on
  the single colorid_f plane (saves ~9000 B of [1,10,...] machinery). Blocked:
  the robust ring discriminator is "the unique colour whose bbox perimeter is
  fully its own colour" = exactly the 2-full-rows ∧ 2-full-cols per-channel test;
  a single-plane horizontal/vertical-run detector is faked by random static.
  If a static-robust single-plane corner/perimeter Conv detector exists this is
  the only path to +0.3 (would land ~16.5).
- ArgMax-based bbox (first/last present row): measured to need an fp16 recast +
  a reversed plane per axis ⇒ no net saving over the Where-sentinel planes.

## INSIGHT (transferable)
⭐ For a translated variable-size crop, route out-of-region cells to a SENTINEL
by **Padding the uint8 id plane with a value ≥ 10** ([1,1,N+1,N+1]) and steering
the per-axis gather index vectors (tiny [N] f16) to index N (the sentinel
row/col) wherever `r > interior_h` / `c > interior_w` via a Where on the index
vector. The gathered crop then IS the label map — eliminates the separate
region-mask plane AND the label-select Where (here −1980 B before re-pad), with
only one Equal(L, arange)→free-BOOL-output at the end. Cheaper than masking the
[1,10,30,30] output. Also: compare fp32 ReduceSum counts DIRECTLY to fp32 bbox
dims (Equal is exact for ints ≤30) instead of recasting the count planes to
fp16 — kills two 600 B [1,10,...] recasts.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-4 opus agent). Do not re-attempt without a new mechanism.

## S9 (2026-07-03) — kojimar 7184.85 teacher: GridSample crop + moment detection (+0.667) ADOPTED
Teacher (overrides/task029.onnx; base_submission = our own mechanism): mem 10736→5436,
params 34→89. Gates: stored fail=0 (re-checked); fresh 2500 uncached: both 0 fails;
no TopK (ArgMax-on-u8 is safe). Latency 0.13ms.
OLD FLOOR REFUTED on two counts:
1. Crop carrier: ONE fp16 GridSample grid [1,30,30,2]=3600B replaces 2×3600B fp32 OneHot
   einsum selectors — GridSample's grid has NO dtype-match constraint vs the fp32 input;
   out-of-interior coords pushed to 3.0 + padding_mode=zeros. 7200→3600.
2. Detection: hollow-rect ring colour via exact moment identity
   n²(4(Σr²+Σc²)−n²−16n+16)==4(Σr²−Σc²)² ∧ n>7 on [1,10] free-input einsums (O(1) bytes)
   replaces 2400B axis-count planes; only 120B presence planes remain for bbox ArgMax.
Backup reports/retired_networks/task029_pre_s9.onnx.
