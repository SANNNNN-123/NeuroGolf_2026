# task158 — 6aa20dc0 (variable-mag dihedral sprite re-stamping)

**Rule:** A canonical 3x3 sprite (c0@(0,0), c1@(2,2) distinct corner colours; transpose-symmetric
c2 body on off-diagonal cells; bg=4th colour). 2-4 "megas" are placed, each with its own
magnification mag∈{1,2,3} and own (hflip,vflip), non-overlapping with margin 2. Mega 0 (the
REFERENCE, always mag=1) is fully drawn; every other mega shows ONLY its two magnified diagonal
CORNER blocks. OUTPUT fills every mega with its full flipped-magnified sprite.
**Current (prior):** 12.85 pts (public net).
**Target tier:** detection/B — non-local stamp+correspondence; the wall was wrongly believed to be
sprite RECOVERY (prior agent capped ~46%). It is NOT — recovery is closed-form and stamping is separable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full 30x30 fp32, 4 role-spread ConvTranspose | det | 933215 | 8882 | 11.24 | 266/266 | exact but heavy |
| 2 | fp16 working planes + combined write | det | 368081 | 8893 | 12.16 | 200/200 | — |
| 3 | channel-contracting Convs (Fcount/value as [1,12,9,9] weight) | det | 193489 | 8887 | 12.78 | 200/200 | — |
| 4 | forward-spread = Conv (pad-topleft) not ConvTranspose; no 38x38 | det | 193489 | — | 12.78 | 200/200 | — |
| 5 | crop pipeline to WORK 26x25 active region (generator size cap) | det | 152769 | 8902 | **13.01** | 200/200 | EXACT, adopted |
| 6 | (alt) write-all, drop exact-cover confirmation | det | 102719 | 6958 | 13.39 | 200/200 | 0.04% LATENT LEAK — NOT adopted |

## Best achieved
**13.01 @ mem 152769 params 8902 — EXACT, generalizes (ISOLATED fresh 200/200; ONNX==target 2000/2000;
solve()==target 50000/50000).** Beats prior 12.85 by +0.16 (MARGINAL by the strict +0.3 rule, but a
TRUE fully-generalizing gap-closer). A write-all variant scores 13.39 but fails 0.04% of instances
(phantom over-stamp) so it would zero those on the LB — rejected in favour of exactness.

## Algorithm (the prior "wall" dissolved)
1. bg = MODE colour (per-channel pixel-count ArgMax). c2 = non-bg colour with MIN bbox span
   (it is confined to the reference 3x3). The two remaining present colours are {c0,c1}.
2. refpos = (min row, min col) of {c2 cells} ∪ {non-bg-non-c2 cells within Chebyshev-2 of the
   c2 bbox} — deterministic, NO window search (verified 2000/2000). canon = the 3x3 window at
   refpos, hflipped iff the "special" corners lie on the anti-diagonal (→ c0,c1 on main diagonal).
3. For each (mag,h,v): match where the two corner blocks == c0/c1 AND the rest of the (3·mag)²
   footprint == bg (a Conv on [eq_bg,eq_c0,eq_c1] with a STATIC [12,3,9,9] kernel; off-grid sentinel).
4. EXACT-COVER collapses to ONE SPATIAL pass (no Loop, no isolation ring, no ref self-exclusion):
   Fcount = Σ_configs forward-spread(match by the Sz² block); uniq = (Fcount==1); a placement is
   CONFIRMED iff its visible mask overlaps a uniq cell. Write confirmed tiles. (20000/20000 numpy.)

## Irreducible-floor analysis
Dominant: four fp16 [1,12,26,25] planes (sat, match, vsat, confirmed ≈15.6KB each) + three bool
[1,12,26,25] gates (≈7.8KB each). The 12 channels = 3 mags × 4 flips are ALL distinct (verified
3000/3000 — a transpose-symmetric body still has 4 distinct flips), the active canvas is at its
26x25 floor (generator caps width≤25, height≤26 — verified maxH=26,maxW=25 over 1000 fresh), and the
exact-cover REQUIRES per-config gating (the value-write needs `confirmed[config]` to know which tile
to stamp), so none of the four 12-ch planes can be contracted. Forward-spread Conv (pad-topleft)
instead of ConvTranspose removed all 38x38 planes; channel-contracting [1,12,9,9] weights removed the
post-spread 12-ch planes. That is the architectural floor for the EXACT net.

## OPEN ANGLES
- Drop exact-cover → 13.39 pts but 0.04% phantom-overstamp leak (write-all). Reconsider only if the
  LB tolerates a 0.04% per-instance leak (it does not — a held-out failing instance → 0 on that one).
- A 2-stage match (shared per-mag footprint plane + cheap corner-colour disambiguation) MIGHT cut
  the flip dimension from the dominant `sat`/`match` planes (untried; risky, est ~−15KB).

## INSIGHT (transferable)
⭐ "Variable-mag dihedral sprite re-stamping" is NOT a correspondence/recovery wall. The canonical
sprite recovers closed-form (min-bbox-span colour locates the fully-drawn reference; a single hflip
fixes orientation), and per-object magnify+flip STAMPING is fully separable as 12 fixed strict-
correlation passes. The exact-cover (deduping phantom corner-block alignments) collapses to ONE
spatial pass: footprint-coverage count → uniq cells → confirm placements whose visible mask owns a
uniq cell — NO iterative naked-singles, NO Loop/NonZero. ⭐ FORWARD-SPREAD (stamp a tile at every
marked top-left) = a Conv with the kernel 180-flipped at the BOTTOM-RIGHT + pads=[k-1,k-1,0,0] (NOT
ConvTranspose) → output stays NxN, no (N+k-1)² plane. ⭐ A grouped per-config spread Conv that must be
SUMMED over configs is better written as a single NON-grouped Conv with a [1,Cfg,k,k] weight — it
contracts the config axis in one op and never materialises the [1,Cfg,N,N] plane. ⭐ A data-
INDEPENDENT crop to the generator's max grid (here 26x25) shrinks every working plane with zero risk.
⭐ ORT (current build) keeps fp16 for Conv/ConvTranspose/Equal/ReduceSum/Where/Min/Max under
ORT_DISABLE_ALL — declare every full-grid plane fp16. ⭐ np.ascontiguousarray inflates a 0-dim numpy
scalar to shape [1]; build Gather index scalars WITHOUT it (true 0-dim init; prod([])=1, params-safe)
so the gathered axis is REMOVED (a [1]-index keeps a stray dim → out-of-bounds on the next Gather).

---

## 2026-06-19 RE-PROBE (current deployed = ext:kojimar7113, 14.53 pts) — MARGINAL, did not beat
The crowd net SUPERSEDED our 13.01 EXACT build: kojimar mem=33059 params=2343 (vs our 152769/8902)
by running the 12-pass stamp pipeline on SMALL all-uint8 per-config blocks (`pair_u81/82/83`
[1,4,24,23]/[1,4,21,20]/[1,4,18,17] — one per mag, 4 flips in the channel axis — not our four
[1,12,26,25] fp16 planes). It PASSES fresh 200/200 (re-verified). So kojimar already applied the
uint8 + plane-reduction levers past our recorded "architectural floor."

To beat +0.3 from 14.525 needs total ≤ 26226 ⇒ cut 9176B. Measured cost (static value_info):
- `color_f` f32 [1,1,30,30]=3600B — the colour-index ENTRY, `Conv(input,w_color)`, consumed ONLY
  by `Cast->color_u8`. **fp32-LOCKED**: Conv reads the fixed fp32 graph input, ORT forces weight
  dtype==input dtype, output inherits fp32; fp16 would need an 18000B input cast. IRREDUCIBLE here.
  (The "Conv keeps fp16" lever needs the Conv input to already be a narrow WORKING plane, not entry.)
- `pair_u8*` (2208+1680+1224) + `ab_u8` 1300 + ~90 small 650B/552B uint8/bool planes = the 12
  (mag×dihedral) stamp passes; already uint8, the documented sole buildable backbone.
ONLY safe lever = dedup ~3 bool/u8 near-duplicate pairs (nonbg_bool+nonbg_u8, mask_a_*, mask_b_*)
≈1950B ⇒ pts ~14.58, gain ~+0.057 << +0.3.
⇒ MARGINAL. Algorithm is SOLVED + generalizing; this is a MEM floor, not an accuracy wall, and the
fp32 entry plane plus the irreducible 12-pass uint8 stamp planes pin it. No beating net written.

## 2026-06-30 from-scratch recheck after user challenge

Re-read the generator instead of trusting this log.  Confirmed semantics:

- reference mega is always `mag=1` and fully visible;
- later megas have `mag∈{1,2,3}` and arbitrary h/v flips;
- later input keeps only the two diagonal corner blocks and hides body pixels;
- body mask comes only from the reference 3x3 sprite.

New probes:

| probe | result | conclusion |
|---|---|---|
| reference anchor by `3x3 non-bg count >= 5` and `>=3 distinct non-bg colours` | unique on stored+arc-gen `266/266` | semantic anchor is simple, but ONNX distinct-colour counting would require a 10-channel local count or equivalent, worse than current diagonal detector |
| relaxed anchor by count plus opposite-corner inequality | many false positives | corner non-bg / distinct-colour condition is load-bearing |
| direct-fill oracle that combines pair detection and stamping | `265/266`; one phantom fill case | pair+stamp cannot be fused naively; strict per-config separation prevents accidental corner-pair recombination |
| replace `invalid_or_bg -> Not -> Cast` with direct invalid-mask QLinearConv detector | failed (`194/266`, then `1/266`; int8 QLinearConv unsupported) | current diagonal QLinearConv is not just both-corners-non-bg; it is a tuned threshold over total non-bg plus diagonal weight, and ORT uint8 QLinearConv cannot cheaply express the inverted signed form |

Important correction to the cheap-anchor intuition: the human anchor rule is easier
to state, but not cheaper in this ONNX cost model.  Current graph already avoids a
10-channel local distinct-colour plane.  The counted slack that remains is mostly
small bool/u8 carriers; the only verified deletable-looking 650B plane could not
be removed under ORT's QLinearConv type constraints.

No source/net change adopted.

## S8 (2026-07-02) — counting-model rebuild (+0.524) ADOPTED, bit-identical
Anchor detector (~12.1KB: 5×650B nonbg + 16×552B QLC-diagonal planes) → free-input einsum
MOMENT STATISTICS per colour (n, Σr, Σc, Σr², Σc² from five [10]-output einsums); integer
score n(Σr²+Σc²)−((Σr)²+(Σc)²) ≤ 2n² identifies the 3×3-confined sprite colour (exact fp32,
products <2²⁴). Entry = single-tap valid Conv [1,10,5,6] (label+crop in-op, kills 4.5KB).
Refpos: bbox-min einsum profiles + 4-candidate diagonal corner-pair test (clip needs BOTH
lower/upper validity masks). Backbone kept verbatim. 18263+2647 vs 32979+2340 → +0.524.
30000 numpy 0 fails; 20000 fresh vs live onnx bit-identical (7/20000 shared inherent fails);
fresh_verify 2500+1500 div 0. Adopted via ONNX materialization (cand read incumbent inits).
TRANSFERABLE: moment-statistics einsums = O(1) counted bytes for any "find the small/confined
component colour" detection.

## S9 (2026-07-03) — epilogue-fold 2nd pass: FLOOR re-confirmed (no change)
13a inapplicable: no walk-einsum carrier — output backbone = 6 QLinearConv stamp passes
(3 mags × correlate+spread) → Where→Pad→Equal; nothing to fold the epilogue into.
Byte-rank: lab_f 2600 fp32-locked (Conv on fp32 input), pair_u8{1,2,3} 2208/1680/1224,
ab 1300×2, painted 900+650, fills 650×5. Params 2647 = per-mag kernels 5/8/11
(200/512/968) → task204 reject-check (per-size convs non-collapsible). Both Cast pairs
load-bearing (QLinearConv needs u8; bool Where mux). Front-end already O(1) (S8 moment-stat).
Slack ≈ 0 clean bytes; +0.1 needs −1990B. DO NOT re-probe without a new mechanism.

## S16 (2026-07-06) — int8 mask-subgraph downcast (+0.0084) ADOPTED, bit-identical
S9 "slack ≈ 0" refined, not overturned: the fp32 mask subgraphs (NOT input-derived) were
downcastable. Two `Cast to=1`(float)→`to=3`(int8) on provably-{0,1,2} boolean masks:
(1) `nb44` (corner-validity mask [1,1,4,4]) — int8 propagated through Slice/Mul/Add/Reshape/
ArgMax corner-grid subgraph → −144B; (2) `okf` (candidate-colour mask [10], feeds only
ArgMax) → −30B. Plus dedup: `idx4`→`SH4` (byte-identical [1] int64 =4) → −1 param.
18089+2646 vs 18263+2647 → +0.0084 (15.0520→15.0604). evaluate fail=0; fresh-gate 2000 fresh
vs live onnx: 0 divergence (both 99.85%, same 3 inherent fails). Rebuilt from source, deployed
net bit-identical to candidate. Old net backed up reports/candidates/task158_v1_backup.onnx.
KILLED (measured): fp16 lab_f = NET LOSS (needs [1,10,30,30] input→fp16 +1800B to save 1300B;
Conv fuses label+30→26×25 crop, crop is load-bearing for ab/pair sizes). pair-conv pad
[1,1,1,1]→[0,0,0,0]+compensating stamp pad = 43 fails (correlation alignment shift).
TRANSFERABLE: input-derived fp32 planes are dtype-locked (Einsum needs input's f32), but
DOWNSTREAM bool/mask fp32 planes that never touch input are free int8 downcasts.
