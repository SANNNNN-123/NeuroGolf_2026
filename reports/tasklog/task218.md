# task218 — 90c28cc7

**Rule:** The 21×21 input holds ONE rectangular "quilt": a (tall×wide) arrangement of solid
axis-aligned colour patches (each one colour 1..9, NO dropout — fully filled), placed at
(rowoffset,coloffset). Block-row i has height depths[i]; block-col j has width lengths[j].
tall,wide∈{2,3}; sum(depths)<21, sum(lengths)<21. The generator guarantees no two block-rows
are identical and no two block-cols are identical, so ADJACENT block-rows/cols always differ in
≥1 cell (block boundaries are detectable from colour changes). OUTPUT is the (tall×wide) grid
whose cell (bi,bj) is the colour of that block. Every block is single-coloured & non-empty, so
block colour = sum(colour-index over block)/count(non-bg cells).

**Current:** 15.29 pts (prior P). **Built:** 15.624 pts, mem 11651, params 146, fresh 200/200 (+800/800 stress).
**Target tier:** A (data-dependent downsample as a double weighted MatMul; no flood-fill — the
quilt is a fully separable rows×cols partition recoverable from per-axis colour-change boundaries).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | task184 idiom, boundary=full-plane row/col Equal; SLICED the 10-ch input | A | 33331 | 63 | 14.58 | 266/0 | correct but 17640B [1,10,21,21] slice plane |
| 2 | Conv on FREE input, crop SINGLE-channel colf to 21×21 | A | 19291 | 63 | 15.13 | 266/0 | killed the 10-ch slice |
| 3 | all downstream fp16 (cast colf→fp16, fp16 MatMuls; CumSum stays fp32) | A | 16351 | 62 | 15.29 | 266/0 | = P, not enough |
| 4 | boundary via per-row/col weighted SIGNATURE (one fp32 MatMul vec) | A | 10843 | 104 | — | 265/1 | one signature collision |
| 5 | TWO independent signatures (w=[i+1, (i+1)²]), boundary if EITHER differs | A | 11651 | 146 | **15.624** | 200/200 +800/800 | **WIN +0.33** |

## Best achieved
**15.624** @ mem 11651 params 146 — adopted? N (build agent does not adopt). Beats prior 15.29 by
**+0.33** (≥+0.3 → adopt-recommend YES). Generalizes 200/200 fresh (+800/800 stress, 21×21 fixed).

## Method (exact)
colf30 = 1×1 Conv Σ k·input_k on the FREE [1,10,30,30] input (fp32, never slice the 10-ch input).
Crop to colf32 [1,1,21,21] fp32; cast → colf fp16 (882B) for the full-plane value/occupancy ops.
occ = colf>0 → fp16; row/col occupancy ReduceMax → in-quilt extent rq_b,cq_b. Boundary detection:
sigrow = MatMul(colf32, Wcol[1,1,21,2]), sigcol = MatMul(Wrow[1,1,2,21], colf32) give TWO weighted
per-row/col signatures (w1=i+1, w2=(i+1)², both fp32-exact, max≈83k<2²⁴). A row r≥1 starts a new
block iff EITHER signature differs from row r-1 (slice tiny [1,1,20,2] sigs, Equal, ReduceMin over
the 2-sig axis, Not); prepend True for row 0; AND with rq_b. block index bri = inclusive
CumSum(newblock,fp32) − 1 (bg rows → −1, excluded by Equal vs 0..2; trailing-bg rows that retain
the last cumsum are excluded by ANDing the selector with rq_b). Same for bci. One-hot selectors
RselH/CselH [1,1,K,21]/[1,1,21,K] fp16 (K=3). Snum=Rsel@colf@Csel, Sden=Rsel@occ@Csel (all fp16,
sums<2048 exact). colour = Round(Snum/max(Sden,1)) where Sden>0 else sentinel 99 → uint8 K×K →
Pad(99)→30×30 → output = Equal(L,arange[1,10,1,1]) BOOL.

## Irreducible-floor analysis
Dominant: colf30 [1,1,30,30] fp32 = 3600B (Conv colour-index entry; Conv inherits the fp32 input
dtype, casting the 10-ch input to fp16 costs 18000B). Second: colf32 [1,1,21,21] fp32 = 1764B (the
crop, feeds both signature MatMuls — fp16 sigs would overflow 2048). Then L (900 uint8 padded
label, output broadcast carrier — Pad rejects bool so pad uint8), colf/occ (882 each fp16). The
signature trick replaced the v1 ~6×840B full-plane comparison planes with [1,1,21,2] vectors
(~168B), the bulk of the v3→v5 drop.

## OPEN ANGLES (re-attack backlog)
- Drop colf32 (1764) by reading signatures from colf30 fp32 directly with 30-len weights and
  casting colf30→fp16 for the value path: TRIED — colf16_30 (1800) > colf32 (1764) and +36 params,
  net WORSE (11831, 15.61). Rejected.
- Compute signatures in fp16 with bounded weights (Σw·9<2048): would remove the 1764 fp32 crop
  (→ ~9900, ~15.79) but tight margin risks fp16 collisions on adjacent distinct rows; would need
  careful weight design + heavy fresh verification. ~+0.16 upside, untried (time).
- occ (882 fp16) feeds only the Sden count MatMul; replacing the count-divide with a per-block
  masked ReduceMax colour would drop occ+occ_b but needs ≥3 per-block planes (larger). No win.

## INSIGHT (transferable)
⭐ A "quilt of solid colour patches → compress to its block grid" with NO separator lines is the
same separable rows×cols downsample as task184 — recover per-axis block boundaries from COLOUR
CHANGES (the generator guarantees adjacent block-rows/cols always differ) instead of all-bg lines.
⭐ Replace an O(N²) full-plane adjacent-row comparison (6×~840B planes) with cheap per-row/col
WEIGHTED SIGNATURE vectors: sig[r]=Σ_c w[c]·colf[r,c] via ONE MatMul (contracts the column axis →
[1,1,N,1]); boundary = sig[r]≠sig[r−1] on the tiny vector. A single linear signature can COLLIDE
(distinct profiles, equal weighted sum) — use TWO independent weight vectors as MatMul COLUMNS
(W[N,2]) and flag a boundary if EITHER differs; collapses the collision rate to ~0 (verified
800/800). ⭐ When cropping a colour-index plane to a smaller fixed active canvas, slice the
SINGLE-channel Conv output (1764B), NEVER the 10-channel input (17640B).

---

## RE-GOLF 2026-06-19 (plane elimination) — 15.62 -> 15.94 (+0.31)

Re-measured prior adopted net: mem 11651, params 146, pts 15.6244 (266/266).

| # | angle | mem | par | pts | fresh | outcome |
|---|---|---|---|---|---|---|
| 1 | sigs+downsample+occupancy all off the ONE colf30 plane (kill colf32/colf/occ); area from selectors not occ | 9626 | 237 | 15.803 | 200 | kept |
| 2 | ReduceSum for area (drop ones inits) | 9626 | 177 | 15.810 | 200 | kept |
| 3 | expand 10-ch on KxK then Pad into FREE output (kill 900B carrier L) | 8906 | 177 | 15.886 | 200 | kept |
| 4 | grouped-Conv finite-diff for boundary | 9096 | 187 | 0.0 | 0 | REVERTED (ORT grouped-Conv bug + bigger) |
| 5 | bool And-of-channel-slices replaces fp16 ReduceMin bridge | 8674 | 179 | 15.911 | 200 | kept |
| 6 | fold -1 into selector ramp Equal(cumsum,R+1) -> kill bri/bci | 8434 | 178 | 15.939 | 500 | **ADOPTED** |

**Best: 15.939 @ mem 8434 params 178**, +0.3147 over prior. Fresh 500/500 all 4 shapes.

**Dominant irreducible intermediate:** colf30 3600B fp32 [1,1,30,30] — the 10->1 colour-index
entry plane. Conv output inherits fp32 input dtype; every downstream contraction (sig MatMuls
AND Snum=Rsel@colf30@Csel) taps this exact plane, so it is materialised once at full fp32.
Folding the colour reduction into the downsample on raw input keeps 10 channels -> also 3600B.

**Transferable lever (compounded):** on an already-working downsample net, (1) run EVERY
downstream op off the single fp32 colour-index entry plane (no 2nd cropped colour plane, no
occupancy plane — per-block AREA = ReduceSum(Rsel)*ReduceSum(Csel) replaces Rsel@occ@Csel);
(2) expand the 10-ch one-hot on the TINY KxK colour grid then Pad it into the FREE uint8
"output" (kills the 900B 30x30 carrier an Equal-to-output usually pins). Micro: bool And of
two channel-slices beats the fp16-ReduceMin "any-hash-differs" bridge; fold cumsum's -1 into
the selector ramp (Equal(cumsum,R+1)) to delete the bri/bci index planes. Grouped-Conv
finite-diff "shortcut" backfired (ORT 1.26 grouped-Conv bug) — verify grouped-Conv before use.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
