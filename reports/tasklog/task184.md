# task184 — 780d0b14

**Rule:** Input is a (tall×wide) grid of solid axis-aligned colour patches (each one colour 1..9 with
~10% random dropout to bg 0), separated by exactly ONE all-bg row between block-rows and ONE all-bg col
between block-cols (no leading/trailing separators); tall,wide∈{2,3}. Output is the (tall×wide) grid whose
cell (bi,bj) is the patch colour at block-row bi, block-col bj. Verified across 1869 usable samples: every
block is single-coloured and non-empty, so block colour = sum(colour-index over block)/count(non-bg cells).

**Current:** 14.43 pts, gen:thbdh6332, mem 28880, params 9975
**Target tier:** A (data-dependent downsample as a double weighted MatMul; no flood-fill needed because the
partition is a clean separable rows×cols grid recoverable by all-bg-line detection + exclusive CumSum).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | [30,30] selectors, double MatMul, label+Equal | A | 39364 | 124 | 14.42 | 169/169 | correct but heavy |
| 2 | shrink selectors to [K=3,30], small KxK label, Pad | A | 16621 | 105 | 15.28 | ok | +0.85 |
| 3 | drop colf32 dup; occ via Greater; cleanups | A | 15739 | 106 | 15.33 | ok | |
| 4 | 4D MatMul (no [30,30] reshapes), uint8 label | A | 11950 | 109 | 15.60 | ok | |
| 5 | colour-chain fp32 (reuse colf32, no colf16), count-chain fp16 | A | 10786 | 109 | **15.70** | 500/500 | adopt-ready |

## Best achieved
15.70 @ mem 10786 params 109 — adopted? N (build agent does not adopt). Beats prior 14.43? **YES (+1.27)**.

## Method (exact)
colf32 = 1×1 Conv Σ k·input_k (fp32). occ_b = colf32>0 (bool); occ = cast f16 → ReduceMax gives row/col
occupancy → in-grid extent Hm1,Wm1 (max r·rowhas) and all-zero rows/cols. seprow = allzero AND in-grid;
bri = exclusive CumSum(seprow) (CumSum needs fp, kept fp32). Selector Rsel_b[1,1,K,30] = (bri==R) AND
non-separator, built bool then cast to BOTH f32 (colour chain) and f16 (count chain); same for Csel_b/CselOT.
Snum = RselF@colf32@CselF (fp32, reuses the Conv plane so NO separate colf16); Sden = RselH@occ@CselH (fp16).
colour = Round(SnumH / max(Sden,1)) where Sden>0 else sentinel 99 → uint8 KxK → Pad to 30×30 with 99 →
output = Equal(L, arange[1,10,1,1]) BOOL. K=3 keeps all post-MatMul planes ≤180B.

## Irreducible-floor analysis
Dominant: colf32 [1,1,30,30] fp32 = 3600B (the Conv colour-index output; fp32 forced because Conv inherits
the fp32 input dtype, and casting the 10-ch input to f16 would cost 18000B). It does TRIPLE duty (Conv out +
colour MatMul operand + occ source) so it cannot be removed. Second: occ [1,1,30,30] f16 = 1800B, needed by
BOTH the count MatMul and the occupancy ReduceMax. occ_b (900 bool) is the unavoidable Greater output feeding
the f16 cast. L (900 uint8) is the padded label. Together ≈ colf32+occ+occ_b+L = 7200B; the rest is small
[1,1,K,30]/[1,1,K,K] selectors and reductions. ~10.8KB is essentially the 2-full-plane (colour+occupancy)
floor for a count-divide downsample.

## OPEN ANGLES (re-attack backlog)
- Eliminate the count plane occ entirely by reading each block's colour as a per-block MAX instead of
  sum/count — but MatMul is a sum-semiring; a separable masked-max over a data-dependent block needs a
  per-block masked ReduceMax (3 planes) which is larger, so not obviously a win. If a 0-param Conv could
  emit max it would drop occ(1800)+occ_b(900) → ~+0.25.
- colf32 fp32 (3600) is the last big plane; only removable if the colour-index could be produced in f16
  without a 10-ch f16 input plane (no current op path).

## INSIGHT (transferable)
⭐ A clean "segment into a tall×wide GRID of patches separated by single all-bg lines + label each block"
is NOT a flood-fill/connectivity wall: it is a fully SEPARABLE rows×cols partition. Recover the per-axis
block index with an EXCLUSIVE CumSum of the all-bg-line indicator (gated to the in-grid extent so trailing
zero rows aren't counted as separators), build [1,1,K,30] one-hot SELECTOR matrices (K=max blocks=3), and
downsample with a double MatMul Snum=Rsel@colf@Csel, Sden=Rsel@occ@Csel; per-block colour = Snum/Sden
(exact because each block is single-coloured & non-empty). The output's data-dependent (tall×wide) shape is
handled "for free" by the selectors: rows R≥tall / cols C≥wide get Sden=0 → sentinel → all-zero output cells,
so a fixed KxK→Pad(99)→Equal(arange) emits the correctly-sized top-left block with zeros elsewhere — no
NonZero/Compress. ⭐ Split a double-MatMul into a fp32 colour chain (reusing the Conv's existing fp32 plane,
killing the fp16 colour cast) and a fp16 count chain — selectors are tiny so keeping two dtype copies is
far cheaper than one extra full [1,1,30,30] plane.


## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.048, policy-gated)

Clean adoption (candidate ≤ incumbent on every gate). Same separable grid-partition
downsample mechanism as the incumbent (CumSum-selector + double contraction), only the
Conv realization changed.

**Mechanism diff (op census, retired vs new):** the two colour/occupancy `Conv` nodes →
`QLinearConv` (int8), the two `Relu` nodes are dropped, and `Cast` count rises 2→6 (the
int8 quant/dequant plumbing). The 3× `Einsum`, 2× `Sign`, 2× `CumSum`, 2× `Equal` and final
`Pad` — i.e. the exclusive-CumSum block-index selectors and the Snum/Sden separable
contraction downsample — are unchanged (16→18 nodes). Colour/occupancy Conv outputs are
integer-valued and only get compared/reduced, so int8 quant is exact here.

**Cost:** mem 2460→2340, params 60→63, pts 17.1680→17.2155 (**+0.048**, cost 2520→2403 −117).

**Gate evidence:** bundled 169/169 fail=0 (both nets). Fresh 2×2000: candidate 0 fails,
incumbent 0 fails, 0 divergence. TopK audit: no TopK in either net.

**Backup + provenance:** incumbent → `reports/retired_networks/task184_pre_s10.onnx`;
candidate source `public_candidates/bobmyers7186/task184.onnx` → `networks/task184.onnx`;
source regenerated via live_to_exact_source --write-src, src↔live reconciled fail=0.

⭐ TRANSFERABLE: on an A-tier separable-partition downsample, the colour/occupancy Conv
output is an **integer** that is only ever compared or reduced → int8 `QLinearConv` (and
dropping the now-redundant `Relu`) is bit-exact and cheaper. Selection: separable
grid/patch-partition nets (task184-family) with a fixed-kernel colour-sum or occupancy Conv
feeding only `Greater`/`Equal`/`ReduceMax`. Gain is small (−117 here); low priority unless
the plane is large.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
