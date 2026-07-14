# task211 — 8d5021e8

**Rule:** Input is a fixed 3-row × 2-col grid with pixels of a single colour on a
black canvas. Output is a 9×4 grid: the input reflected about a horizontal mid-axis
(cols 0–1 = mirror, cols 2–3 = direct) and tiled vertically 3× as [flip|direct|flip].
Every output cell (R,C) COPIES exactly one input cell: r=row_src[R], c=col_src[C] with
row_src=[2,1,0,0,1,2,2,1,0], col_src=[1,0,0,1]. Pure spatial copy (Tier-S geometry).
**Current:** 17.67 pts, GridSample(fp32 9×4)+Pad, mem 1440, params ~81
**Target tier:** S (pure copy) — but bounded by the irreducible 9×4×10 sample tensor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice 3×2 → fp16 → 2× Gather(col,row) → Pad | S | 1320 | 28 | 17.79 | 200/200 | marginal |
| 2 | Slice 3×2 → fp16 → GridSample 9×4 (fp16) → Pad | S | 1080 | 86 | 17.94 | 200/200 | BEST, marginal |

## Best achieved
17.94 @ mem 1080 params 86 — adopted? N (caller decides). Beats prior 17.67? +0.27 (MARGINAL, <+0.3).

## Irreducible-floor analysis
Dominant intermediate = the pre-pad sample tensor [1,10,9,4] fp16 = 720B. It cannot be
removed (the 9×4 output content, 10 channels, must materialize once) nor narrowed below
fp16 (Pad rejects bool, GridSample outputs float). The fp16 *entry* costs 360B: a fp32
Slice [1,10,3,2] (240B) + Cast→fp16 (120B). The slice is mandatory because GridSample's
output dtype follows its input dtype — to get a fp16 720B sample (vs the public net's
fp32 1440B) the active region must be cast, and casting the *whole* input is 18000B, so
the cast is applied to the tiny sliced region. GridSample beats a 2-stage Gather here
because ONE op makes the [1,10,9,4] result, avoiding the col-gather's extra [1,10,3,4]
(240B) tensor — that's the 1320→1080 win.
Params: grid [1,9,4,2]=72 dominates; slice(6)+pad(8)=14. At opset 16 (GridSample needs
≥16) Slice/Pad MUST use input-tensor form so their amounts ARE counted — attribute-style
(free-param) Slice/Pad is only available at opset ≤9/≤2 and is incompatible with
GridSample. Total = 1080 + 86 = 1166; +0.3 needs ≤ e^(25−17.97) ≈ 1129, i.e. 37B short.

## OPEN ANGLES (re-attack backlog)
- Collapse to a 1-channel colour-index plane via Conv(bias=+1) then expand to 10-ch
  one-hot in the FREE Equal output. FAILS: the 10-ch expansion needs a [1,1,30,30] fp16
  plane (1800B) before Equal — the padding-to-30×30 in 1 channel is dearer than keeping
  10 channels at 9×4 (720B then Pad→free). Net worse (~2148B). Recorded as a dead end.
- Sample only the 3-row middle block (grid [1,3,4,2]=24 params) and rebuild 9 rows via
  flip+Concat — saves 48 params but the extra flip(240B)+Concat(720B) tensors cost more
  mem than the 48 params saved. Dead end.
- A fp16 entry cheaper than 360B (slice 240 + cast 120). No op slices-and-casts in one;
  casting the full input is 18000B. Appears to be a hard floor.

## INSIGHT (transferable)
⭐ GridSample on a CAST-fp16 SLICE of the active region halves the sample-tensor cost vs
the public fp32-GridSample idiom (gs 1440→720), because GridSample output dtype follows
input dtype: pay a tiny fp32 Slice + fp16 Cast on the active K×K block first. One
GridSample also beats a 2-stage row/col Gather (no intermediate gather tensor).
⭐ But it caps at +0.27 here: with GridSample you're locked to opset≥16, where Slice/Pad
amount-tensors ARE counted as params (the free attribute-form Slice/Pad only exists at
opset ≤9). When the win hinges on a ~70B grid + ~14B slice/pad, that opset tax is what
keeps a pure-copy task marginal. Discriminator: a fixed-small copy/mirror/tile is genuine
Tier-S geometry, but the single materialized output-content tensor (here 10ch×9×4 fp16 =
720B) plus the fp16-entry + grid is a real ~1166B floor — re-derive e^(25−P_target)
before promising +0.3.

## S9 (2026-07-03) — mechanism-14 separable-remap einsum (+0.277) ADOPTED
Single 5-operand Einsum 'ra,ai,zcij,bj,sb->zcrs', mem=0: mirror-stack 3x2->9x4, rowmap [2,1,0]x3 colmap [1,0,0,1], 396->300.
Gates: stored fail=0; uncached fresh 2000+600: 0/0/0 (bit-identical). No TopK.
NOTE: scan projection was ~8x optimistic — output axis must span the FULL 30 (grading
tensor [1,10,30,30]), so U tables are [30,K] not [out,K]. Backup task211_pre_s9.onnx.
