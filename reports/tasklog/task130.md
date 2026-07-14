# task130 — 5614dbcf

**Rule:** The 9x9 input is a FIXED 3x3 tiling of 3x3 blocks. Each block is either a solid box of
one non-gray colour (1..9, gray=5 excluded) or a noise block (background 0 + a few scattered gray
pixels). Gray may also overwrite a handful of a box's cells. Output is the 3x3 grid whose cell (R,C)
is the box colour of block (R,C), or 0 if no box. Because gray and bg are the only non-box values, a
block's box colour = (sum of colour-index over the block excluding gray & bg) / (count of those cells);
box blocks have count>0, noise blocks count==0 -> 0.

**Current:** 16.67 pts (public net).
**Target tier:** A/S (fixed-stride downsample as two strided Convs — no full-size plane materialises).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | double-MatMul (task184 idiom), const selectors, colf [1,1,30,30] | A | 7902 | 392 | 15.98 | ok | 3600B colf32 entry caps it; <16.67 |
| 2 | two stride-3 Convs -> [1,1,10,10] (Snum,Sden); 30x30 Equal carrier 900B | A | 3109 | 207 | 16.89 | ok | no full plane; +0.22 only |
| 3 | one-hot the small 3x3 label then Pad uint8 into the FREE output | A/S | **2389** | **207** | **17.14** | **200/200** | adopt-ready, +0.47 |

## Best achieved
17.14 @ mem 2389 params 207 — adopted? N (build agent does not adopt). Beats prior 16.67? **YES (+0.47)**.

## Method (exact)
W_color[1,10,3,3]: per-channel weight tiled over the 3x3 kernel, = k for k!=0,5 else 0 (drops bg & gray).
W_occ[1,10,3,3]: = 1 for k!=0,5 else 0. Snum = Conv(input, W_color, stride=3) -> [1,1,10,10] = block
colour-sum; Sden = Conv(input, W_occ, stride=3) -> [1,1,10,10] = block box-cell count. Cast both to f16,
valid = Sden>0, colour = Round(Snum / max(Sden,1)) where valid else 0 -> uint8 [1,1,10,10]. Slice the
active top-left 3x3, one-hot via Equal(Lsmall, arange[1,10,1,1]) -> [1,10,3,3] bool, Cast uint8, Pad to
[1,10,30,30] with 0 (the Pad IS the free `output`). Output declared UINT8 (harness compares >0).

## Irreducible-floor analysis
Dominant: Snum & Sden [1,1,10,10] fp32 = 400B each (Conv inherits fp32 input dtype). The six downstream
f16 [1,1,10,10] working planes (~200B each) total ~1200B. No 30x30 plane materialises because the
30x30 carrier was moved into the FREE padded output. ~2.4KB is near the strided-conv floor for this task.

## OPEN ANGLES (re-attack backlog)
- Fold the f16 chain (SnumH/SdenH/den_safe/colr0/colr/Lf, ~1200B) into fewer ops: e.g. one fused
  reciprocal-mul + Round, or do the divide directly on the fp32 Convs and cast once (~+0.05-0.1).
- The two Convs could in principle be ONE Conv with a [2,10,3,3] weight emitting [1,2,10,10] (800B,
  same bytes) then Slice channels — no byte win, skip.

## INSIGHT (transferable)
⭐ A FIXED regular block partition (size-S tiling of S×S blocks) makes the whole "data-dependent
downsample" collapse to TWO STRIDED CONVS (stride=S, S×S kernel) emitting the [1,1,G,G] block grid
directly — NO colour-index [1,1,30,30] plane, NO selector MatMuls, NO CumSum line detection. Pair a
colour-sum kernel (w[k]=k) with a count kernel (w[k]=1), masking unwanted colours (bg, noise gray) to
0 in BOTH weights; colour = round(sum/count). And move the 30×30 one-hot carrier into the FREE output
by Pad-ing a tiny [1,10,K,K] uint8 one-hot (Pad accepts uint8, rejects bool) — kills the last
full-size plane. This beats the task184 separator-detection MatMul idiom whenever block boundaries
are generator-fixed rather than data-dependent.
