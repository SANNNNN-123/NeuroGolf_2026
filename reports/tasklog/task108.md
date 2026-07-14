# task108 — 46f33fce

**Rule:** Input is a 10×10 grid (top-left of the 30×30 canvas). Colored pixels live ONLY at odd cells `(2r+1, 2c+1)` for r,c∈[0..4]. Each colour-k pixel becomes a 4×4 block in the 20×20 output: `output[4r:4r+4, 4c:4c+4]=k`. Equivalently for EVERY channel (bg ch0 included): `output[i,j]=input[2*(i//4)+1, 2*(j//4)+1]` for i,j<20, else 0. A fixed 4× block-upscale of the odd-cell 5×5 sublattice, zero-padded to 30×30. (period/scale are CONSTANT, not data-dependent.)

**Current:** 17.50 pts, single `GridSample` (grid [1,30,30,2]=1800 params, mem 0), the documented GRIDSAMPLE-AT-FLOOR.
**Target tier:** S (pure spatial copy/upscale → mem 0 in principle) — but a single op cheap enough is the obstacle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice odd cells [1,10,7,7] + depthwise ConvTranspose s4 K4 → output | S | 1960 | 168 | 17.34 | — | fp32 slice plane dominates, <P |
| 2 | Slice [1,10,6,6] + depthwise ConvTranspose s4 K7 → output | S | 1440 | 498 | 17.43 | — | best slice/kernel tradeoff, still <P |
| 3 | fp16-cast the slice | S | 2940 | 168 | 16.96 | — | cast ADDS a plane (fp32 slice + fp16 cast both counted) |
| 4 | **group=1 ConvTranspose [10,10,4,4] on FULL input, ch0 reads+subtracts colours** | S | 0 | 1600 | **17.62** | **200/200** | **beats P by +0.12 (MARGINAL)** |
| 5 | sparse_initializer for the weight (nnz=304) | — | — | — | — | — | ORT load fails: harness `sanitize_model` does NOT remap sparse-initializer names → node input unresolved. Sparse inits unusable. |

## Best achieved
17.62 @ mem 0 params 1600 — adopted? N (build prompt: do not adopt). Beats prior 17.50? Y but only +0.12 → **MARGINAL** (gate wants +0.3).

## Irreducible-floor analysis
Colours (ch1-9) upscale EXACTLY for free with a depthwise stride-2 K-4 ones ConvTranspose written straight into `output` (mem 0, 160 params): even input cells are 0 in colour channels so even/odd tap collisions cancel. The ONLY obstacle is **channel 0**: bg=1 at EVERY even input cell, and a single depthwise kernel CANNOT remove the leak (the odd-cell tap and the even-cell tap land on the SAME kernel position → contradictory weights). Fix requires ch0 to read the colour channels and subtract them → a group=1 dense weight `[C_in=10, C_out=10, 4, 4]`. Only 19 of the 100 in/out pairs are nonzero (ch0-self, 9 colour-self, 9 colour→ch0 subtractions) but **dense storage counts all 1600 elements**. The 81 zero pairs (1296 wasted params) cannot be dropped: kh,kw are shared across all pairs (so can't shrink below 4×4 = the block size), channel counts are fixed at 10, and sparse initializers are broken by the harness sanitizer (attempt 5). 1600 params → 25−ln(1600)=17.62 is the single-op ceiling.

## OPEN ANGLES (re-attack backlog)
- Any future ORT/harness build that remaps sparse_initializer names would make attempt-5 (nnz≈304 → ~19.3) viable instantly — re-test sparse weights if the sanitizer changes.
- A hypothetical op taking TWO 1-D index tables (separable remap) would give params≈60, mem 0 — none exists in opset-10/11 (Gather is per-axis → forces a full intermediate plane; GatherND/GridSample need a full [30,30,2]=1800 index/grid).
- Two-op Concat(ch0_plane, colours_plane)→output: blocked by the 32400B colour intermediate plane.

## INSIGHT (transferable)
⭐ A FIXED block-upscale of an odd-cell sublattice is a single depthwise `ConvTranspose` (stride 2, pad_top=2 to align odd cell a=2r+1 → block [4r..4r+3], asymmetric `pads=[2,2,30,30]` to crop the 58×58 back to top-left 30×30) — **mem 0, writes `output` directly, colour channels EXACT for free**. The catch that defeats a pure depthwise upscale of one-hot grids: **channel 0 (background) is 1 at the even cells**, and even/odd cells collide on the same ConvTranspose kernel tap, so ch0 leaks into colour cells. Fix = a group=1 weight where ch0 also reads and SUBTRACTS the colour channels (`W[k,0]=−BIG`), pushing ch0 negative wherever a colour sits; after (out>0) it's exact. Cost is the dense [10,10,K,K] (here 1600) since the 81 zero pairs are still counted. ⭐ Sparse `graph.sparse_initializer` is UNUSABLE with this harness: `sanitize_model` renames only dense `graph.initializer`, leaving sparse-init names un-remapped → ORT "input is not a graph input/initializer" load error. Do not chase sparse-weight param savings.

## S9 (2026-07-03) — kojimar 7184.85 teacher: separable-remap Einsum (+1.175) ADOPTED
Single Einsum 'ra,ai,zcij,bj,sb->zcrs' with U(30,5) row-block table + S(5,30)
latent→odd-col table; out = P·in·Pᵀ per channel (P=U@S) = 4× block-upscale of the
odd-cell 5×5 sublattice, zero-padded past 19. mem=0 (single node writes free output),
params=300. REFUTES this tasklog's old "no separable-remap op" OPEN-ANGLE floor.
Gates: stored fail=0 (my re-check too); fresh 2500 uncached: teacher 0, incumbent 0,
div 0 (bit-identical). No TopK. Source: overrides/task108.onnx (base_submission was
our own mechanism). Backup reports/retired_networks/task108_pre_s9.onnx.
⭐ TRANSFERABLE: fixed separable spatial remap/upscale = one 5-operand einsum, mem 0.
