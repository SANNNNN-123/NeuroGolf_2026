# task225 — 93b581b8

**Rule:** Grid is ALWAYS 6x6. A single 2x2 block of four DISTINCT random colours
sits at top-left (row,col), row/col in 1..3: cells (row,col)=c0, (row,col+1)=c1,
(row+1,col)=c2, (row+1,col+1)=c3. The input holds only that block. The output
keeps the block and adds a 2x2 monochrome stamp of each colour at a diagonal
corner offset (clipped to grid): c0→(row+2,col+2), c1→(row+2,col-2),
c2→(row-2,col+2), c3→(row-2,col-2). Block + 4 stamps are pairwise disjoint
=> output = block + stamps. All painted cells lie inside the 6x6 grid.
**Current (public):** 15.69 pts (gen:thbdh6332). Prior custom attempt 14.03 (REJECTED).
**Target tier:** B (label-map + Equal) — colours are random per-instance so a fixed
Conv can't route them (Tier S blocked); the 4 stamps + block carry 4 different
scalar colours over separable rectangles, not a single row⊗col overlay, so it is
a multi-colour label map, not Tier A separable one-hot.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30x30 reductions + 6x6 label, fp32 pad | B | 19752 | 119 | 15.10 | 200/200 | below P |
| 2 | Slice input→[1,10,6,6], 6x6 work, fp32 pad | B | 12408 | 65 | 15.57 | 200/200 | below P |
| 3 | + uint8 pad (900 not 3600) + factored rank-1 label | B | 6828 | 65 | 16.16 | 200/200 | WIN +0.47 |

## Best achieved
16.16 @ mem 6828 params 65 — adopted? pending main (NOT adopted by build agent).
Beats prior public 15.69? **Y (+0.47, clears +0.3)**. 800/800 stored-style + 200/200
isolated fresh (genverify fresh_pass logic, true generator).

## Irreducible-floor analysis
Dominant intermediates: Slice inp6 [1,10,6,6] fp32 = 1440B, padded uint8 label
[1,1,30,30] = 900B, then 8× [1,10,6,1]/[1,10,1,6] per-channel reductions = 240B
each. The 1440 colour-read is the floor for recovering 4 random colours by
position; the 900 is the standard uint8 label plane. Both are real.

## OPEN ANGLES (further squeeze, ~+0.1-0.15 only — low ROI)
- Slice channels 1..9 only ([1,9,6,6]=1296) and drop CHM/rowKc/colKc (saves ~620B)
  by shifting the colour index by +1; would land ~16.3. Skipped: sub-0.3 polish,
  adds indexing risk to a clean +0.47 win.
- Build the label colvecs in uint8 directly to avoid the float→uint8 Cast chain.

## INSIGHT (transferable)
⭐ Two compounding levers turned a near-floor "rejected" task into a clean win:
(1) GRID IS FIXED 6x6 → Slice the free input to the active window FIRST, so every
per-channel reduction is [1,10,6,1] (240B) instead of [1,10,30,1] (1200B). (2) PAD
IN uint8, NOT fp32 — Cast the small 6x6 label to uint8 BEFORE Pad so the 30x30
sentinel plane is 900B, not 3600B (ORT Pad accepts uint8; it only rejects bool).
Also: a multi-colour label over disjoint separable rectangles factors as a SUM of
rank-1 (rowmask ⊗ colvec) terms — group regions by their row band so only ~4
[1,1,6,6] planes are materialised instead of one per region. mem_profile.py
mis-reports shapes for opset-11 broadcast graphs (showed a phantom 36000 Mul / 3600
Conv); trust harness.evaluate's number, which uses the real ORT trace.
