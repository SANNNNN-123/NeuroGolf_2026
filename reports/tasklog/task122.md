# task122 — 5168d44c

**Rule:** A green(3) "track" is a line of dots spaced every 2 cells along one axis
(horizontal row, or vertical column when xpose=1). A red(2) 3x3 box is centred on
one green dot. OUTPUT = input with the green track UNCHANGED and the red box
translated by +2 cells ALONG the track direction (right if horizontal, down if
vertical); vacated cells reset to bg/green, green wins over red on the box centre
row. Verified closed form: green channel identical in==out; red channel out =
shift(red_in, +2 along track axis); orientation horizontal <=> max-per-row-green >
max-per-col-green (track axis carries >=3 dots, perpendicular <=1 per line).
**Current:** 17.17 pts, custom:task122 (single Conv5x5+b, mem 0), params 2510.
Prior stored 17.17 (public conv5x5+b); prompt P was stale at 16.84.
**Target tier:** S (single Conv writes [1,10,30,30] output directly, NO intermediate).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | closed-form label map: slice red/green, orient-select +2 shift, L->Equal | B | 34929 | 37 | 14.54 | 200/200 | exact but full 30x30 fp32 planes -> well below P |
| 2 | crop-to-21 + fp16 downstream estimate | B | ~6600 | 37 | ~16.1 | — | still below P; >=3 full ~21x21 planes |
| 3 | single Conv5x5+b (10,10,5,5)+B, verified-exact int weights, mem 0 | S | 0 | 2510 | **17.17** | **200/200** | adopted candidate |

## Best achieved
**17.17 @ mem 0, params 2510 — isolated fresh 200/200.** Beats prompt P=16.84 by
**+0.33**. Ties the existing stored conv (same structure is the param floor).

## Irreducible-floor analysis
At the pure-conv param floor. The map is fully LOCAL (each output cell decided by a
5x5 input window): the +2 red shift needs reach 2 on BOTH axes (orientation chosen
locally from the green dots in-window), forcing a 5x5 kernel; the output value_info
is fixed [1,10,30,30] so the conv must read all 10 in-channels and write all 10
out-channels. params = 10*10*5*5 + 10 = 2510 (element count, NOT nonzeros — only
in/out channels {0,2,3} and 181/2500 taps are actually nonzero, but sparsity does
not lower the count). mem = 0 because the Conv writes `output` directly with no
intermediate tensor. Any closed-form alternative pays >=1 full-canvas plane
(>=1764B fp32 at 21x21, ~3528B for the two colour channels) so lands ~16.0-16.5,
strictly below this conv.

## OPEN ANGLES (re-attack backlog)
- Cut params below 2510 only if the 7 dead in/out channels could be dropped without
  an intermediate. Slicing input to channels {0,2,3} costs a [1,3,30,30]=10.8KB
  plane (far worse than mem=0). No op restricts conv input channels for free.
- A 3x3 conv (910 params) cannot reach +2 -> insufficient. No sub-5x5 path.
- Dropping bias (10 params) is negligible and breaks the bg/red `>0` thresholds.

## INSIGHT (transferable)
⭐ A LOCAL shift-of-a-stamp-along-a-track (small bounded displacement, orientation
read from the local track pattern) collapses to a SINGLE Conv-into-output (Tier S,
mem 0) — the orientation choice need not be a global scalar; a 5x5 linear filter +
`>0` threshold resolves "which way is the track here" per cell. When mem=0 is on the
table, a learned/verified integer conv BEATS every closed-form label-map (which
must pay >=1 full-canvas fp32 plane). params = element count of [out,in,kh,kw], so
channel/tap sparsity is worthless for scoring — the kernel reach (here +2 -> 5x5)
and the fixed 10-channel I/O set the irreducible floor.
