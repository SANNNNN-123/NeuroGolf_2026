# task317 — ce22a75a

**Rule:** size=3 fixed, so the grid is ALWAYS 9x9 = a 3x3 array of 3x3 blocks. A
subset of blocks are marked by a single GRAY(5) pixel at the block CENTRE
(3r+1, 3c+1). The output fills the ENTIRE 3x3 block of every marked block SOLID
with BLUE(1); unmarked/background cells stay 0. Equivalently:
  ch1_out = 3x3 sum-conv (dilation) of gray channel 5;
  ch0_out = ch0_in − 8-ring(ch5)  [ch0_in is itself the in-grid mask: bg=1
            in-grid, 0 off-grid; the gray center's 8-ring hits its 8 bg cells].
**Current:** 18.20 pts, gen:thbdh6332, mem 0, params 900 (dense Conv[10,10,3,3]).
  ⚠️ CORRECTION: an earlier tasklog claimed "FRESH-RATE 0.00 / real LB ~0". That
  is STALE — the slot now holds the dense-conv gen net, RE-VERIFIED fresh 200/200
  this session (stored networks/task317.onnx). There is NO generalization gap.
**Target tier:** detection/at-floor — the mem-0 single dense conv IS the floor.

## Attempts (this session)
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-plane conv + Pad-ingrid + Equal→free, fp16 planes | B | 11862 | 192 | 15.60 | 200/200 | correct but full planes blow mem; worse |
| 2 | (analytic) single label conv [1,1,30,30] + Equal | B | 3600 | 100 | 16.78 | — | worse than baseline |
| 3 | (analytic) 2-ch conv [1,2,30,30] + Pad→10ch | B | 7200 | 180 | 16.09 | — | worse than baseline |
| 4 | reproduce dense mem-0 conv (sparse weights, exact floor) | floor | 0 | 900 | 18.20 | 200/200 | matches floor, no gain |

(Note: a PRIOR session recorded a 17.53 crop-9x9+Pad-uint8 route. That only beat a
phantom "real 0" — since the stored net actually generalizes at 18.20, the 17.53
route is strictly WORSE and must NOT be adopted. 18.20 is the live floor.)

## Best achieved
18.20 @ mem 0 params 900 — adopted? N (equals public net). Beats prior 18.20? N.

## Irreducible-floor analysis
MEM-0 SINGLE-CONV-AT-FLOOR. Output must be the [1,10,30,30] graph output (all 10
channels scored) to keep mem 0 ⇒ conv weight is [10,10,3,3]=900. Reducing OUTPUT
channels forces a Pad with a ≥7200B intermediate (16.09); reducing INPUT channels
forces a Slice ≥3600B; collapsing to one label plane is ≥3600B fp32 (16.78).
Cross-channel ch5→{ch0,ch1} cannot be a grouped conv (a group spanning indices
0,1,5 needs size ≥6, not a divisor of 10 → group=1 only). Footprint is solid 3x3
(no stride gaps) so dilation cannot shrink k. params=900 irreducible; every
plane-materializing decomposition scores strictly below 18.20.

## OPEN ANGLES (re-attack backlog)
- None structural. No mem-0 ONNX primitive emits all 10 channels with <900 weights
  while doing ch5→{ch0,ch1} cross-channel + ring-subtract. Treat as closed.

## INSIGHT (transferable)
⭐ Dense mem-0 Conv[10,10,k,k] whose output IS the graph output, doing a
cross-channel marker→fill that also emits the SUBTRACTIVE bg channel-0, is at hard
floor: params count ELEMENTS so 900 is irreducible and every plane-based golf
(label/Pad/Slice) is worse because mem-0 beats any 30x30 intermediate. The input's
own channel-0 is the free in-grid mask, but the dense conv already exploits it
(ch0_in center − ring(marker)). Confirms BUILD_PROMPT floor warning → BAIL.
⭐ PROCESS: always RE-VERIFY a tasklog's "does-not-generalize / real 0" claim
against the CURRENTLY stored net before chasing a gap-closer — slots get replaced
and stale fresh-rate claims invent phantom +17 gains. Here the stored net is 200/200.
