# task181 — 760b3cac

**Rule:** 6×9 active grid. A 3-wide cyan (8) "conway" sprite sits in cols 3..5,
rows 0..2; a FIXED yellow (4) shape sits in rows 3..5, cols 3..5. A `flip` flag
horizontally mirrors the whole 9-wide grid (applied to input AND output). OUTPUT
= INPUT plus a horizontally-reflected copy of the cyan sprite on the OPPOSITE
outer side: flip=0 → reflected block in cols 0..2 (out col m ← in col 5−m);
flip=1 → cols 6..8 (out col m ← in col 11−m). Yellow unchanged. flip recoverable
from yellow: at (3,3)⇔flip0, at (3,5)⇔flip1. Cyan always at input cols 3..5.
**Current:** 17.91 pts, ext:kojimar6275, mem 1108, params 86
**Target tier:** B (one-hot relabel routed into the FREE output) — the public net
already uses the 6×9-active-grid + final-Pad-to-30 idiom; cannot reach Tier S/A.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30×30 Gather(cyan) + Where(refl, cyan1hot, input) | B | 9246 | 81 | 15.86 | — | two fp32 30×30 slices kill it |
| 2 | 6×9 active grid, Equal→[1,10,6,9] bool one-hot, bool Pad→output (opset13) | B | 1014 | 70 | 18.01 | 200/200 | beats +0.10 (MARGINAL) |
| 3 | trim one-hot to 9 channels ([1,9,6,9], pad ch9) | B | 960 | 69 | 18.064 | 200/200 | beats +0.15 (MARGINAL) |

## Best achieved
18.064 @ mem 960 params 69 — adopted? N (MARGINAL, < +0.3). Beats prior 17.91? +0.15 only.

## Irreducible-floor analysis
Dominant intermediate = the `onehot6x9` route-into-output plane: [1,9,6,9] bool =
486B (10 channels = 540B before the unused-ch9 trim). This is structural: every
output cell of the 6×9 active region must carry a one-hot across the colour axis,
and the active region is genuinely 6 rows × 9 cols (cyan reflection reaches col 0
AND col 8; yellow fills rows 3-5). 9×6×9 = 486 is the floor. The remaining ~474B
is two 108B fp32 channel-Slices (cyan ch8 / yellow ch4) + a handful of 27B bool
masks + the runtime flip-selected colidx. The ALTERNATIVE route — `Where(refl_full
[1,1,30,30] bool, cyan1hot, input)` (add cyan, copy rest, no yellow/bg rebuild) —
costs a 900B 30×30 bool condition, strictly worse than 486B. Both routes floor
~18.0–18.08; the public net sits at the same structural point (1108). +0.3 (≥18.21)
is unreachable for this relabel.

## OPEN ANGLES (re-attack backlog)
- Pure Tier-S Gather of input along axis=3: blocked — the cyan channel needs an OR
  of two source columns (original + reflected), which a single Gather can't express.
- Sub-486B one-hot: would need <10-channel output OR <54-cell active region; both
  fixed by the rule (channels land at indices 0/4/8; active region is 6×9).
- Eliminating one fp32 slice via a single 1×1 Conv colour-collapse: forces a 3600B
  full-canvas plane — strictly worse than the two 108B active-region slices.

## INSIGHT (transferable)
- ⭐ bool `Pad` (one-hot → 30×30 output) is REJECTED under opset 11 ("Pad has
  unsupported type tensor(bool)") but ACCEPTED under **opset 13** — Pad-13 added
  bool support. For one-hot-relabel tasks that pad a small active-region one-hot
  into the FREE output, declare opset 13 and keep the one-hot bool (itemsize 1)
  instead of Cast→uint8 (which adds a full 540B duplicate plane).
- ⭐ Trim the one-hot to only the USED colour channels (here 0..8, dropping the
  never-used maroon ch9) and recover the dropped channel(s) as zero in the SAME
  final Pad (pad +k on the channel axis end) — free ~10%/channel mem cut on any
  relabel whose palette excludes the top channels.
- The 6×9-active-grid + final-Pad idiom is already near its structural floor; a
  one-hot relabel cannot beat ~18.0 once the active region and palette are fixed.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.003)
**Mechanism (op-census diff):** One fewer flag index (`flag_index` [1,4]→[1,3]). −1 param.
**Old→new:** mem 201→201, params 168→167.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task181_pre_s10.onnx`; source `public_candidates/bobmyers7186/task181.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
