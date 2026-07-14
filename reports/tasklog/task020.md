# task020 — 11852cab

**Rule:** Complete a symmetric "blast" of 4 concentric diamond rings around a center: ring0=center
(colors[0]); ring1=4 diagonal neighbors (colors[1]); ring2=4 axis cells at dist2 (colors[2]); ring3=4
diagonal cells at dist2 (colors[3]). INPUT shows all rings full EXCEPT one ring which keeps only one cell;
OUTPUT completes all rings. Task = detect center + 4 ring colors, re-stamp the full fixed pattern.
**Current:** 15.71 pts (custom:task020, adopted from gen:vyank6322 14.14), mem 10708, params 160
**Target tier:** B (label-map). Center+colors detection is non-local-ish but bounded; floor ~3600.
10708 leaves headroom.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | center via stamp-correlation==10, ring color via max-over-ring, restamp uint8 L on 10×10 crop → Pad → Equal | B-ish | 10708 | 160 | 15.71 | 200/200 (+1500/1500) | ADOPTED (+1.57) |
| 2 | RE-ATTACK: 8×8 crop (center∈[3,6]²), channels 1..8 only, drop presW/onesW convs (pres=V>0), fused Sum | B | 5422 | 145 | 16.38 | 200/200 (+1500/1500) | ADOPTED (+0.67) |

## Best achieved
16.38 @ mem 5422 params 145 — adopted? **Y**. Cumulative 14.14→16.38 (+2.24).

## Irreducible-floor analysis (after attempt 2)
At/near floor. mem_profile: **2048B fp32 [1,8,8,8] crop** dominates + 900B uint8 L[1,1,30,30] (feeds free
Equal). The crop is irreducible-ish: an 8-channel small tensor needs either an fp32 crop (2048) or casting
full input to small dtype first (≥9000B cast) — Pad can't retype, so the crop is the cheapest gateway. The
900B L is minimal (30×30 needed so Equal writes straight into FREE output incl. selective bg channel0).

## OPEN ANGLES (mostly exhausted — diminishing)
- split crop into channels{1,2,3,4}+separate color-8 grab (<2048) — needs 2 Pads, likely net-neutral.
- 30×30 uint8 V via Conv→Cast then crop — rejected, 3600 fp32 conv output > 2048 crop.

## INSIGHT (transferable)
⭐⭐ **Geometry bounds beat dtype tricks.** The highest-leverage move on a size-bounded detection task:
read the generator's coordinate bounds and shrink the working canvas to the TRUE active region (center∈[3,6]²
+radius2 ⇒ 8×8, not 10×10 or 30×30), AND crop out unused color channels (colors∈{1,2,3,4,8} ⇒ channels 1..8).
An 8×8×8 fp32 crop (2048) beats any fp16-via-full-cast path because Pad cannot retype and the full-size cast
intermediate dominates. Also: derive masks from each other (pres=(V>0); in-grid via 0-pad border) to DELETE
whole channel-reduction convs, not just downcast them. → promote to project memory.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.126)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 1346 (mem 1176 + params 170), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: int64->int32 recast on counted index tensors (cell_base, keep_pos_by_sum).
