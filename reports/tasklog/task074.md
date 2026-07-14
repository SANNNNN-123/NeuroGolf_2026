# task074 — 3631a71a

**Rule:** A pattern is drawn with full dihedral (D4) symmetry about the centre of
the [0, 2*trisize)=[0,32) region (trisize=16): the 8-cell orbit
{(r,c),(c,r),(r,N-1-c),(c,N-1-r),(N-1-r,c),(N-1-c,r),(N-1-r,N-1-c),(N-1-c,N-1-r)}
with N=32 all share one colour. Several axis-aligned maroon(9) rectangles are
stamped over the 30x30 grid, occluding cells; the generator guarantees every
occluded cell still has ≥1 visible (non-maroon) orbit member. Output replaces
each maroon cell with the symmetry-implied colour (non-maroon cells unchanged).
Reconstruction = elementwise max over the 8 D4 pullbacks of (input with
maroon→0): value=Σ_{k≠9} k·input_k; M1=max(v,flipR,flipC,flipRC);
result=max(M1, transpose(M1)) (transpose group = transpose of M1).
**Current:** 15.01 pts, mem ~?, params ?
**Target tier:** A (separable flips/transpose, 0-param geometry; entry plane is the
irreducible fp32 colour-index 3600B floor — true Tier-S mem 0 is impossible since
output colours copy arbitrary input colours).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 32x32 canvas, 9 fp16 planes, uint8 L+Equal | A | 26532 | 40 | 14.81 | — | below P |
| 2 | variadic Max + uint8-slice | A | 21660 | 40 | 15.02 | — | tie |
| 3 | flips land at 30x30 via single per-axis reversed Slice (32-canvas only as pad source) | A | 19148 | 49 | 15.14 | — | over |
| 4 | drop uint8 cast: Equal(res30_fp16, arange_fp16) direct | A | 18248 | 49 | 15.19 | 200/200 | MARGINAL |

## Best achieved
15.186 @ mem 18248 params 49 — adopted? **N (recommend N)**. Beats prior 15.01 by
+0.176 only (< +0.3 threshold). Exact: 267/267 stored, fresh 200/200.

## 2026-06-29 live-frontier refresh

Current live/source is much better than the old fp16 D4 reconstruction:
**15.889480 pts @ mem 9000 params 50**.  It uses a single 1x1 colour-index Conv
with maroon mapped to 0, then all D4 pullbacks are uint8 (`Transpose`, row/col
`Gather`, `Max`) and final `Equal` writes the free bool output.  Mem profile:
`color_f` 3600B fp32 plus six 900B uint8 D4/label planes.

The remaining floor is structural.  The 3600B colour-index plane is needed
because arbitrary non-maroon colours are copied.  The six 900B planes are the
minimal uint8 max-of-D4 chain in the current formulation; switching to bool
one-hot before the final Pad would expand to 10 channels and is worse.  Treat
this as current floor unless a fused D4 gather/max op or sub-byte carrier is
available.

## Irreducible-floor analysis
Entry colour-index plane val32 is fp32 [1,1,30,30] = 3600B (cannot go below; the
10→1 reduction must be fp32 per FLOOR_RESEARCH). val30 fp16 1800B (cast, also the
identity term + Pad input — reused, not waste). v = 32x32 fp16 pad source 2048B
(needed because the symmetry centre is at 15.5, so flips read index 31-i ⇒
source indices up to 31; the canvas must reach width 32). Downstream is the
MINIMAL D4 set: fR,fC,fRC,M1,M1t,res30 = six 30x30 fp16 planes (1800B each).
Total 3600+1800+2048+6·1800 = 18248. The 6 downstream planes are each genuinely
distinct orbit-group results; the transpose trick already halves them (M1 + one
transpose covers all 8 images). No plane is removable without breaking exactness.

## OPEN ANGLES (re-attack backlog)
- Eliminate the 2048B 32-canvas pad source: would need the 3 reversed flip slices
  to read mirror indices 31-i without a width-32 source. Tried per-flip
  Pad-on-30-canvas (slice 28-wide + pad 2) — costs MORE (3 slices+3 pads ≈10.3KB
  vs shared v 2048+3 slices 5.4KB). No cheaper alignment found because the
  generator centre is fixed at 15.5.
- Conv with pads=[0,0,2,2] emitting 32x32 fp32 directly to drop val30: measured
  WORSE (4096 fp32 pad + needs a separate 1800 identity Slice).
- Any sub-3600 entry: blocked — colour copy ⇒ fp32 index plane mandatory.
- Remaining gap to +0.3 is ~2100B with no structural lever left; this is at floor
  for a max-of-D4-pullbacks reconstruction.

## INSIGHT (transferable)
⭐ D4-symmetric FULL-GRID occluded fill (no crop) = max(M1, transpose(M1)) where
M1=max over the 4 axis-flips of the maroon-zeroed value plane — same idiom as
task400 but WITHOUT the cutout-crop (output is the whole grid), so it skips
ArgMax/Gather entirely. ⭐ When the symmetry centre is at a HALF-integer (n+0.5)
in a k-wide grid, a flip maps index i→(2*centre−i); landing the flipped axis at
the in-grid 30 via ONE per-axis reversed Slice (start=2*c, end=2*c−30, step=−1)
keeps flip planes at 30x30 instead of growing to the padded source size — but you
still pay one padded fp16 source (here 2048B) because the slice must READ indices
beyond 29. ⭐ Skip the uint8 cast plane: Equal(result_fp16, arange_fp16) is
integer-exact for colour values 0..8 and emits the bool one-hot straight into the
FREE output (saves the 900B uint8 plane vs cast-then-Equal-uint8).
This task is at floor (+0.18) — confirms the ~3600+N·1800 fp16 reconstruction
ceiling: a 6-downstream-plane D4 fill simply lands ~15.2, just under a 15.0 floor.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-3 opus agent; see agent report in submission_log context). Do not re-attempt without a new mechanism.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k7(cost 9050): 10.4M 패치, 931k pos viols. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
