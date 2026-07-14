# task222 — 91714a58

**Rule:** 16×16 grid = random single-pixel colour noise (density 0.5) + ONE SOLID
axis-aligned rectangle ("box") of a single colour, width,height ∈ [2,8], area ∈ [9,16],
placed interior (box cells in rows/cols 1..14). The generator guarantees box-colour noise
has NO same-colour 4-neighbour (isolated) and never extends the box. OUTPUT = input
restricted to the box rectangle; every other in-grid cell → background (ch0); off-grid
(rows/cols 16..29) → all-zero. Closed-form: keep(r,c) iff (r,c) is in a fully-filled
single-non-zero-colour window of shape 3×3 OR 2×5 OR 5×2 (every valid box contains and is
tiled by such windows; ≥9-cell uniform windows essentially never occur in 50%-density
noise — measured 0/20000; the smaller 2×3/3×2 set leaks ~3/20000). The box = union of
those windows.

**Current (deployed kojimar):** 15.62 pts, per-channel QLinearConv patch-score + ArgMax-
colour + rect-conv, mem 11451, params 444 (bottleneck: a 7056B fp32 `active_nonzero`
[1,9,14,14] slice).
**Target tier:** A (window-fill detection routed into the free Where output; no global argmax).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior fp16 3-shape (3×3/2×4/4×2), Where(bg) out | A | 12466 | 72 | 15.56 | 0/3000 | exact but BELOW deployed |
| 2 | + uint8 dilation/Max (opset 13) | A | 11056 | 72 | 15.68 | 0/3000 | beats deployed |
| 3 | + BOOL Pad (opset 13) → one 30×30 plane | A | 10352 | 72 | 15.75 | — | |
| 4 | + all-uint8 uniformity via QLinearConv (S, k·mx) | A | 8866 | 108 | 15.90 | 0/3000 | |
| 5 | + shared scalar quant params, dedup weights, {3×3,2×5,5×2} | A | 8736 | 78 | **15.916** | 0/3000 | **best** |

## Best achieved
**15.9159** @ mem 8736 params 78 (mem+par 8814) — adopted? pending. Beats deployed 15.6161
by **+0.2998** (rounds to +0.30; vs the prompt's stated 15.62 that is +0.296). Exact:
266/266 stored, 0/3000 fresh. **~2 bytes short of the strict +0.3-over-15.92 bar.**

## Irreducible-floor analysis
Dominant intermediates (mem 8736): colf30 [1,1,30,30] fp32 = **3600B** (the 10→1 colour-index
collapse; Conv inherits the fp32 input dtype — cannot be narrowed, and any alternative
collapse is ≥3600B); keep_b [1,1,30,30] bool = **900B** (the Where cond MUST be a full 30×30
bool — the box is a rectangle but off-grid needs OR-semantics that break the separable
row⊗col association, so one 30×30 plane is forced); colf fp32 slice **784B** (Slice inherits
fp32, can't be skipped). Everything else is uint8/bool at the 14×14 active region. The
uniformity test needs S (sum) AND k·mx — both materialised; uint8 Add/Sub/Mul and uint8
Gather-indices are all ORT-rejected, so the integer `S == k·mx` (two QLinearConvs) is the
only exact uint8 form; min/max-complement and Cauchy-Schwarz alternatives cost the same or
more planes.

## OPEN ANGLES (re-attack backlog)
- **The last ~2 bytes:** removing the 3 kmx QLinearConv planes (~390B) would clear +0.3 with
  margin, but no ORT-legal uint8 op computes `k·mx` (or an equivalent uniformity test)
  without a per-shape plane. Re-probe if a future ORT build adds uint8 Add/Mul or ReduceMin.
- Eliminate the 784B fp32 `colf` slice if a future op can Slice-and-Cast in one node.
- Collapse keep_b (900B) to separable row⊗col masks IF the off-grid OR-semantics can be
  folded into the free bool output without a full plane (currently blocked by the AND/OR
  mismatch between box-rect and off-grid).

## INSIGHT (transferable)
⭐ **uint8 pipeline at OPSET 13 unlocks a half-footprint floor that opset 11 hides.** uint8
`MaxPool`, `Max`, `Greater`, `Equal`, `ReduceMax`, **BOOL `Pad`**, and **`QLinearConv` (exact
integer conv: all scales=1, zero-points=0, output ≤255)** ALL run under ORT_DISABLE_ALL at
opset **13** but FAIL at opset 11. A window-sum becomes an exact uint8 QLinearConv and `k·mx`
a 1×1 uint8 QLinearConv, so an integer `S == k·mx` uniformity test runs with zero fp planes —
half the fp16 cost. (Blocked even at opset 13: uint8 `Add`/`Sub`/`Mul`, and `Gather` with
uint8 indices.) ⭐ Scalar (shape `[]`) QLinearConv scale/zero-point inputs are accepted and
SHAREABLE across many convs → quant params cost ~3 elements total. This is the kojimar
QLinearConv idiom generalised — and it beats the deployed kojimar net because collapsing
10→1 colour index FIRST (3600B) is far cheaper than its per-channel 7056B fp32 slice.

## S9 (2026-07-03) — single-tap 17×17 valid-Conv crop (+0.086) ADOPTED
Entry Conv1x1→colf30 3600B + Slice → ONE [1,10,17,17] valid Conv, pack weights at tap
(1,1), emits box interior 14×14 direct. mem 8736→5136, params 78→2952 (dense kernel —
params count zeros, sparse banned; caps the win at +0.086 not +0.3). Bit-identical
2500+600 uncached 0/0/0, no TopK. Trade curve checked: s=14 optimal (u8 downstream
scales s²). Floors: keep_b 900 bool (off-grid OR), colf 784. Backup task222_pre_s9.onnx.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k5(cost 8088): hinge unsolved, val 100% fail. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).

## S16 (2026-07-06) — llccqq624 public net: dropped redundant Slice + its counted plane (+0.158) ADOPTED
Engine-proof loop (match→mine→fresh-gate→adopt). Source = llccqq624 "verified circuit union"
(7235.83 dump). Grader: mem 5992→5096, params 211→202 → cost 6203→5298, points 16.267→16.425.
Fresh gate: fresh_verify 222 1500 = incumbent fail 0, candidate fail 0, **candidate != incumbent = 0**
(bit-identical on all 1500). Structural diff vs old S9 net: dropped 1 `Slice` node (21→20 nodes) →
eliminates its ~900B counted output plane; everything else identical (1 Conv, 6 MaxPool, Pad×2…).
⭐ TRANSFERABLE: safe-golf "redundant post-crop Slice whose window is already produced by the valid-Conv
tap can be dropped — the Conv already emits the interior; the Slice was re-cropping an already-cropped
plane." Same pattern to scan: any of our valid-Conv-crop nets (196/193/396) carrying a trailing Slice.
Backup: scratchpad/backup/task222.onnx.bak. Bit-identical ⇒ private-LB safe.
