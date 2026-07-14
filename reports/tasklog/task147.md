# task147 — 67385a82

**Rule:** Input = green (colour 3) pixels scattered on a small (W,H in 3..6) black
grid, top-left anchored on the 30×30 canvas. Per cell: a green pixel with >=1
orthogonal green neighbour → cyan(8); an isolated green pixel (0 orthogonal
green neighbours) → green(3); in-grid background stays background(0); off-grid
stays all-zero. A GENUINE orthogonal-4-neighbourhood per-pixel recolor (output
colour of a green cell depends on its neighbours), NOT a 1×1 recolor.

**Current:** 18.187 pts, generic conv3×3+b, mem 0, params 910 (verified:
networks/task147.onnx is a single Conv, mem 0, 910).
**Target tier:** detection/at-floor — minimal encoding is one dense 3×3 Conv→output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | multi-plane slice+plus-Conv+Equal route (prior agent) | B | 31500 | 33 | 14.64 | — | worse |
| 2 | ANALYTIC single dense 3×3 Conv→output (== current) | — | 0 | 910 | 18.187 | 200/200 | ties floor |
| 3 | CROP-TO-ACTIVE decomposition: Slice ch3+ch0 to 6×6, plus-Conv nbr count, build {bg,gout,cyan} u8 6×6 blocks, Concat→9ch block, Pad to 30×30 (free output) | B+ | 1044 | 70 | **17.98** | **200/200** | correct & general but BELOW floor |
| 4 | grouped Conv group=2 ([10,5,3,3]=460) | — | — | — | — | — | INFEASIBLE: cyan out-ch8 ← green in-ch3 spans channels 3..8 (6 wide); 6 ∤ 10 (group sizes 1/2/5/10), no equal group co-locates in-3 with out-8 |
| 5 | sparse_initializer Conv weight (~16 nonzeros → ~16 params) | — | — | — | — | — | INFEASIBLE: ONNX checker rejects Conv with sparse weight type ("W typestr T has unsupported type sparse_tensor"); harness calculate_memory REQUIRES check_model(full_check=True) → "could not be measured". (ORT itself runs it, but the static checker blocks it; routing sparse→Cast→dense fails shape inference: Inferred=8 Declared=1.) |

## Best achieved
17.98 @ mem 1044 params 70 (attempt 3) — fresh 200/200, but BELOW the 18.187
single-conv floor. NOT adopted.

## Irreducible-floor analysis (decomposition route, attempt 3)
The crop-to-active decomposition is CORRECT and general but cannot clear +0.3.
Structural mem floor, even with ZERO compute planes:
- TWO f32 slices [1,1,6,6] (green ch3 + bg ch0) = 144+144 = **288 B** (Slice
  preserves the f32 input dtype; both are required — green for the conv,
  ch0 is the ONLY in-grid-extent signal so bg/off-grid can be distinguished).
- ONE 9-channel uint8 carrier block [1,9,6,6] = **324 B** (output colours sit at
  channels {0,3,8}; the carrier must be ≥9 deep; a single shared zero-block init
  fills the gap channels so it costs only 36 params, not 216).
- ⇒ slices+block = 612 B; with ~50 B params → score 18.50 (only +0.31) WITH ZERO
  compute. But the unavoidable compute (plus-Conv + cyan/gout decisions, ~432 B
  even all-fp16/bool/uint8) drags mem to 1044 → **17.98**.
- Alternatives tried & rejected: single [1,4,6,6] slice (576) + cast (1692 total,
  worse); index-plane L→Pad(sentinel)→Equal needs a 1×1 30×30 plane (1800 B,
  worse than the 324 block); Gather-channel-remap of a 4-ch block emits a
  [1,10,6,6] (360 B, worse than the Concat block).

## Verdict: INFEASIBLE (cannot beat 18.187 by ≥+0.3)
Three independent ceilings, all confirmed empirically this session:
1. **Grouped-conv shrink blocked** — green in-ch3 must couple to cyan out-ch8;
   that span (3..8 = 6 channels) exceeds any equal group dividing 10.
2. **Sparse-init shrink blocked** — ONNX checker (which the harness runs in
   calculate_memory) rejects a sparse Conv weight type; can't route it dense.
3. **Decomposition capped** — slices(288)+carrier(324)+forced compute(~432)
   ≈ 1044 → 17.98, strictly below the 18.187 dense-conv floor.

## INSIGHT (transferable)
⭐ The CROP-TO-ACTIVE + Pad-into-free-output lever (task227/387/143) does NOT
beat a mem-0 single-dense-Conv floor when the output needs ≥3 colour channels at
SPREAD indices (here {0,3,8}). The 9-deep uint8 carrier block (≥ ~9·36 = 324 B)
PLUS two forced f32 6×6 slices (288 B) already eat the entire budget before any
compute, so the decomposition lands ~17.98 < the 18.19 single-conv floor. The
decisive pre-test: count (a) the max output channel index that must carry a
colour (carrier depth) and (b) the number of distinct f32 input-channel slices
needed; if depth·36 + 144·#slices > e^(25−(P+0.3)), the crop route is dead before
you write it. Here depth 9 + 2 slices ⇒ 612 B floor ⇒ best-case 18.50 with zero
compute, infeasible once real compute is added.
⭐ NEW negative result: a sparse_initializer can in principle make a near-floor
dense Conv almost free (params count only nonzeros), and ORT runs it, BUT the
harness's calculate_memory calls onnx.checker.check_model(full_check=True) which
rejects Conv's sparse weight type — so the sparse-init shrink is permanently
blocked for any Conv-weight task under this harness.
⚠️ TOOLING NOTE: `/tmp/arc-gen/tasks/task_1478ab18.py` is a DIFFERENT task than
task147 — the correct generator is task_67385a82.py (per reports/arc_mapping.json
"147"→"67385a82"). Always resolve the generator via arc_mapping.json, not by
guessing the hash filename, before running isolated fresh.
