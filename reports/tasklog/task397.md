# task397 — fcc82909

## 2026-06-29 mechanism screen

Rule: for each 2x2 colour box, draw a green 2-wide shadow whose height equals the
number of distinct colours in the box.

Current source score: 17.045628 @ mem 2648 params 200. The graph detects 2x2 boxes
with a compact `TopK`/`Gather`/`ScatterElements` pipeline. The largest tensor is
`cond30` [1,1,30,30] bool = 900 B, used by `Where(cond30, green, input) -> output`.

No rewrite adopted. The obvious attempt to avoid `cond30` by composing in a 10x10
crop first is worse, because it materializes a counted [1,10,10,10] one-hot crop
before padding. Current routing keeps the 10-channel result as the free graph
output and pays only a one-channel full-canvas condition.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2648/17.05; cond30 900B bool=optimal Where(cond,green,FREE input) delta-route (must be bool 30x30), code_f 320B forced-fp32 box detector. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** One redundant `And` removed (6→5; 58→57 nodes). −3B.
**Old→new:** mem 2585→2582, params 200→200.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task397_pre_s10.onnx`; source `public_candidates/bobmyers7186/task397.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
