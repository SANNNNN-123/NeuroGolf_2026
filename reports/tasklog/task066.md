# Task066

## Mechanism

Generator `2dd70a9a` is a marker-routed hidden path task.

- Input preserves red endpoint pair `2`, green endpoint pair `3`, and cyan marker/noise `8`.
- Hidden blue path cells are black in input and must become green in output.
- In canonical orientation the red pair is on the right side and the green pair is on the left side.
- Two families exist:
  - `S`: right vertical from red pair, horizontal bridge, left vertical to green pair; deliberate cyan turn markers at `(mid-1,left)` and `(mid,right+1)`.
  - `U`: right vertical down from red pair, bottom bridge, left vertical up to green pair; deliberate cyan markers at `(bottom+1,left)`, `(bottom,right+1)`, plus anti-side-entry marker at `(red_top,right-1)`.
- Flip/transpose produce four effective orientations.

## 2026-06-28 semantic probe

Python solver using dihedral normalization, endpoint pairs, zero-only path painting, and marker checks:

- stored validation: reached 3/4 initially; after duplicate-corner fix reached 4/4 before strict generator-range filters.
- fresh random: about 99.7%+ exact before final tie-break work.
- main failure mode: random cyan noise can accidentally satisfy the wrong U/S marker pattern, so marker checks alone are insufficient.
- useful tie-break: include generator parameter constraints and candidate-family priority, but validation examples are not perfectly captured by strict random-range filters, so range checks must be soft rather than hard.

## Current graph bottleneck

`task066` current exact-preserve graph is already a semantic-ish path scorer, but expensive:

- 301 nodes, roughly 19 KB profiled intermediates.
- top memory contributors are full `20x20` input slices/reductions (`Slice` channel plane, row/col `ReduceMax`) and boolean path masks.
- Simple dtype conversion cannot produce a large win; the real win requires compiling only a small set of S/U candidate masks and avoiding broad scan-style preserved graph machinery.

## Reusable insight

Register as `marker_routed_hidden_path_compiler`: solve endpoint-pair hidden path tasks by generating a bounded set of candidate path masks from endpoints and deliberate marker pixels, then choose with soft generator constraints to suppress random-noise false markers.

## Open angle

Build a source-owned ONNX compiler for the canonical S/U families:

1. Reduce to four generator orientations, not full arbitrary graph replay.
2. Extract endpoint row/col scalars from row/col presence.
3. Build S and U masks from row/col comparisons.
4. Validate marker pixels with `GatherElements`.
5. Use soft legality/tie-break instead of strict range rejection.
6. Emit a one-channel uint8 label plane and final output, avoiding early 10-channel expansion.

## 2026-06-29 verified tail Pad compression

Small source-owned compression adopted.

- Previous source/live: `points=15.179079`, `memory=17763`, `params=652`, stored `266/266`.
- Replaced two zero-mask concatenations:
  - `Concat(hidden4, hidden_right) -> hidden_wide`
  - `Concat(hidden_wide, hidden_bottom) -> hidden30`
- with one constant-zero `Pad(hidden4, hidden_pad30) -> hidden30`.
- Removed the unused zero initializers `hidden_right` and `hidden_bottom`.
- New source/live eval: `points=15.240210`, `memory=17163`, `params=160`, stored `266/266`.
- Fresh side-by-side against the previous live graph: `2000` eligible examples, output divergence `0`; both graphs shared the same rare task-level failures (`7/2000`), so this patch is semantics-preserving relative to the incumbent.

Insight: when a bool active mask is extended to the harness canvas only by
concatenating zero blocks, `Pad` can remove an intermediate mask plane and large
zero initializers. This is a mechanical compression, not a semantic path-solver
breakthrough.

## 2026-06-30 (S2) — int64→int32 gather-index micro-golf (LANDED)

Output-preserving safe-golf (playbook §3.7). The three `cyan_norm` GatherElements
indices `n_{red_cyan,green_cyan,marker_out}_gather_i` were int64 `[1,20,1]`
(160B each). They are built as `Add(n_red_cyan_row_zero[i64 zeros], col_i3)` where
`col_i = Cast(n_*_max)`. Flipped the three `Cast …col_i` targets int64→int32 and the
shared `n_red_cyan_row_zero` initializer to int32 zeros (params unchanged — element
count identical). All nine affected value_infos retyped 7→6. GatherElements accepts
int32 indices; the index values are colour offsets (bounded, never overflow int32).

- Before: `memory=17163`, `params=160`, `points=15.240210`.
- After:  `memory=16899`, `params=160`, `points=15.255606` (−264B, +0.0154).
- Gate: bundled 266/266 identical; **fresh-equivalence old-vs-new = 0 divergences
  over 1500 arc-gen instances** (generator at /tmp/arc-gen). The incumbent's own
  ~0.3% fresh-failure gap is pre-existing and unchanged (this patch is bit-identical).

## 2026-07-01 big-mechanism re-attack

Rechecked this as the main remaining high-cost task001-style candidate in
016..100.  The current graph still has real mass (`cyan_4d` 1600B, row/col
profiles 1200B each, `hidden30` 900B, plus many 20x20 bool masks), so a semantic
family compiler would be a meaningful win if it could replace the scorer.

Python oracle result:

- A four-family candidate set `{S_down,S_up,U_down,U_up}` with the observed red
  and green endpoint pairs covers the generator: over 5000 fresh instances,
  exactly one candidate mask equals the true hidden path in every case.
- Marker predicates alone are not enough.  Random cyan noise creates many false
  candidates: in 5000 fresh instances the number of marker-valid candidates
  ranged from 1 up to 14.
- Simple tie-breaks fail:
  - longest hidden mask: 803/10000 fresh failures;
  - longest full geometric path: 184/20000 failures after cyan-clear filtering;
  - cyan-clear + family preference: 147/20000 failures.
- The local scorer also blocks `If`/subgraph branching: the harness rejects any
  node with GRAPH/GRAPHS attributes in `calculate_memory`, so branch-local
  candidate construction cannot hide intermediate memory.

Verdict: still the best big candidate, but not unlocked.  The family generator
is correct; the missing piece is the exact current-style legality/tie-break in a
smaller form.  Do not replace the incumbent with a marker-only or longest-path
compiler; both are visibly worse on fresh generation.

## S8 (2026-07-02) — free-input einsum plane deletion (+0.522) ADOPTED, bit-identical
And×7 = parallel orientation-mux (NOT sequential). Kept [1,20] legality machinery verbatim
(2026-07-01 lesson: semantic re-solves fail fresh). Detection ReduceMax planes → free-input
einsums 'bchw,ck->bkh/bkw'; cyan family (5 planes) → einsums over stacked data-dependent
selector v_f[1,4,30] (Equal-built one-hots, task110 style; also removes GatherElements OOB
risk); epilogue 9 bool planes → ONE fp16 einsum 'bkh,bkw->bhw' (orientation = operand swap).
9955+166 vs 16803+256 → +0.522. Fresh 2500×2+1500 div 0. fp16 einsum OK in ORT CPU.

## S11 (2026-07-03) — mech-15 finder scout: KILL — cost = 220-node S/U family legality SELECTION; output is a routed bent path, not separable rects. Carrier planes already einsum-folded S8. mech-15 cannot do family selection.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k7(cost 10121): 3.7M 패치, ~400k viols(선형분리 불가, contradictions 아닌 halfspace 불가가 binding). 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
