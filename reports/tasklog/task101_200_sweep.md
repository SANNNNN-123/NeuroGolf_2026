# task101-200 sweep — mechanism-first review

## 2026-06-30 range review

Scope: only tasks 101 through 200.  The review order was mechanism-first:
inspect low-score/high-memory tasks, compare public teacher candidates, then
fall back to generic optimization only where a source-owned exact path looked
plausible.

Global probes:

- `compression_sweep.py --tasks 101..200`: no onnxsim/simple-compression
  candidate adopted; every task reported zero usable gain.
- Public teacher scan over 600 candidate ONNX files for tasks 101..200:
  no candidate scored above current live.  The only notable deltas were
  task138 with 21 fewer params but 23 more memory bytes and slightly lower
  points, and task187 variants that were materially worse than live.

High-ROI deep checks:

- task101: semantic rule was clarified and a red-anchor/maximal-scale oracle
  matched stored data, but source-owned ONNX attempts were equal-or-worse.
  Current 15.21 / 16940 / 905 stands.
- task102: square-ring rule is simple but already encoded as four compact
  ring detectors plus fill.  Removing any side length fails; crop/active-region
  rewrites were worse or invalid.  Current 17.17 / 2414 / 113 stands.
- task158: current live equals public candidates.  The remaining visible slack
  is bool/u8 duplicate carriers around QLinearConv inputs; these are load-bearing
  because logic needs bool and QLinearConv needs uint8.  Expected safe gain is
  only around +0.05 even if a delicate rewrite worked.
- task118: information-loss wall.  Invisible crosses and length ambiguity make
  exact generalization impossible from the input.
- task133: closed-form semantics are known, but the deployed magnify-Gather form
  is far cheaper than shift-stamp reconstruction.
- task187: crop2 flood-fill is the best exact/live candidate in this repo.
  Extra iterations improve rare fresh distance but reduce stored score.
- task191: already improved by the 8-orientation stacked-conv build.  The
  remaining floor is the 8-channel match/activation plane; no-Relu and grouped
  reductions fail or are worse.
- task173: TopK width is not slack; constructed worst-case visible count reaches
  the current bound.
- task138, task110, task198, task145, task192: tasklogs and fresh/public probes
  all point to current graphs being at their known plane floors.

Secondary 16.x tasks were spot-checked against tasklogs and current manifest.
Several old logs mention unadopted wins, but current live already supersedes
those with smaller public/teacher graphs.  No source-owned mechanism with a
credible +0.3 path was found in 101..200 during this sweep.

Conclusion: for this range, there is no verified adoption candidate right now.
The next useful work is either (1) a genuinely new primitive for thresholded
QLinear/bool carrier fusion, which would affect task158-style graphs, or (2)
move to another task range rather than spending cycles on documented plane
floors.
