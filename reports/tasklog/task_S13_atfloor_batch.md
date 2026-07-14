# S13 UNKNOWN triage — at-floor batch (235, 322, 347, 362, 393)

## 2026-07-03 S13 — opus triage verdict: all AT-FLOOR (genuine floors)

These 5 UNKNOWN-bucket tiny nets were triaged for byte-saving levers; all sit at the
irreducible floor within our op vocabulary. No candidate built. Do not re-probe without a
genuinely new idea.

| task | grader mem/params | why AT-FLOOR |
|---|---|---|
| 235 | 111/59 | `sample_indices` INT64[1,1,3,3,3] GatherND barcode read; GatherND indices are int64-mandatory and the 27-cell pattern is irregular (not a strided Slice). |
| 322 | 144/56 | `color_weights` FLOAT[1,10,2,2] Conv weight must match fp32 input dtype (fp16 forces an 18KB fp16 input Cast); `pad_axes` must stay int64 (Pad rejects int32 axes). |
| 347 | 117/26 | MaxUnpool `I` (INT64[1,2,3,3]) and `output_shape` are int64-mandatory; templates already fp16. |
| 362 | 408/113 | walk-einsum family; `coordinate_weight`/`foreground_channels` are fp32 Einsum operands (must match fp32 input); scatter carriers minimal. |
| 393 | 125/19 | TopK output/K int64-mandatory; Slice `ends` shared with TopK's k → can't drop without a split saving <4B → <0.03 pts. |

Reminder captured this session: `src.harness.calculate_params` = `math.prod(init.dims)` =
ELEMENT COUNT, so int64→int32 on any INITIALIZER saves 0 params (only node-output int64
index tensors are counted by bytes in `calculate_memory`). 103/395/376 were false-flagged on
this and measured-refuted (Δparams=0); see their tasklogs.