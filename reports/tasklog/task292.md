# task292 — FLOOR (2026-07-01)

mem/params at structural floor. 168-param ScatterND addresses exactly 2ch x 3rows x 7 col-groups of distinct cells (no dup coords); ScatterND onto FREE input minimal. Mask/Where needs 9000B carrier.

No source change.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** pink source cells (2 channels × 3 rows × 7 column-groups of distinct cells) are scattered to fixed distinct output locations; a Slice+Mul masks the source then ScatterND writes it onto the free input.

**Cost (grader mem 252, params 182):** ops Slice/Mul/ScatterND only. Counted intermediates: `updates` [1,2,3,7] fp32 168B, `pink_source` [1,1,3,7] fp32 84B. Params dominated by `scatter_idx` [1,2,3,7,4] int64 1344B (the address table) + four [3] int64 slice specs 24B each + `update_mask` [1,2,1,1] 8B. Output [1,10,30,30] fp32 36000B is FREE.

**Blocker class:** mem0-param-game / assignment. Working mem is tiny (252B); the true cost is the 1344B int64 ScatterND index enumerating each destination cell. This is a hardcoded coordinate assignment table — already logged FLOOR (no dup coords, minimal scatter).

**Lever:** int64→int32 index would halve `scatter_idx` 1344→672B, BUT the safe-golf note excludes the *ND family (ScatterND/GatherND) from the proven int32-index-safe list — needs explicit ORT grader verification before trusting. No other lever visible.
