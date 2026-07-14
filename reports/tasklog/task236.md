# task236 — occupancy XOR of two stacked blocks -> green

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** two equal-size blocks are stacked and separated by a full yellow (4) divider row; output = green (3) where exactly ONE block is occupied (symmetric difference of the two occupancy masks), else background. Verified on train ex1+ex2 (occupancy XOR).

**Cost (grader mem 208, params 29):** ops Slice×2/Equal/Where/Pad only. Counted intermediates: `top` [1,1,4,4] fp32 64B, `bottom` [1,1,4,4] fp32 64B, `patch4` [1,4,4,4] uint8 64B, `same` [1,1,4,4] bool 16B. Params: all int64 slice/pad index specs (`pad_to_output` [6] 48B, four [3] slice specs 24B each). Output [1,10,30,30] uint8 9000B is FREE.

**Blocker class:** already-at-floor / mem0-param-game. A 5-op graph; counted mem is two fp32 input-block slices (immovable dtype) + a bool compare. Params are pure int64 index constants. Nothing to reduce beyond dtype of the index specs.

**Lever:** `top`/`bottom` are fp32 input slices (fp32-input invariant → immovable). `patch4` [1,4,4,4] uint8 (64B) looks like a redundant broadcast staging plane before Where — check whether it can be folded into the Where inputs directly. Int64 slice/pad specs → int32 would trim params ~half but they are Slice/Pad indices (verify grader). Marginal.
