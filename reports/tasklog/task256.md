# task256 — a65b410d

## 2026-06-29 compact-output screen

Current source score: 17.011118 @ mem 2898 params 50.

Rule: a red horizontal segment defines a triangular green/blue fill inside a
variable width/height top-left grid.  Width and height are independent random
extensions beyond the triangle.

The current graph pays for `black` [1,1,13,13] fp32 = 676 B to recover the true
in-grid background, then builds compact four-channel output (`bg`, `blue`, `red`,
`green`) as [1,4,13,13] bool = 676 B and pads it directly to the free output.

No rewrite adopted.  A full label-map route would need at least a 13x13 label plus
a 30x30 padded label before `Equal` (~1069 B), larger than the compact one-hot.
The black/in-grid slice is not removable from the red segment alone because the
generator chooses width and height independently; red geometry does not identify
the full rectangular grid boundary.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** a red horizontal segment inside a variable-size top-left grid defines a triangular green/blue interior fill; grid width and height extend independently (random) beyond the triangle.

**Cost (grader mem 2223, params 56):** ops ReduceMax×3/ReduceSum×3/Less×3/Where×3/ArgMax/Sign/Equal/Pad. Counted intermediates: `label30` [1,1,30,30] uint8 900B (full-canvas label), `red` [1,1,8,8] fp32 256B, `M`/`base`/`label` [1,1,13,13] 169B each. Params: `pads_lab` [8] int64 64B, `rowc13`/`colc13` fp32 52B each. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** full-output-carrier. The dominant counted plane is the 900B full 30×30 uint8 label emitted before the final Equal→one-hot. The prior log tested a compact 4-channel [1,4,13,13] route and found it ≥ the label plane; the black/in-grid recovery is forced because width and height are independent so the red segment alone does not bound the rectangle (info coupling).

**Lever:** no clean lever. Compact one-hot already explored (log: ≥ label). The three [1,1,13,13] fp32/uint8 working planes (rowc/colc are fp32) could be fp16 (binary/small-int) for ~a few hundred B, but the 900B label carrier dominates and is structural.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
