# task156 — 694f12f3

## 2026-06-29 compact-bitwise screen

Current source score: 17.536063 @ mem 1697 params 47.

Rule: two yellow rectangles have their interiors recoloured according to which
rectangle is smaller/taller, with optional vertical flip.

The source already uses a compact label program: channel-4 slice, 3x3
`AveragePool` to detect rectangle interiors, scalar row-sum tests to determine
orientation, then a 5-channel 10x10 one-hot (`out5`, 500 B) padded directly to the
free output.

No rewrite adopted.  A full uint8 label-map route would need the 10x10 label plus
a 30x30 padded label before final `Equal`, larger than the current 5-channel
compact one-hot.  The 3x3 interior detector is the semantic floor for distinguishing
border yellow from interior recolour cells.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** two yellow rectangles have their interiors recoloured according to which is smaller/taller, with an optional vertical flip.

**Cost (grader mem 1697, params 47):** ops Gather×4/Less×4/Where×4/And×2/Or×2/AveragePool/BitShift/Einsum/Equal/Pad×2. Counted intermediates: `out5` [1,5,10,10] bool 500B (compact 5-channel emission), `c4_f` [1,1,10,10] fp32 400B (channel-4/yellow mask as float), `avg3` [1,1,8,8] fp32 256B (3×3 AveragePool interior detector), several [1,1,10,10] uint8/bool 100B planes. Params: two [8] int64 pad specs 64B. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** full-output-carrier (with a per-cell interior-detection read). The 500B `out5` compact one-hot is the emission; the 3×3 AveragePool + fp32 mask is the semantic floor for separating border-yellow from interior recolour cells.

**Lever:** fp16 recast candidate — `c4_f` [1,1,10,10] (400B) is a 0/1 yellow mask cast to fp32 → fp16 is exact, halving to 200B; `avg3` (256B) AveragePool output feeds only a Greater threshold, fp16-safe. Potential ~330B if ORT AveragePool/Gather accept fp16. Worth a bit-identical probe.

- S12 추가: 위 fp16 recast 레버는 측정 반증(KILL) — 대상 평면이 fp32 input 직생산(Slice/Einsum)이라 producer-측 fp16 불가, Cast 경계비용이 절감을 초과 (384: +17804B, 126: +56B, 156: +44B). dtype 레버 재탐사 금지.
