# task229 — keep mode colour, recolour rest to grey

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** 3×3 grid; keep the most-common (mode) colour cells unchanged and recolour every non-mode cell to grey (5). Colour-agnostic mode detection + binary recolour.

**Cost (grader mem 212, params 28):** ops ReduceSum/ArgMax/Where/Equal/Cast×3/Concat/Add/Slice/Pad. Counted intermediates: `output_small` [1,9,3,3] bool 81B (compact 9-channel one-hot), `counts` [1,10] fp32 40B, `mode_channel` [1,1,3,3] fp32 36B. Params: `pad_ch_hw` [8] int64 64B, `channel_ids` [1,9,1,1] uint8 9B. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** full-output-carrier (tiny). The 81B `output_small` 9-channel one-hot over 3×3 is the emission; everything else is a channel-vector/3×3 read. Near floor for a 3×3 recolour.

**Lever:** the output only ever uses TWO colours (the mode + grey 5), so a [1,2,3,3] one-hot (18B) plus a channel-scatter could replace the [1,9,3,3] plane (81B) — but the mode channel is data-dependent, so routing the mode one-hot to its dynamic output channel needs a scatter/Einsum that likely costs back the saving. Marginal; probe only if a cheap channel-route exists.
