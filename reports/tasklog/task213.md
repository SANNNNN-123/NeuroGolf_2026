# task213 — 8e1813be

**Rule:** Input has n equally-spaced full-width/height stripes (every 3rd line from row `offset`), each a distinct colour from {1,2,3,4,6,7,8,9} (never gray=5/black=0); a gray/black box marker is overlaid somewhere (pure distractor, weight-0 colours). Output is an n×n block: for xpose=0 each ROW r is solid colour[r]; for xpose=1 (vertical stripes) the transpose — each COLUMN c is solid colour[c].

**Current (prior):** 14.84 pts. Prior net built TWO `[1,10,30,30]` BOOL `And` planes (out_horiz, out_vert) — declared 9000B but the ORT trace counted them at fp32 36000B each ⇒ mem 169586 (the as-built net scored only 12.95).
**Target tier:** A (separable row⊗col block routed into the FREE bool output; no per-cell colour plane needed beyond a single colour-index carrier).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior (two [1,10,30,30] And→Or) | — | 169586 | 1013 | 12.95 | — | baseline as-built |
| 1 | canonical row-solt frame, And3 into output | A | 7405 | 157 | — | 105/200 | WRONG: vert needs col-solid, not row-solid |
| 2 | orientation Where → [1,1,30,30] cidx (fp32) → gate → Equal | A | 20125 | 162 | 15.08 | 200/200 | ok but +0.24 MARGINAL |
| 3 | drop redundant hascol/keep; int32 gate | A | 18325 | 162 | 15.18 | 200/200 | ok |
| 4 | cast oc vectors to int32 pre-broadcast (1 fewer fp32 plane) | A | 14965 | 162 | 15.38 | 200/200 | ok +0.54 |
| 5 | drop full colf Conv; per-axis ReduceMax presence + 1×1 weight-k collapse | A | 13765 | 162 | **15.46** | 500/500 | **ADOPTED +0.62** |

## Best achieved
15.458 @ mem 13765 params 162 — adopted (file only, not pipelined). Beats prior 14.84 by +0.62 (Y).

## Irreducible-floor analysis
Three [1,1,30,30] full planes remain: (a) two presence reductions `ReduceMax(input,[3])`/`[2]` → [1,10,30,1]/[1,10,1,30] (1200B each, fp32 — ReduceMax rejects narrow dtypes); (b) `cidx_u` = orientation-select `Where(horiz, ocr[1,1,30,1], occ[1,1,1,30])` (int32 3600); (c) `cidx_g` = block-gate `Where(blockin, cidx_u, -1)` (int32 3600, feeds the FREE-output `Equal`). The two int32 carriers are the cost driver. They resist fusion: the colour along the slot axis (orientation Where) and the 2-D n×n block cut (r<n ∧ c<n) are independent broadcasts that cannot collapse into a single binary `Where`. The Equal feeder MUST be int32 (ORT under ORT_DISABLE_ALL rejects float16/uint8 Equal — both tested and InvalidGraph), so the gated index plane can't go narrow.

## OPEN ANGLES (re-attack backlog)
- Fuse the orientation-select and block-gate into ONE int32 plane: pre-gate ocr[r]→sent for r≥n and occ[c]→sent for c≥n (small vectors), then a single 2-D cut handles only the perp axis — but analysis shows the perp cut is still a full plane, so net plane count is unchanged. Worth a hard look for a 1-plane formulation (~−3600B → ~15.7).
- Narrow the two presence planes: a no-pad row/col-sum Conv `W[1,10,1,30]`/`[1,10,30,1]` could emit the [1,1,30,1]/[1,1,1,30] line vector directly (skip the 1200B presence plane), but the stripe sum is k·W with data-dependent W — needs a divide-by-width; uncertain payoff.

## INSIGHT (transferable)
⭐ "distractor box + REGULARLY-SPACED stripes → compacted n×n" (task213) collapses to a **single colour-index carrier** routed into the FREE bool output via `Equal(cidx_int, arange_ch[1,10,1,1])` — the 10-ch expansion never materialises as an intermediate. Orientation (row-solid vs col-solid) is handled WITHOUT two candidate planes by an **orientation `Where(horiz_scalar, colour_by_row[1,1,30,1], colour_by_col[1,1,1,30])`** that broadcasts both per-line vectors into ONE [1,1,30,30] plane. The n×n block cut is a separable `(r<n) ∧ (c<n)` and is applied as a **sentinel gate** (`Where(block, cidx, -1)` then Equal-to-arange[0..9] matches nothing outside) — no separate mask plane reaches the output. ⭐ Replace a full `Σk·input_k` colour Conv ([1,1,30,30], 3600B) with **per-axis `ReduceMax(input,[3])`/`[2]` presence ([1,10,30,1], 1200B) + a 1×1 weight-k channel-collapse Conv** when you only need per-row/per-col line colours (full-width lines ⇒ presence==value). ⚠️ ORT under ORT_DISABLE_ALL: `Equal` accepts int32/int64/bool only — **NOT float16 or uint8** (both InvalidGraph), so any index plane feeding an Equal output op is pinned at int32.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.003)
**Mechanism (op-census diff):** Boolean-logic simplification: And 6→4, Or 8→6, Not 2→1 (69→64 nodes). −5B.
**Old→new:** mem 1838→1833, params 49→49.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task213_pre_s10.onnx`; source `public_candidates/bobmyers7186/task213.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.

## S16 (2026-07-06) — public bit-identical golf (llccqq624) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
