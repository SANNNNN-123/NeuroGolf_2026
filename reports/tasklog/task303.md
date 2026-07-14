# task303 — c1d99e64

**Rule:** Input = black bg (colour 0) + one foreground colour; a set of full ROWS and COLS
were forced entirely black ("straightaways"). The generator guarantees every non-line in-grid
row/col has >=2 colours before painting, so a line row/col is exactly an in-grid row/col whose
every cell is black. Output: `out[r][c] = red(2) if (row r all-black OR col c all-black) else input[r][c]`.
**Current:** 15.95 pts, Conv→fp32 v-plane + uint8 label + Equal, mem 8520, params 23
**Target tier:** A — separable row/col line masks routed into the FREE output; no per-cell colour plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | rsum/csum col-reduce → black_row vs in_row; Or(rowline,colline); Where(red,input) | A | 3960 | 31 | 0.0 | 7/200 | off-grid leak: line row paints red into off-grid cols |
| 2 | + in-grid rect gate (row_ingrid AND col_ingrid) | A | 5760 | 31 | 16.34 | 200/200 | ADOPTED |

## Best achieved
16.34 @ mem 5760 params 31 — adopted? Y. Beats prior 15.95? Y (+0.39).

## Irreducible-floor analysis
Dominant: rsum/csum `[1,10,30,1]`/`[1,10,1,30]` = 1200 B each (2400 B total). A per-channel
single-axis reduce is the cheapest way to split the per-row/col BLACK count (channel-0 slice,
120 B) from the per-row/col IN-GRID count (sum over channels) WITHOUT ever materialising a
`[1,*,30,30]` colour plane (slicing ch0 of input first = 3600 B; slicing chs 1..9 = 32400 B).
Plus three `[1,1,30,30]` bool planes (lineraw=OR, ingrid=AND, linemask=AND) at 900 B each —
the OR of two independent axis line-conditions, each needing the orthogonal in-grid gate, is a
hard 3-full-plane minimum here.

## OPEN ANGLES (re-attack backlog)
- Collapse the 3 bool full-planes to 2: fold the in-grid gate into the OR operands
  `(rowline∧col_ingrid)∨(colline∧row_ingrid)` is still 3; no obvious 2-plane form found.
- Try MatMul-based fg-count to drop one reduce — unlikely to beat the 1200 B reduce floor.

## INSIGHT (transferable)
"Recolour all-black in-grid rows/cols" is closed-form Tier A: line = (per-axis BLACK count ==
per-axis IN-GRID count) ∧ (in-grid count > 0); both counts come from ONE per-channel
single-axis col/row reduce `ReduceSum(input,[3])`/`[2]` (1200 B), splitting black via a 120 B
channel-0 slice. CRITICAL: a 1-D row line broadcasts across ALL 30 cols incl. off-grid, so the
final mask MUST be gated by the in-grid rectangle `row_ingrid ∧ col_ingrid`, else off-grid cells
get painted red.

## S9 (2026-07-03) — kojimar teacher: fractional-encoding line detect (+0.109) ADOPTED
Div(black_count,dim) + ArgMax(Min(count,1)) derives height/width; Greater(a,bv) builds
the red mask arithmetically — no separate bool line planes. mem 1598→1430. Bit-identical
2500 uncached (0/0/0). ArgMax-on-u8 safe (not TopK). Backup task303_pre_s9.onnx.
⭐ TRANSFERABLE: fractional/ArgMax arithmetic encoding collapses bool line-detection
cascades on all-black row/col recolour tasks.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 1452 -> 1450 (+0.001); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].