# task84 — 3bd67248

**Rule (cristianoc oracle):** square grid; col 0 holds a colour, rest background. Output overlays
colour-4 on the bottom row (cols 1..W-1) and colour-2 on the anti-diagonal cells (A-1-c, c). Overlay
positions depend ONLY on grid size; overlaid cells are always old-colour 0. Routed via a single
ScatterElements onto the FREE output.

## S5 win — dedup replicated table (LANDED +0.060)
**Before:** mem 1797, params 241, total 2038. The `row_offsets` init was (1,5,2,21)=210 params but the
5 channels were IDENTICAL copies — pure redundancy.
**Change:** store once as (1,1,2,21)=42 params, broadcast the channel axis via a small ones-plane scaled
by last_row (=A-1). Channel dim stays 5 (must address channel 4). c=0 offsets (-22,+9) preserved (route
masked col-0 writes into padding so the coloured col 0 is never wiped).
**After: mem 1837 (+40), params 83 (−158), total 1920, pts 17.44.** evaluate fail 0/175;
`fresh_verify 84 "" 1500` fail 0. ⭐ TRANSFERABLE: a param table replicated identically across the
channel axis → store 1 channel + broadcast (params are element-count, so dedup beats the small mem add).

## S11 (2026-07-03) — FLOOR CONFIRMED at 1895B; free-einsum route priced at 41172B (23x worse)
Dossier lever (+0.85 via mech-15 bottom row + residual diagonal scatter) REFUTED by build:
⭐TRANSFERABLE CONSTRAINT — only ONE op writes the free graph output. A hybrid
(free einsum for the separable part + scatter for the rest) cannot compose without a
counted [1,10,30,30] intermediate (18-36KB). Full epilogue-fold forces the A-dependent
non-separable anti-diagonal operand to a counted [3,30,30] fp32 plane (fp32 mandatory —
input-matching) → measured 41172/1299 = 14.34 (gates clean, kept as pricing artifact at
reports/candidates/task084_signed.py). Incumbent ScatterElements-into-FREE-input encodes
row=last_row−c in a [1,5,2,21] index plane — already optimal. Every scatter operand priced
minimal: indices int32 floor, updates dtype-BOUND to fp32 data (⭐ScatterElements updates
can never be recast when data is the free input), 5-ch/2-slot/21-col spans forced,
data-dependent updates (invalid-col masking) can't become initializer. dtype_overpay_scan's
084 entry (+0.458 U8 updates) = false positive by the same dtype-binding rule.
