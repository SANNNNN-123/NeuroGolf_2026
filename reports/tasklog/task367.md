# task367 — e73095fd

**Rule:** Grid W,H in [10,20] (fits 20x20). 2-4 gray(5) RECTANGLE OUTLINES (>=3x3, 1px gray
perimeter, BLACK interior), non-overlapping. A box's column may be -1..W-wide+1 so a box can be
CLIPPED by exactly ONE column at the LEFT (col=-1) or RIGHT (right wall off-grid); never clipped
vertically. Straight 1px gray LINES connect box edges to other boxes / the grid border; lines stay
gray. OUTPUT: every BLACK interior cell of a box outline -> YELLOW(4); gray & background unchanged.
The hard part: line-formed loops fake a 4-direction enclosure, AND boxes can be edge-clipped.

**Current (prior):** 13.14 pts, mem 141212, params 144 (public net). Prior agent: "INFEASIBLE".

**Target tier:** B — interior fill is a closed-form per-box geometric predicate; not a flood wall.

## EXACT rule found (verified 500/500 fresh, isolated)
A black cell is a box interior iff, with the NEAREST gray wall in each of the 4 directions
(rows rU/rD, cols cL/cR; a grid-border CLIP allowed for at most ONE of {left,right}), the rectangle
[rU,rD]x[c0,c1] is a genuine box outline:
- top & bottom walls fully gray across [c0..c1]   (row CumSum span == width)
- left & right walls fully gray across [rU..rD]    (col CumSum span == height; skip a clipped side)
- the HORIZONTAL walls TERMINATE at the box corners: g(rU,c1+1),g(rD,c1+1) NOT gray (right) and
  g(rU,c0-1),g(rD,c0-1) NOT gray (left).  **Only the two HORIZONTAL corner-termination checks are
  load-bearing** (vertical ones are redundant) — they are exactly what rejects a straight line wall
  (a line extends PAST the corner; a box wall terminates AT it).
Clip boundaries use the in-grid mask (grid = solid top-left rect, off-grid = ch0&ch5 both 0):
left clip -> c0=0; right clip -> c1 = last in-grid col of the row (= ReduceSum(ing,row)-1).
The sentinel-ring idea does NOT apply: a clipped box's interior is GENUINELY reachable from the
border through the 1-col clip gap, so flood-from-border is wrong; rectangle-gate is the right tool.

## Attempts
| # | angle | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|
| 1 | naive 4-dir enclosure | - | - | - | 81/300 | line-loops over-fill |
| 2 | rect-perimeter, no corner gate | - | - | - | 85/300 | line-loops fake rect |
| 3 | + horizontal corner-termination | - | - | - | 1500/1500 | EXACT rule found (numpy) |
| 4 | ONNX scans+CumSum+GatherND, 30x30 fp32 | 1150740 | 2355 | 11.04 | 104/104 | exact, too big |
| 5 | crop to 20x20 | 386300 | 1124 | 12.13 | ok | |
| 6 | fp16 working planes (CumSum stays fp32) | 262700 | 1124 | 12.52 | ok | |
| 7 | gridRmax via ReduceSum (drop 2 scans) | 235580 | 1046 | 12.63 | ok | |
| 8 | flat-index GatherND ([A,A,1] not [A,A,2]) | 192540 | 3486 | 12.81 | ok | |
| 9 | reach-8 scans (3 doubling steps; walls <=5 away) | 177180 | 1541 | **12.91** | 500/500 | best, EXACT |

## Best achieved
**13.46 @ mem 102880 params 128 — EXACT GATHER-FREE, isolated fresh 500/500.** Beats 13.14 by **+0.32 (≥+0.3)**.
(Prior best was 12.91 @ 177180 with GatherND; the documented "open angle" gather-free corner gate WAS closed.)

| 10 | GATHER-FREE prefix-carry (this agent) | 102880 | 128 | **13.46** | 500/500 | EXACT, WIN |

### How the GatherND machinery was eliminated (the closed open-angle)
The prior 97.3% gather-free attempt used a LOCAL corner-termination test. The exact replacement is a
DIRECTIONAL CARRY of run endpoints — no per-cell gather, no int64, no GatherND:
- Per black cell, the four nearest gray walls' info is carried IN from the 4 directions by packing
  `prio*BIG + (val+OFF)` only at gray cells (else a fp16-safe sentinel) and taking a directional
  prefix-max; decode `val = pack mod BIG − OFF`, `src = floor(pack/BIG)` (reversed for up/left carries).
  Prefix/suffix-max = `MaxPool` with a one-sided full-length 1-D kernel (ZERO params). A=20 keeps every
  packed integer < 2048 (fp16-exact); BIG=32, OFF=10, P=20.
- Run endpoints (Lstart/Rend/Tstart/Bend) = prefix-max of left/top edge marks and suffix-min (= −prefix-max
  of −ramp) of right/bottom edge marks; edge marks from g AND ¬(shifted g).
- Predicate at interior cell (r,c): nearest gray UP rt / DOWN rb / LEFT c0 / RIGHT c1 (clip side -> grid
  edge via in-grid `lastcol`); top wall run endpoints carried down == [c0,c1] (corner-termination is
  exactly `topR==c1 ∧ botR==c1`); bottom wall same; left/right vertical walls span [rt,rb] (skip a
  clipped side). Verified 1500/1500 numpy oracle, then port-for-port in ONNX.
- Clip = "no gray to the right/left in this row" (rightCol/leftCol carry = −99 when unseen). In-grid
  `lastcol` (= grid width−1) needed because the active crop A=20 ≠ true grid width; derived from
  `max(black,gray)` since input has only those two colours.

### Memory levers that crossed 105KB
fp16 everywhere on the 20×20 crop; share the 4 `prio*BIG+OFF` planes across all carries; drop dead `src`
outputs (4 of 8 carries only need val); fold OFF into prio; drop the per-val unseen-sentinel Where (val
correctness is gated by have_top/have_bot + clip flags); derive in-grid from black∨gray slices (avoids the
3600B f32 30×30 channel-ReduceMax). 155760 -> 122320 -> 117120 -> 110880 -> **102880**.

## Irreducible-floor analysis
The exact rule needs PER-CELL 2-D reads of CumSum/gray at the four wall positions (rU,rD,cL,cR)
for the span + corner checks = 12 `GatherND`, 9 distinct flat indices. Each index is a mandatory
`int64` plane (GatherND rejects int32 under onnx.checker full_check) of [A,A,1]=3200B + an int64
Cast(3200B) = ~6.4KB/index x 9 ≈ 58KB of irreducible int64 machinery; the 20x20 crop is the
SMALLEST exact canvas (34% of grids are >18 wide so A cannot drop below 20). Scans (~40KB) + shifts
(~27KB) + fp16 arithmetic (~50KB) bring the floor to ~177KB → ~12.9 pts. Even maximal further fp16/
bool packing lands ~12.95–13.1; **crossing 13.44 (=13.14+0.3, ≈105KB) is not reachable while the
exact corner-termination requires the GatherND wall reads.**

## OPEN ANGLES (re-attack backlog)
- GATHER-FREE PROPAGATION (would remove ~58KB int64 + the gather machinery): top/bottom/left/right
  wall validity via run-AND/run-OR along axes + down/inward propagation reached **97.3% (16/600)**.
  The residual fails are NON-edge line-loops that fake a clipped box; the propagation's local
  corner-termination (`g 1&2 cells past the corner`) cannot replicate the exact wall-span gate.
  Closing it appears to require re-introducing the exact span read — i.e. the gathers. If a future
  agent finds an EXACT gather-free corner gate, this likely drops to <100KB → a real win.
- Replace col-span gathers with a vertical run-AND keyed on the wall column (needs identifying the
  leftmost interior column gather-free) — untried, ~21KB potential.
- Fold the 2 right-corner gathers into the row-span psum reads (share c1+1 index) — marginal.

## 2026-06-19 re-probe vs kojimar (P=14.77, ext:kojimar7113) — INFEASIBLE to beat +0.3
kojimar SUPERSEDED our 13.46 carry net with a MATCHED-FILTER BANK: mem 22036, params 5667, pts 14.77.
Structure (15 nodes): crop ch0(black)+ch5(gray) to 20x20 -> Concat(2,20,20) -> **QLinearConv rect_w[25,2,9,9]
int8 -> v9 u8 (1,25,20,20)=10000B** (the dominant plane) -> QLinearConv fill_w[1,25,5,5] -> v10 fill ->
Where/Concat into the 10-ch output. The 25 channels = **5 wides x 5 talls (w,t in 3..7)**: channel
k=(t-3)*5+(w-3) is the exact 9x9 outline template for a (w,t) box anchored at its TOP-LEFT INTERIOR CORNER.
Each template: +1 on interior cells, -64 flanking the gray walls + at corner-termination positions (one cell
past the corner). bias=-(interior count); valid response = interior_count>0. fill_w[k] stamps the (w-2)x(t-2)
interior, padded [4,4,0,0] to dilate up-left from the anchor. Replicated EXACT in numpy 300/300.

WHY +0.3 IS UNREACHABLE (proven, not estimated):
- mem ALONE = 22036 caps the score at 25-ln(22036)=**15.00**, i.e. params=0 (impossible) only reaches +0.23.
  +0.3 needs total mem+params <= 20537; realistic best (~mem 17600 + params 4700) = 14.99. Short.
- The dominant `v9` (25ch x 20x20 u8 = 10000B) is IRREDUCIBLE: the box-vs-line-loop discriminator needs
  FULL-OUTLINE verification (gray wall span + horizontal corner-termination). Confirmed NON-SEPARABLE: naive
  4-dir enclosure = 46/150 (line-loops fake it); a single size-agnostic linear filter can't verify 25 distinct
  wall configurations (-64 penalties at one size's wall positions kill valid larger boxes). So all 25 (w,t)
  templates are mandatory; the full 9x9 footprint is used (largest 7-wide box + corner-term reach).
- The only documented alternative (directional MaxPool-carry, this tasklog attempt #10) is 102880B == FAR worse.
- Param-only trims (zero_full 900 const, smaller kernels) cap at ~14.84; cannot cross 15.07.
- Corner-L observation (lines are axis-aligned/straight -> L-corner = box corner) is true and cheaply detectable,
  BUT fill from a corner seed needs variable-extent propagation = a bounded flood (~12-16 fp16 planes ~ 25KB+) or
  the enumerate-sizes bank; neither beats kojimar's 10000B u8 bank. (4-corner-cell + term predicate = 238/400,
  insufficient without full wall-span check.)

VERDICT: kojimar's net is at the practical floor for this rule. No structural lever crosses +0.3.

## 2026-06-30 (S8) — OUTPUT-ROUTING beats the "15.00 ceiling" the verdict missed
The +0.3 verdict above only costed the DETECTION (v9 = 10000B u8 bank, irreducible — agreed). It
treated the OUTPUT reconstruction as fixed, but it wasn't: the live net rebuilt the 10-channel output
with a `Concat` of THREE Pad-to-30×30 planes (v5/v11/v13 = 2700B) + a **900-param `zero_full` carrier**
+ a Split/Where re-derivation of channel 0. Because the rule is purely ADDITIVE (output = input with
interior 0-cells → colour 4), all of that is dead weight — the FREE input already carries every
unchanged channel. Collapsed to one routed op:

    cond   = Cast(Pad(v10))            # [1,1,30,30] bool interior mask
    output = Where(cond, e4, input)    # e4 = one-hot[1,10,1,1] @ ch4, fp32

**14.90 → 15.05 (mem 18700→16200, params 5623→4733), LANDED.** Fresh-gated on 2000 generated
instances (/tmp/arc-gen): 0 divergence vs incumbent, 0 wrong vs truth. This crosses the 15.00 the
verdict called a hard cap (that cap assumed the output planes were mandatory). The detection v9
(10000B) remains the floor, so further gains still need a cheaper enclosure discriminator.
Transferable: **additive task ⇒ `Where(mask_pad, e_color, input)` onto the FREE input** — never
Concat-reconstruct channels the input already has. (Most of 351-400 was already so-routed by the
base; 367 was the lone Concat-rebuild holdout. 363 LOOKED additive but fresh-gate caught 3/2000
divergences — its channel reconstruction is load-bearing — so it was rejected, not landed.)

## INSIGHT (transferable)
⭐ "fill box-outline interiors over line clutter" IS closed-form & exact (not a flood wall): the
discriminator between a real box wall and a fake line-loop wall is **HORIZONTAL CORNER-TERMINATION**
— a box wall's gray run STOPS at the corner; a line passes straight THROUGH (gray persists 1–2 cells
past where the corner should be). Combine: nearest-gray walls (doubling prefix-max/min, capped at the
generator's max box size so 3 steps suffice) + CumSum wall-span equality + the 2 horizontal
corner-termination reads. Clipped boxes are NOT a sentinel-ring case (interior truly touches the
border); detect clip via the in-grid mask and set the clipped side's boundary to the grid edge.
⭐ FLAT-INDEX GatherND (index [N,1] over a flattened [H*W] tensor) HALVES the int64 index cost vs a
2-coord [N,2] index AND removes the Concat — but int64 is still mandatory (full_check rejects int32),
so 12 per-cell 2-D reads still floor a 20x20 net near ~12.9 pts.

## S8 (2026-07-02) — anchor-row crop + Pad fold (+0.070) ADOPTED, div 0
v9 anchor rows 20→18 (generator proof: boxes never vertically clipped ⇒ rows 0/19 never fire;
10000→9000B); output Pad folded into fill conv pads (kills 900B Pad + 400B v10 + 8p).
14800+4725 vs 16200+4733 → 15.051→15.121. Fresh 2500 cached + 800 + 400 uncached + 3000 vs
live onnx: all div 0. Epilogue-fold REJECTED here (u8 QLC domain; einsum entry ≥3600B vs
1800B Where). Floors: v9 9000B (25 non-separable templates), Slice+Cast 4000B, epilogue 1800B
(opset-11 checker rejects bool Pad).

## S11 (2026-07-03) — mech-15 finder scout: KILL — 4725 params = int8 25-template box-outline discriminator bank (rect_w+fill_w), not a LUT; yellow rect EXTENTS come from non-separable outline verification (naive enclosure 46/150). Carrier already folded (S8 Where-onto-input).


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 19525 -> 16690 (+0.157)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].
- Less-fusion: replaced Equal(r5c,0)+Cast(v_z→bool)+And with single Less(r5c,v_z) (valid since v_z∈{0,1}: r5c<v_z ≡ r5c==0 & v_z==1); dropped v_eq/v_zb planes. -800B mem (16100→15300), +0.049 pts (15.2774→15.3266). Fresh-gate 2500: 0 divergence vs incumbent.
