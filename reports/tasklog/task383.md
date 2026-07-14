# task383 — f1cefba8

**Rule:** One axis-aligned box (2-px OUTER ring colour C0, SOLID inner block colour C1) sits on a bg-0 canvas. A few "barnacle" markers are single C1 pixels placed ON the C0 inner-ring line (rows brow+1/brow+tall-2, cols bcol+1/bcol+wide-2). Each marker projects a STRIPE perpendicular to the ring it sits on: a top/bottom-ring marker -> a full COLUMN stripe through its column; a left/right-ring marker -> a full ROW stripe through its row. Output = clean box PLUS, for each stripe line: INSIDE the box the crossed col/row flips to C0; OUTSIDE the box the crossed col/row is painted C1 along the full grid extent (perpendicular direction only). The construction is fully SEPARABLE into 1-D row/col vectors.
**Current:** 14.37 pts (prior public net), mem unknown
**Target tier:** A — separable row/col routed into a free bool output; no flood-fill, no 2-D component labelling. Tier S is blocked because output colours C0/C1 vary per instance (a fixed Conv can't route arbitrary colours), so a recovered-scalar colour-index plane + Equal one-hot is the minimal admissible form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | separable 1-D vectors, fp32 planes, Mul-reduce colf | A | 90368 | 24 | 13.59 | 266/266 | correct but heavy |
| 2 | 1x1 Conv for colf (kill [1,10,30,30]) | A | 54368 | 24 | 14.10 | - | better |
| 3 | cast colf->fp16, axis-reduce ingrid, fp16 downstream | A | 39364 | 24 | 14.42 | - | better |
| 4 | compact label (min 30x30 plane count) | A | 34864 | 24 | 14.54 | 266/266 | better |
| 5 | Where-fuse mask+select (drop Cast/Mul planes) | A | 29464 | 25 | 14.71 | 266/266 | beats target |
| 6 | drop nonbg gate in C0 recovery (bg colf=0<C0) | A | 27664 | 25 | 14.77 | 500/500 | FINAL |

## Best achieved
14.771 @ mem 27664 params 25 — adopted? N (left at src/custom/task383.py for caller). Beats prior 14.37? Y (+0.40).

## Irreducible-floor analysis
Dominant intermediate is the one fp32 colour-index entry plane `colf32` [1,1,30,30] = 3600B from the 1x1 Conv (the 10->1 reduction must emit fp32; ORT upcasts). It is immediately cast to a single fp16 `colf` (1800B) and ALL downstream full-canvas ops run in fp16 (task377 lever). Remaining mem is ~9 fp16 [1,1,30,30] planes (1800B each: colf, colf_inner, colf_c0, c1ring_r/c, out_ig, out_val, in_val, L) and ~6 bool planes (900B each). These are the minimum needed for: colour recovery (2 masked-max planes), stripe detection (2 masked-max planes), and the 4-stage label Where chain. Cannot drop below the colf32 fp32 entry without casting the whole input to fp16 (18000B, worse).

## OPEN ANGLES (re-attack backlog)
- Fuse the two colour-recovery masked planes (colf_inner, colf_c0) — both reduce to a [1,1,1,1] scalar; a single packed accumulation (e.g. additive band encoding of "inner vs ring colour") could collapse them to one plane (~ -1800B -> ~14.85).
- The stripe-detection c1ring_r/c1ring_c could potentially share one masked plane if ring-row and ring-col detection were packed into one ReduceMax over a combined ring mask with a magnitude band, but the two reductions are along different axes so a single plane is non-trivial.
- in_val/out_val/L Where chain (3 fp16 planes) might fold via an additive colour-index arithmetic (rowcode + colcode + box term) routed into ONE final Equal, but the label is piecewise (not rank-1 separable), so this needs a banded encoding — uncertain payoff.

## INSIGHT (transferable)
"Markers as pattern-breakers projecting perpendicular stripes" decoder: a single anomalous pixel ON a structural ring encodes a full line perpendicular to that ring — detect via (colour==C1) reduced over a ring-line mask, NOT as a 2-D correspondence problem. The whole task is separable into 1-D ibr/ibc/ringrow/ringcol/SC/SR vectors and only materializes ~4 unavoidable 30x30 Where planes for the piecewise {bg,C0,C1,off-grid} label. ⭐ Reusable micro-lever: for a masked argmax/max over a 30x30 plane, `Where(mask_bool, colf, sentinel)` then ReduceMax FUSES the mask+select into ONE plane vs Cast(mask)->Mul->ReduceMax (saves a full fp16 plane each, ~1800B per recovery). Also: when excluding one colour for a max, you often don't need a separate "nonbg" gate — bg/off-grid cells carry value 0 which loses to any real colour >=1 in the ReduceMax.

---

## 2026-06-19 re-golf from ext:kojimar7113 (16.10) — NEW FINAL: 16.29 (+0.19)
The 14.77 src above is SUPERSEDED — kojimar7113 (mem 7246) was adopted and is far better than
our 30x30-fp16 separable build (27664). Re-golfed FROM kojimar's net:

**Kept kojimar's front-end** (the real win vs the old src): a dilated 2x2 Conv (only [0,0] weight
nonzero, dilations=[6,6]) emits the colour-index plane CROPPED to the **24x24 active region**
(width/height <= 16+randint(4,8) <= 24): bg->10, off-grid->11, colour c->c. This makes the forced
fp32 entry plane only **2304B** (24x24) instead of 3600B (30x30) — crop-to-active applied via the
dilated conv. bbox via ArgMax of per-row/col occupancy; C0/C1 corner reads; marker indicators
row_select_u8/col_select_u8.

**Replaced kojimar's back-end** (TopK + ScatterND + GatherElements/ScatterElements = 3 sequential
24x24 colour planes + index/scatter machinery, total 7246) with a **plane-lean separable Where**:
1. in_row/in_col come FREE from the box-occupancy vectors `row_any_b`/`col_any_b` (box contiguous
   => "row has a box cell" == "in-box row") — kills the whole ramp+compare in-box machinery.
2. Fold the in-grid gate INTO tiny 1-D stripe-colour VECTORS rowval[1,1,1,24]/colval[1,1,24,1]
   (off-grid->11 sentinel, in-box->C0, out-box->C1), so the stripe MASK stays a 1-D marker
   broadcast (rowmark/colmark) — **no full 24x24 And-mask plane**.
3. final = Where(rowmark, rowval, Where(colmark, colval, color)) => only **3 sequential uint8
   24x24 colour planes** (color, final0, final), Pad to 30x30 (900B), Equal -> FREE bool output.

mem 7246 -> **5974** (params 96). **opset 12** required (uint8 ReduceMin/Max need opset>=12;
scorer checks domain not version). Verified stored 266/266 + ISOLATED fresh **200/200**.

**Floor / why +0.3 (16.40) is infeasible:** entry fp32 2304 (input fp32 => can't be born fp16;
casting input = 18000B) + Pad-to-30x30 output plane 900 + 3 sequential uint8 24x24 colour planes
1728 (two-AXIS stripe painting needs base+after-col+after-row; rowmark[r]|colmark[c] is 2-D-coupled
=> min 2 nested Wheres) + irreducible bbox/marker/colour-read machinery ~950 + 96 params ~= 5974.
Even machinery->~300 lands ~5328 -> 16.34, still short of 16.40. WORK=24 is hard (gen can hit 24x24).

**INSIGHT:** A "stripe/line-union painted with position-dependent colour" task beats scatter via the
separable Where ONLY if you (1) reuse the box-occupancy ArgMax vectors AS the in-box masks (free)
and (2) fold the in-grid gate into the tiny per-axis colour VECTOR so the stripe condition stays a
1-D broadcast (zero full And planes). But two AXES still force 3 sequential colour planes; with a
forced fp32 entry + 30x30 output the floor sits ~16.3 — re-golf buys +0.19, not +0.3.

## S16 (2026-07-06) — public bit-identical golf (llccqq624, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
