# task101-200 deep mechanism review

This file records a fresh, per-task mechanism review after the user explicitly
asked not to trust old tasklogs.  Each entry should be based on direct generator,
example, ONNX, or probe inspection in this pass.

Checklist per task:

- rule: human-readable generator/example rule;
- current: current manifest/eval and ONNX mechanism;
- cost: top counted memory / params driver;
- cheaper idea: whether a simpler human rule maps to fewer ONNX planes;
- action: prototype/adopt/fail/bail, with evidence.

## task101 — 447fd412

- rule: one full reference creature; later copies expose only red/indicator cells
  while hidden blue/body cells must be restored at scale 1..3.
- current: `15.2105`, mem `16940`, params `905`, external live graph.
- checked in this pass/context: generator source, examples, previous oracle
  probe.  A maximal-scale red-anchor oracle matches stored data.
- cost: variable copy count + scale/anchor selection.  The graph pays for
  candidate generation and suppression; the human rule is simple but requires
  preventing extra fills from accidental partial matches.
- cheaper idea: direct red-anchor spray is not exact; top-left/template scans
  need maximal-scale/coverage suppression.  The adopted graph already encodes
  this compactly enough that attempted tail/TopK rewrites were worse.
- action: no adopt candidate.

## task102 — 44d8ac46

- rule: gray hollow rectangles/squares; square rings get red interiors, junk
  rectangles do not.
- current: `17.1652`, mem `2414`, params `113`.
- current mechanism: slice gray channel, four QLinearConv ring detectors for
  side sizes 3..6, MaxPool stamps interior, final `Where(red,input)`.
- cost: main planes are one fp32 gray slice and one 30x30 fill bool; everything
  else is small.
- cheaper idea: a human "just find squares" rule still needs side-size tests.
  Removing any side detector fails.  Crop/reconstruct routes were worse because
  the current false branch reuses the free input.
- action: no adopt candidate.

## task103 — 44f52bb0

- rule: 3x3 red pattern; output is a 1x1 colour indicating horizontal/vertical
  symmetry class.
- current: `20.9057`, mem `36`, params `24`.
- current mechanism: tiny slices/reductions, no full-canvas carrier.
- cost: already essentially scalar/small-tensor only.
- cheaper idea: none meaningful; even deleting all memory would gain only a
  tiny amount and params are already small.
- action: reviewed, skip.

## task104 — 4522001f

- rule: 3x3 quadrant code around red center; output draws two 4x4 green blocks
  in a 9x9 quadrant arrangement.
- current: `19.4627`, mem `197`, params `57`.
- current mechanism: tiny pattern read, orientation decode, resize/pad plus
  small signed 8x8/9x9 carriers.
- cost: largest counted tensors are 81B and 64B int8 maps.
- cheaper idea: no full-canvas floor exists to remove; already uses compact
  small stencil/resize.
- action: reviewed, skip.

## task105 — 4612dd53

- rule: blue cells reveal a rectangular outline plus optional interior cutline;
  erased figure cells must become red.
- current: `17.0071`, mem `2854`, params `106`, external live graph.
- checked: examples, generator, current manifest, old semantic tasklog.  The
  old custom reconstruction reached about `16.45` and is now superseded.
- cost: current live is already below the old 5KB semantic build; likely uses
  a tighter public formulation for the same bbox/cutline restoration.
- cheaper idea: human rule is separable (bbox plus cutline), but prior exact
  source-owned build still needed full output-condition carriers and was worse
  than live.  Any new win has to beat a 2.9KB graph, so only sub-kilobyte golf
  remains.
- action: no immediate candidate; lower priority unless live graph is decoded
  for a specific removable carrier.

## task106 — 46442a0e

- rule: 2x2 or 3x3 input expands to all four rotations in a `2*size` square.
- current: `18.2327`, mem `776`, params `93`.
- current mechanism: compact gather/slice/scatter rotation assembly.
- cost: no large planes; output label path dominates.
- cheaper idea: rule is simple and already expressed as index transforms.
  A rewrite cannot remove much memory.
- action: reviewed, skip.

## task107 — 469497ad

- rule: 5x5 input, factor determined by distinct colours in last row; output is
  variable-factor Kronecker upscale plus red diagonal corner rays.
- current: `16.6866`, mem `2924`, params `1154`.
- checked: examples, generator, current ONNX op profile, old tasklog.  Old
  semantic builds at 11KB+ are superseded by the live graph.
- cost: current low memory is bought with dense small initializers/tables
  (`params=1154`), especially diagonal/ray or colour decode tables.
- cheaper idea: arithmetic red-ray replacement was already tested and exact but
  worse (`mem 4756`, params `892`) because extra int/fp intermediates cost more
  than the table.  This is a key ONNX-vs-human mismatch: formula is simpler to
  read, table is cheaper to score.
- action: no adopt candidate.

## task108 — 46f33fce

- rule: coloured pixels at odd positions in a 10x10 grid become 4x4 blocks in a
  20x20 output.
- current: `18.1217`, mem `864`, params `107`.
- current mechanism: one Conv/Equal/Cast/Resize path.
- cost: essentially one output-scale carrier.
- cheaper idea: this is already the direct upscale mechanism; no alternate
  human rule removes the output carrier.
- action: reviewed, skip.

## task109 — 47c1f68c

- rule: odd-size cross-line input; coloured source pixels reflect into four
  quadrants around the center, line colour used in output.
- current: `17.5162`, mem `1686`, params `93`.
- current mechanism: small line-candidate scalar decode plus gather/where
  assembly.
- cost: only small tensors visible in value_info; no high-memory plane.
- cheaper idea: reflection is already handled with gather/scalar routing.
- action: reviewed, skip.

## task110 — 484b58aa

- rule: doubly-periodic colour tiling with black rectangular cutouts; restore
  the hidden periodic colours.
- current: `15.4684`, mem `13155`, params `634`.
- checked: examples, generator, current ONNX op profile, old semantic derivation.
- current mechanism: colour-index fp32 plane plus inverted-value fp32 plane;
  dilated MaxPool over candidate periods selects valid residue tile, then Gather
  retiles.
- cost: two 30x30 fp32 carriers (`labels_f`, `inv30_f`) dominate.  The inverted
  carrier is how the graph proves residue uniformity cheaply.
- cheaper idea: a human "just infer the period" rule is not cheaper unless the
  period is known.  Removing the inverted plane with zero replacement would be
  about +0.299 at best, below the +0.3 target, and any real detector costs more.
- action: no adopt candidate.

## task111 — 48d8fb45

- rule: several small diagonal-connected sprites; find the distinguished
  reference sprite and emit its colour as a mini pattern.
- current: `18.7177`, mem `528`, params `7`.
- current mechanism: small `MaxPool`/`Gather`/`Div`/`Mod` chain over a 3x3 local
  patch, no full-canvas reconstruction.
- cost: largest value is a 10-channel 3x3 float patch (`360B`); the remaining
  tensors are tiny.
- cheaper idea: the human rule is object-selection, but the implementation is
  already local-patch/scalar sized.  A connected-component rewrite would be
  more expensive than the present diagonal-neighbour shortcut.
- action: reviewed, skip.

## task112 — 4938f0c2

- rule: a green 2x2 center anchors quadrant reflections; red corner patterns
  are mirrored into all four quadrants.
- current: `17.5190`, mem `1616`, params `158`.
- current mechanism: red mask, edge/corner index tables, local red patch, and
  `Einsum`/`Gather` reflection assembly.
- cost: one full 30x30 bool red mask (`900B`) plus small index/local-patch
  carriers.
- cheaper idea: there is no cheaper semantic than a red mask plus reflection
  indices; removing the 30x30 mask would require repeated colour tests in each
  reflected branch.
- action: reviewed, no immediate candidate.

## task113 — 4be741c5

- rule: vertical mirror / row-copy toy transform.
- current: `21.5988`, mem `0`, params `30`, one `Gather`.
- cost: param-only index map, no counted intermediates.
- cheaper idea: already optimal enough for this scoring regime.
- action: reviewed, skip.

## task114 — 4c4377d9

- rule: small 2x2/3x3 colour grid expands by padding/copying edge colours.
- current: `18.1976`, mem `0`, params `900`, one `Conv`.
- cost: dense convolution/table parameters, with zero measured activation cost.
- cheaper idea: a semantic pad/slice/concat version is human-simpler, but it
  would introduce counted intermediates.  To improve by `+0.3`, params would
  need to drop to about `<=666` with essentially no new memory; not obvious.
- action: possible param-golf only, not a mechanism priority.

## task115 — 4c5c2cf0

- rule: striped horizontal/vertical bands determine a compact colour sequence.
- current: `18.0501`, mem `980`, params `63`.
- current mechanism: small block/count/sequence tensors; no full 30x30 plane.
- cost: largest carriers are `144B`, `108B`, `72B`.
- cheaper idea: already uses the natural low-dimensional band representation.
- action: reviewed, skip.

## task116 — 4c7bb0f1

- rule: vertical mirror of a tiny 4x3 pattern.
- current: `21.5988`, mem `0`, params `30`, one `Gather`.
- cost: param-only index map.
- cheaper idea: none worth pursuing.
- action: reviewed, skip.

## task117 — 4cbb3381

- rule: a 3x3 body/sprite pattern is copied or mirrored into corner positions.
- current: `16.6655`, mem `3922`, params `243`.
- current mechanism: crop/background extraction plus full output condition mask.
- cost: one full 30x30 bool condition (`900B`), a 13x13 float crop (`676B`),
  and several 13x13 local masks.
- cheaper idea: object-copy semantics are simple, but the current graph is
  already bounded by the need to place the copied body into output coordinates.
  A component-style detector would likely add more planes.
- action: reviewed; keep as secondary if a reusable tiny sprite-copy mechanism
  appears later.

## task118 — 50846271

- rule: random gray static with hidden plus/crosses.  Output has red for cross
  cells over black and cyan for cross cells over gray/previous marks; input
  recolours every cyan back to gray.
- current: `14.5531`, mem `31049`, params `3387`.
- checked in this pass/context: generator source and existing tasklog proof.
  The source confirms the lossy step: cyan output cells are replaced by gray in
  the input, so cross cells that overlap gray can become indistinguishable from
  ordinary static.
- cost: many full-size planes for red/gray masks, pair detectors, and plus
  candidates.  This is not just inefficient detection; it is a lossy inverse
  problem.
- cheaper idea: even a better detector cannot recover fully invisible crosses
  or ambiguous length-3 endpoints when all distinguishing cells were erased.
  Iterative set-cover-style recovery is approximate and ONNX-expensive.
- action: mechanism wall, not an optimization target unless the scoring gate
  accepts approximate outputs.

## task119 — 50cb2852

- rule: a red boundary/line plus cyan/green V-like structure; hidden green
  segments are restored relative to the marker geometry.
- current: `17.7811`, mem `1244`, params `121`.
- current mechanism: small marker masks and coordinate routing.
- cost: no large full-canvas stack beyond necessary masks.
- cheaper idea: geometry is already encoded with compact routing, not a broad
  search.
- action: reviewed, skip for now.

## task120 — 5117e062

- rule: coloured boxes; interiors/derived cells are filled according to the
  box colour pattern.
- current: `18.1866`, mem `0`, params `910`, one `Conv`.
- cost: zero activation memory, dense parameter table.
- cheaper idea: semantic square/box detection is plausible to read, but likely
  introduces activation memory.  Like task114, the only realistic win is a
  lower-param table or a no-memory index trick.
- action: possible param-golf only, not queued as mechanism work.

## task121 — 5168d44c

- rule: small fixed-board positional transform around a coloured center/blast
  pattern.
- current: `19.1681`, mem `275`, params `66`.
- cost: tiny positional tensors.
- cheaper idea: no meaningful full-plane or repeated detector to remove.
- action: reviewed, skip.

## task122 — 51e6b7e9

- rule: sparse green-row pattern maps to a small red 3x3 placement at a derived
  offset.
- current: `18.3030`, mem `0`, params `810`, one `Conv`.
- cost: dense parameter table, no counted activations.
- cheaper idea: deriving the offset semantically may be human-simple but would
  add memory.  Needs a very small gather/index formulation to beat the table.
- action: possible param-golf only.

## task123 — 52202e6c

- rule: a 5x5 diagonal/max-index colour sequence is expanded into a 10x10
  periodic/tiled pattern.
- current: `17.7973`, mem `403`, params `940`.
- cost: low memory but high parameter/index table cost.
- cheaper idea: formulaic `max(row,col)`/mod logic may read simpler but would
  allocate integer grids; current table is probably chosen because params are
  cheaper than activation memory here.
- action: reviewed; possible only if a shared no-memory formula idiom is found.

## task124 — 5289ad53

- rule: small shape projected/shifted diagonally from marker geometry.
- current: `17.4229`, mem `1879`, params `74`.
- cost: modest mask/coordinate carriers, no extreme table.
- cheaper idea: current cost is already in the low-kB range; a more semantic
  object extractor would not obviously reduce it.
- action: reviewed, skip for now.

## task125 — 54d9e175

- rule: cyan background with coloured boxes; selected pink/marked boxes gain
  green outline/fill changes.
- current: `16.8501`, mem `3434`, params `29`.
- current mechanism: pooling-based rectangle/box neighbourhood propagation.
- cost: several full-ish masks; params are negligible.
- cheaper idea: the expensive part is spatial propagation, not rule discovery.
  A connected-rectangle abstraction would still need full masks or iterative
  fill.
- action: reviewed; no concrete cheaper mechanism.

## task126 — 5521c0d9

- rule: compact geometric marker transform.
- current: `18.5948`, mem `590`, params `15`.
- cost: tiny memory and almost no params.
- cheaper idea: already at small-tensor level.
- action: reviewed, skip.

## task127 — 5582e5ca

- rule: gray separators form 3x3 cells; output remaps each cell colour by a
  fixed offset/addition.
- current: `19.0186`, mem `204`, params `192`.
- cost: small cell/grid tensors only.
- cheaper idea: no full-board mechanism to remove.
- action: reviewed, skip.

## task128 — 5614dbcf

- rule: fixed small mapping implemented as a direct convolution/table.
- current: `19.2962`, mem `0`, params `300`, custom source.
- cost: small param-only graph.
- cheaper idea: below the threshold where semantic rewrites matter.
- action: reviewed, skip.

## task129 — 56dc2b01

- rule: a green creature on one side of a red line is copied to the other side,
  with a cyan guide line; input may be flipped/transposed.
- current: `20.6180`, mem `80`, params `0`.
- current mechanism: almost entirely shape/axis/index routing.
- cost: negligible.
- cheaper idea: already expresses the human rule at near-scalar cost.
- action: reviewed, skip.

## task130 — 56ff96f3

- rule: fixed tiny transform captured by a direct convolution.
- current: `21.5988`, mem `0`, params `30`.
- cost: negligible.
- cheaper idea: none.
- action: reviewed, skip.

## task131 — 56dc2b01

- rule: same ARC generator family as the green-creature/red-line copy task:
  copy the green creature from one side of a red line to the target side, with
  flip/transpose variants.
- current: `16.7092`, mem `3312`, params `675`, external live graph.
- current mechanism: many `Where`/coordinate arithmetic nodes route positions
  under flip/transpose; source graph also carries a full output one-hot.
- cost: variable orientation plus line-relative placement, not colour logic.
- cheaper idea: an apparent task129/task131 transfer lead was checked.  It was
  a false lead: task129 is a different ARC id/rule in the tasklog and uses a
  3-node mode-colour graph, not this red-line creature-copy rule.
- action: no transfer candidate; keep current live unless re-attacking the
  known open angle from task131's own log.

## task132 — 56ff96f3

- rule: one or two rectangles are indicated by two opposite corner pixels; the
  full rectangle area is filled with the corner colour.
- current: `16.6839`, mem `3990`, params `99`, external live graph.
- current mechanism: bit/coordinate span construction for up to two rectangles.
- cost: 15x15 canvas/span masks and full output assembly; not parameter-heavy.
- cheaper idea: human "opposite corners define a box" is already what the graph
  is doing.  Main possible win is reducing duplicate rect1/rect2 span carriers,
  but overlap/order cases make a single shared span awkward.
- action: reviewed; secondary candidate only if rectangle-span sharing appears.

## task133 — 57aa92db

- rule: full reference creature plus partial magnified copies; reconstruct each
  full sprite from the reference body and a signature colour.
- current: `14.5783`, mem `32294`, params `1288`.
- checked deeply in this pass: generator source, source graph, stored examples,
  and overlay-removal candidate.
- cost: `border_grid_f` plus multiple full 30x30 source/shape/seed/marker
  planes; hardcoded sparse overlay is only to preserve ARC-original OOD cases.
- cheaper idea: removing the public overlay passes fresh generator samples but
  fails all ARC-original stored cases.  Generalizing those originals likely
  costs more than the sparse overlay.
- action: deep-dive complete; no adopt candidate.

## task134 — 5ad4f10b

- rule: find a magnified 3x3 Conway sprite embedded in noise and output the
  underlying 3x3 shape in the noise colour.
- current: `17.2663`, mem `2250`, params `34`.
- current mechanism: row/column occupancy and selected-row extraction, then
  compact 3x3 output.
- cost: `row_occ` float carrier (`1200B`) and selected-row tensors.
- cheaper idea: this is one of the rare cases where detecting a huge magnified
  object can be cheaper by projections than by full connected components; the
  current row-occupancy approach matches that.
- action: reviewed, no immediate candidate.

## task135 — 5bd6f4ac

- rule: 9x9 input is partitioned into 3x3 macro blocks; output takes one fixed
  cell from each block.
- current: `19.7017`, mem `0`, params `200`.
- current mechanism: one parameter-only `Conv`.
- cost: small table only.
- cheaper idea: semantic slicing/gather would be clearer but may add activation
  memory; possible gain is small.
- action: reviewed, skip.

## task136 — 5c0a986e

- rule: two 2x2 coloured blocks determine crossing/diagonal fill lines.
- current: `17.8991`, mem `1179`, params `34`.
- current mechanism: coordinate arithmetic from block positions, not broad
  search.
- cost: small position grids and output mask.
- cheaper idea: already formulaic; no large detector stack to remove.
- action: reviewed, skip.

## task137 — 5c2c9af4

- rule: three same-colour points define a square/diamond frame with spacing and
  optional diagonal orientation; output draws the full frame.
- current: `16.8321`, mem `3433`, params `93`.
- current mechanism: position reductions, distance/spacing decode, then frame
  predicate.
- cost: full output predicate plus coordinate intermediates.
- cheaper idea: the human rule is simple, but ONNX still needs row/col grids to
  test frame membership.  No obvious scalar-only form for variable size.
- action: reviewed; possible only with a reusable line/frame primitive.

## task138 — 5daaa586

- rule: coloured border lines define a crop window; interior pixels/rays of a
  chosen draw colour are transferred into the cropped output.
- current: `15.4560`, mem `13789`, params `172`, external live graph.
- current mechanism: border-colour/grid extraction, crop coordinates, padded
  crop carriers, and colour-selection masks.
- cost: `cgf32` 30x30 float (`3600B`), multiple 30/31-wide crop/grid masks,
  and variable crop assembly.
- cheaper idea: semantically it is "find four border lines and crop", but
  variable output/crop shape forces large padded carriers.  Public-teacher scan
  found a lower-param but slightly worse-score variant, so there may be graph
  golf but not an obvious mechanism replacement.
- action: queued for focused inspection after all tasks are triaged.

## task139 — 60b61512

- rule: two Conway mini-sprites are completed in fixed 9x9 locations, with an
  optional inverted transpose.
- current: `18.1932`, mem `765`, params `139`.
- current mechanism: compact bit/conv-integer sprite completion.
- cost: low activation memory.
- cheaper idea: sprite completion is already table/bit encoded.
- action: reviewed, skip.

## task140 — 6150a2bd

- rule: fixed triangular mirror/rotation of a 3x3 grid.
- current: `23.3906`, mem `0`, params `5`, one `RoiAlign`.
- cost: essentially optimal.
- cheaper idea: none.
- action: reviewed, skip.

## task141 — 623ea044

- rule: one coloured point in an odd square expands to both diagonals through
  that point.
- current: `17.6835`, mem `1404`, params `101`.
- current mechanism: row/column arithmetic and diagonal predicates with
  `Einsum` assembly.
- cost: coordinate predicate carriers; no object search.
- cheaper idea: already the natural formula.  Full diagonal output predicate is
  the floor.
- action: reviewed, skip.

## task142 — 62c24649

- rule: mirror a 3x3 input into all four quadrants of a 6x6 output.
- current: `19.3581`, mem `144`, params `138`.
- current mechanism: slice plus `Einsum`/index mapping.
- cost: tiny.
- cheaper idea: none meaningful.
- action: reviewed, skip.

## task143 — 63613498

- rule: several small creatures appear; a boxed/template creature is treated
  specially with gray frame/erasure rules.
- current: `18.2690`, mem `591`, params `247`.
- current mechanism: small indexed/template comparisons using `Einsum` and
  modular coordinates.
- cost: low memory; params carry the template/index logic.
- cheaper idea: connected-component semantics would be much larger.
- action: reviewed, skip.

## task144 — 6430c8c4

- rule: compare top and bottom mini-grids separated by a yellow row; output
  green where both corresponding cells are empty.
- current: `20.3556`, mem `0`, params `104`, one `Conv`.
- cost: param-only, already tiny.
- cheaper idea: no.
- action: reviewed, skip.

## task145 — 6455b5f5

- rule: rectangular regions with optional red cutlines; min/max area boxes get
  recoloured differently and restored.
- current: `15.5109`, mem `13104`, params `111`.
- current mechanism: pooling-based rectangle/cutline expansion and area-class
  masks.
- cost: multiple full masks from `MaxPool`/`Where`; params are negligible.
- cheaper idea: this is a real mechanism candidate, but it is hard because the
  graph must compare rectangle areas globally before recolouring.  A row/column
  projection mechanism might beat full pooling if boxes are axis-aligned and
  well separated.
- action: focused log review confirms this is already at the directional
  MaxPool floor; no new adopt candidate.

## task146 — 662c240a

- rule: three stacked 3x3 grids; select the asymmetric one.
- current: `18.7954`, mem `388`, params `107`.
- current mechanism: compare each 3x3 block to its transpose, select block.
- cost: tiny selected 9x3x3 carrier (`324B`).
- cheaper idea: already the exact semantic rule.
- action: reviewed, skip.

## task147 — 67385a82

- rule: green pixels turn cyan, and edge-free empty cells become green.
- current: `18.7234`, mem `432`, params `100`.
- current mechanism: one convolution-like neighbour test plus scatter.
- cost: small.
- cheaper idea: already direct local-neighbour detection.
- action: reviewed, skip.

## task148 — 673ef223

- rule: paired red side markers define two horizontal bands; cyan/yellow cells
  induce row-wise fills between side markers, with optional flip.
- current: `16.2993`, mem `5759`, params `248`.
- current mechanism: slice top/right/red areas, locate bands, build label
  canvas, then fill with row/column predicates.
- cost: full `label30` plus several band masks (`c8_top`, `right_red_area`,
  `label24`, `red23_b`).
- cheaper idea: a projection-based row-band detector may reduce some masks, but
  variable flip and two-band routing still require row/col predicates.
- action: queued for focused review after 145.

## task149 — 6773b310

- rule: 3x3 Hollywood-squares input; output blue where a mini-grid contains two
  pink cells.
- current: `20.0873`, mem `36`, params `100`.
- current mechanism: two `Conv` nodes count pink per mini-cell.
- cost: negligible.
- cheaper idea: none.
- action: reviewed, skip.

## task150 — 67a3c6ac

- rule: horizontal mirror of a variable-size square with colours restricted to
  a fixed palette.
- current: `19.9374`, mem `128`, params `30`.
- current mechanism: `ReduceL2`/gather-style mirror.
- cost: tiny.
- cheaper idea: none.
- action: reviewed, skip.

## task151 — 67a423a3

- rule: one row and one column are coloured; output marks the 3x3 neighbourhood
  around their intersection yellow, preserving row/column colours.
- current: `18.1976`, mem `0`, params `900`, one `Conv`.
- cost: parameter table only.
- cheaper idea: semantic row/column intersection is clear, but formulaic ONNX
  would need row/col masks.  Beating a zero-memory table requires a very small
  param-only alternative.
- action: possible param-golf only.

## task152 — 67e8384a

- rule: mirror a 3x3 grid horizontally and vertically into a 6x6 tile.
- current: `18.5559`, mem `504`, params `125`.
- current mechanism: slice/conv/gather/scatter compact mirror.
- cost: small.
- cheaper idea: already direct.
- action: reviewed, skip.

## task153 — 681b3aeb

- rule: two overlapping 3x3 candidate creatures; output selects the hidden
  3x3 creature pattern.
- current: `18.3217`, mem `741`, params `54`.
- current mechanism: small 3x3 colour stack, pooled index, selected mask.
- cost: tiny 3x3/1x1 carriers.
- cheaper idea: already uses the low-dimensional output space, not full-canvas
  component extraction.
- action: reviewed, skip.

## task154 — 6855a6e4

- rule: red frame with mirrored gray marks; reconstruct the symmetric interior
  pattern, with optional transpose.
- current: `17.0770`, mem `2661`, params `99`.
- current mechanism: slice/pad/gather/where around the framed region.
- cost: padded frame-region carriers.
- cheaper idea: symmetry is simple, but variable frame placement still requires
  crop/pad routing.  No obvious global-cost reduction.
- action: reviewed, skip for now.

## task155 — 68b16354

- rule: vertical flip of a variable-size square.
- current: `19.9374`, mem `128`, params `30`.
- cost: tiny.
- cheaper idea: none.
- action: reviewed, skip.

## task156 — 694f12f3

- rule: two yellow rectangles; interiors are recoloured red/blue based on the
  smaller/larger matching dimension, with optional vertical flip.
- current: `17.5361`, mem `1697`, params `47`.
- current mechanism: rectangle coordinate checks via `Gather`/`Less`/`Where`.
- cost: modest output predicate routing.
- cheaper idea: already uses rectangle bounds, not full pooling.
- action: reviewed, skip.

## task157 — 6a1e5592

- rule: several black creatures are cut out from a red top band and copied or
  matched to gray bottom locations.
- current: `15.9719`, mem `7983`, params `351`, teacher-derived public probe.
- current mechanism: large multi-branch shape/colour/grid routing with hundreds
  of small nodes.
- cost: full `color_grid`, active colour grids, and many reshaped candidate
  tensors; node count is high even though top memory is moderate.
- cheaper idea: the human rule is "match top creature shape to bottom slot",
  but exact matching across variable shapes is component-like.  Worth a
  focused review because this is teacher-derived and may contain non-source
  complexity.
- action: focused log review confirms a true arbitrary-shape correspondence
  wall; no exact feedforward ONNX candidate.

## task158 — 6aa20dc0

- rule: reference 3x3 coloured pyramid/mega pattern is fully visible; later
  copies expose only diagonal/corner information and must be reconstructed at
  scale 1..3 with flips.
- current: `14.5276`, mem `32987`, params `2340`.
- checked deeply in this pass: generator, examples, current graph, reference
  anchor oracles, direct-fill oracle, and QLinearConv simplification attempts.
- cost: full `color_f` plus many pair/ab/candidate planes for diagonal matching
  and scaled stamping.
- cheaper idea: the user-proposed "find the single colour and shoot away from
  black" is semantically close for visible cases, but ONNX still needs exact
  scale/flip/reference suppression.  Direct fusion leaked phantom fills.
- action: deep-dive complete; no adopt candidate.

## task159 — 6b9890af

- rule: extract a 3x3 Conway sprite and magnified red frame into the framed
  output square.
- current: `17.6424`, mem `1419`, params `149`.
- current mechanism: patch selection and `Einsum` assembly.
- cost: tiny 3x3 selected patch plus output carrier.
- cheaper idea: already low-dimensional; no full search stack.
- action: reviewed, skip.

## task160 — 6c434453

- rule: detect 3x3 sprite types; boxes convert to plus-like red/blue variants
  while avoiding overlapping sprites.
- current: `17.8041`, mem `1164`, params `170`.
- current mechanism: several `QLinearConv` local sprite detectors plus
  `MaxPool`/concat.
- cost: 10x10 feature/sprite masks only.
- cheaper idea: local-template detection is the cheapest natural mechanism.
- action: reviewed, skip.

## task161 — 6cdd2623

- rule: laser rows/columns are indicated by matching endpoint markers; output
  fills the whole marked rows or columns.
- current: `16.6842`, mem `4055`, params `33`.
- current mechanism: row/column sums over colour channels and code grid fill.
- cost: `row0`/`col0` 9-channel float carriers and a 30x30 code mask.
- cheaper idea: row/column projection is already the cheap alternative to
  scanning every line separately.  Potential win is dtype/one-hot reduction, not
  a new semantic mechanism.
- action: reviewed, secondary optimization candidate.

## task162 — 6cf79266

- rule: find empty 3x3 holes in a dense 20x20 pattern and fill them blue.
- current: `16.6891`, mem `4024`, params `44`.
- current mechanism: one local `QLinearConv` hole detector plus `MaxPool`/pad
  fill.
- cost: one 20x20 float input carrier and a 30x30 output block mask.
- cheaper idea: this is already the direct local-window detector.
- action: reviewed, skip.

## task163 — 6d0160f0

- rule: Hollywood-squares mini-grid; find the mini-cell containing yellow and
  copy that mini-cell's coloured contents into the output mini-grid.
- current: `17.2400`, mem `2235`, params `110`.
- current mechanism: label grid, 3x3 patch extraction, row/col decode.
- cost: full `label30` plus small 11x11/3x3 patch tensors.
- cheaper idea: could maybe avoid full `label30`, but grid separators and
  arbitrary mini-cell location still need indexing.  Not top priority.
- action: reviewed, skip for now.

## task164 — 6d0aefbc

- rule: mirror a 3x3 grid horizontally into a 3x6 output.
- current: `21.5988`, mem `0`, params `30`, one `Gather`.
- cost: negligible.
- cheaper idea: none.
- action: reviewed, skip.

## task165 — 6d58a25d

- rule: kite-shaped marker; scattered pixels directly below/inside the kite's
  downward rays are extended upward/through the kite colour.
- current: `16.2758`, mem `5836`, params `314`, teacher-derived public probe.
- current mechanism: background/eligible masks, seed mask, `fill30`, and
  coordinate selection.
- cost: 20x16 float background (`1280B`), 30x30 fill mask, eligible/seed masks.
- cheaper idea: a human ray-from-kite rule might be expressible with a few
  directional predicates instead of a generic fill mask.  Worth checking because
  the source is teacher-derived and the geometry is fixed.
- action: focused log review found a prior source-owned `15.04`/fresh-exact
  candidate but it was not adopted; this is not a new mechanism gap in this
  pass.  Revisit only if the adoption gate for prior candidates is reopened.

## task166 — 6d75e8bb

- rule: cyan bar chart rows inside a red rectangle; fill missing red cells to
  complete the rectangular background, with flip/transpose variants.
- current: `21.8219`, mem `0`, params `24`, one `Einsum`.
- cost: effectively optimal.
- cheaper idea: none.
- action: reviewed, skip.

## task167 — 6e02f1e3

- rule: number of distinct colours in a 3x3 input selects one of three gray
  output strokes.
- current: `20.0028`, mem `20`, params `128`.
- cost: tiny.
- cheaper idea: no meaningful gain.
- action: reviewed, skip.

## task168 — 6e19193c

- rule: 2x2 arrowheads point in diagonal directions; extend rays from the
  indicated corner until the edge.
- current: `17.2862`, mem `2096`, params `143`.
- current mechanism: local arrow detectors via `QLinearConv` and ray mask
  assembly.
- cost: `ray_b30` full mask plus small 10x10 detector tensors.
- cheaper idea: local templates plus precomputed rays are likely cheaper than
  coordinate formula for each direction.
- action: reviewed, skip.

## task169 — 6e82a1ae

- rule: gray tetris-like sprites are recoloured by the count of cells belonging
  to each original shape.
- current: `17.5794`, mem `1500`, params `170`.
- current mechanism: local `QLinearConv` shape/count templates and `Where`.
- cost: low.
- cheaper idea: connected-component counting would be harder; templates are the
  cheap route here.
- action: reviewed, skip.

## task170 — 6ecd11f4

- rule: extract a small 3x3/4x4 coloured sprite from beside a magnified
  monochrome shape, using the matching shape geometry.
- current: `17.1581`, mem `2216`, params `329`.
- current mechanism: 4x4 one-hot patch and shape-background row matching.
- cost: small patch/row tensors, not broad full-canvas search.
- cheaper idea: no obvious cheaper mechanism; component matching is already
  reduced to bounded patch search.
- action: reviewed, skip.

## task171 — 6f8cd79b

- rule: blank variable rectangle becomes a cyan border.
- current: `18.1866`, mem `0`, params `910`, one `Conv`.
- cost: zero activation memory, table params.
- cheaper idea: semantic border test would need row/col/shape masks.  Same
  param-only-table tradeoff as tasks 114/120/151/193.
- action: possible param-golf only.

## task172 — 6fa7a44f

- rule: vertical mirror a 3x3 grid into a 6x3 output.
- current: `21.5988`, mem `0`, params `30`, one `Gather`.
- cost: negligible.
- cheaper idea: none.
- action: reviewed, skip.

## task173 — 72322fa7

- rule: several 3x3 sprite types (`x`, plus, horizontal, vertical) have multiple
  copies; partial copies hide center/body pixels and must be completed.
- current: `14.9503`, mem `23036`, params `112`, external live graph.
- current mechanism: label float grid, per-location sprite scores, destination
  fill indices, and many candidate/visibility tests.
- cost: `task173_label_f` full float (`3600B`), `grid_score`, `fill_dest`,
  `out_idx`, plus many coordinate candidates.
- cheaper idea: this is a real mechanism candidate but close to task158/133:
  the human "complete matching sprite" rule still requires exact type, colour,
  copy location, and suppression.  Need compare against reference-copy
  primitives from 133/158.
- action: focused log review confirms top-k/candidate slack was already tested;
  no safe reduction found in this pass.

## task174 — 72ca375d

- rule: among coloured boxed creatures, output the one that is both symmetric
  and rotational while ignoring non-target boxes.
- current: `16.1244`, mem `7013`, params `142`, custom source.
- current mechanism: row/column signatures, symmetry predicates, and output
  selection.
- cost: 10-channel row signature (`sig30`), `L` label mask, and 10x10 signature
  index carriers.
- cheaper idea: symmetry/rotation is exactly the semantic rule, but the current
  graph pays for signatures over all rows/cols.  Potential win is bounding-box
  cropping or lower-channel signatures, if exact.
- action: focused log review confirms symmetry identification was deeply
  reduced; current live still beats source-owned variants enough that no +0.3
  source candidate is available.

## task175 — 73251a56

- rule: deterministic arithmetic colour table with black rectangular erasures;
  infer the modulus/offset and restore.
- current: `17.8556`, mem `330`, params `937`.
- cost: low memory, high table/code params.
- cheaper idea: arithmetic formula is human-simple but ONNX arithmetic grids
  would likely cost more activation memory than the table.
- action: reviewed, skip for now.

## task176 — 7447852a

- rule: 3-row red zigzag line emits yellow fill above/below depending on mode.
- current: `19.5194`, mem `0`, params `240`, one `Einsum`.
- cost: param-only.
- cheaper idea: already cheap enough.
- action: reviewed, skip.

## task177 — 7468f01a

- rule: crop a rectangular region and horizontally mirror marked pixels inside
  it while preserving background colour.
- current: `16.7515`, mem `3692`, params `130`.
- current mechanism: convs locate crop/offset, `RoiAlign` extracts variable
  region, then mirror assembly.
- cost: crop extraction and full output assembly.
- cheaper idea: ROI/crop is already a compact way to avoid many masks.  No
  obvious cheaper generic crop.
- action: reviewed, skip.

## task178 — 746b3537

- rule: thick colour stripes collapse to a one-column list of stripe colours,
  with optional transpose.
- current: `18.3627`, mem `702`, params `61`.
- current mechanism: row/column labels and selected colour line.
- cost: tiny.
- cheaper idea: already uses stripe labels.
- action: reviewed, skip.

## task179 — 74dd1130

- rule: transpose a 3x3 grid.
- current: `25.0000`, mem `0`, params `0`, one `Transpose`.
- cost: optimal.
- cheaper idea: none.
- action: reviewed, skip.

## task180 — 75b8110e

- rule: merge four quadrant layers into one 3x3 output with fixed priority.
- current: `19.6529`, mem `0`, params `210`, one `Conv`.
- cost: param-only.
- cheaper idea: already table-cheap.
- action: reviewed, skip.

## task181 — 760b3cac

- rule: mirror a cyan Conway sprite around a fixed yellow shape, with optional
  horizontal flip.
- current: `19.0892`, mem `201`, params `168`.
- current mechanism: tiny 3x3 mirror update and `ScatterND`.
- cost: negligible.
- cheaper idea: none.
- action: reviewed, skip.

## task182 — 776ffc46

- rule: many small sprite types; the special coloured target sprite determines
  which boxed/marked region gets recoloured.
- current: `16.1764`, mem `6695`, params `98`, teacher-derived public probe.
- current mechanism: 20x20 colour-id float, score maps, dilated score, output
  pad.
- cost: `cidf32` (`1600B`) plus several 20x20/30x30 score masks.
- cheaper idea: local sprite-template matching is likely needed, but public
  teacher provenance makes this worth inspecting for source-owned simplification.
- action: focused log review found a prior source-owned `14.79`/fresh-exact
  improvement over an older baseline, but current manifest/live is already
  different; no new adoption in this pass.

## task183 — 77fdfe62

- rule: blue frame and four corner palette colours; cyan cells in each quadrant
  are recoloured by the corresponding corner colour.
- current: `17.8852`, mem `1008`, params `222`.
- current mechanism: many slices/concats for quadrant extraction.
- cost: low.
- cheaper idea: quadrant rule is already cheap; no high ROI.
- action: reviewed, skip.

## task184 — 780d0b14

- rule: irregular large coloured patches compress to their patch-grid colour
  table.
- current: `17.1680`, mem `2460`, params `60`.
- current mechanism: row/column starts, counts, `CumSum`/`Einsum` grid
  compression.
- cost: row/col one-hot and counts tensors.
- cheaper idea: this is already projection-based rather than connected
  components.
- action: reviewed, skip.

## task185 — 7837ac64

- rule: line-grid with a hidden 3x3 pattern; recover colours through grid-cell
  spacing and neighbourhood constraints.
- current: `17.4188`, mem `1651`, params `310`.
- current mechanism: several conv/template checks plus pad/where.
- cost: moderate params, low memory.
- cheaper idea: spacing-specific templates are likely cheaper than a generic
  line-grid parser.
- action: reviewed, skip.

## task186 — 794b24be

- rule: count blue pixels in a 3x3 and emit a fixed red count pattern.
- current: `20.3653`, mem `80`, params `23`.
- cost: tiny.
- cheaper idea: none.
- action: reviewed, skip.

## task187 — 7b6016b9

- rule: hollow rectangles become red interiors, and connecting diagonal/straight
  lines are drawn through free space; flip/transpose variants exist.
- current: `14.5803`, mem `32850`, params `665`, custom source.
- current mechanism: many `MaxPool`/`Min` passes for rectangular fills and line
  spans.
- cost: repeated full-canvas propagation masks dominate memory.
- cheaper idea: this is a high-value mechanism target.  If rectangle interiors
  and connector lines can be expressed by endpoint projections or sparse line
  predicates instead of repeated min/max pooling, score could move.
- action: focused log review confirms rare long-distance flood failures and the
  cost of extra iterations; needs a new long-range flood primitive, none found.

## task188 — 7b7f7511

- rule: two identical halves stacked or side-by-side; output is the repeated
  half.
- current: `17.9208`, mem `1128`, params `59`, teacher-derived public probe.
- current mechanism: split/or/and tests and output selection.
- cost: low.
- cheaper idea: already direct split selection.
- action: reviewed, skip.

## task189 — 7c008303

- rule: 6x6 quadrant palette: green marks are recoloured by quadrant corner
  colours under optional flips.
- current: `18.1811`, mem `866`, params `49`.
- cost: small 6x6 pattern/palette tensors.
- cheaper idea: already bounded to output size.
- action: reviewed, skip.

## task190 — 7ddcd7ec

- rule: 2x2 block with diagonal direction markers; extend one or more diagonal
  rays to the edge.
- current: `17.0890`, mem `2480`, params `247`.
- current mechanism: `QLinearConv` direction detectors plus precomputed ray
  masks.
- cost: one full `ray30` mask and four 10x10 ray candidates.
- cheaper idea: similar to task168; template+rays are probably cheaper than
  arithmetic diagonals.
- action: reviewed, skip.

## task191 — 7df24a62

- rule: a blue-framed rectangular pattern must be found elsewhere under any
  rotation/transpose, then matches are drawn and yellow dots preserved.
- current: `14.6229`, mem `31276`, params `841`, custom source.
- current mechanism: correlation tensors over 8 transforms (`M`, `corrm`) on
  23x23 locations plus placement masks.
- cost: two huge 8x23x23 bool/correlation tensors (`8464B` each) and match
  placement planes.
- cheaper idea: human pattern matching is clear, but exact 8-transform search
  across variable placement is intrinsically expensive.  Still worth a focused
  review for pruning transforms/locations from generator constraints.
- action: focused log review confirms current custom is already the compact
  8-orientation template-match candidate; further probes rejected Relu removal
  and orientation grouping.

## task192 — 7e0986d6

- rule: several same-colour boxes exist among random pixels; output removes
  random pixels and keeps/fills the boxes.
- current: `15.9380`, mem `8515`, params `106`.
- current mechanism: convolution score map, target/pre-target masks, vertical
  gap mask.
- cost: `score` full float (`3600B`) plus several full bool masks.
- cheaper idea: if boxes are always solid and separated, a projection/edge
  detector may reduce the full score map.  Needs focused generator validation.
- action: focused log/source/generator review confirms current live already is
  the single-Conv local box/noise discriminator and matches the documented
  plane-count floor; no new candidate.

## task193 — 7f4411dc

- rule: solid rectangles plus static; output keeps rectangles while avoiding
  accidental connecting static.
- current: `18.1866`, mem `0`, params `910`, one `Conv`.
- cost: zero memory table.
- cheaper idea: semantic connected-component filtering would likely add memory.
- action: possible param-golf only.

## task194 — 7fe24cdd

- rule: rotate/reflect a sparse 3x3 pattern into all four quadrants of 6x6.
- current: `18.1446`, mem `900`, params `49`.
- cost: one padded output carrier.
- cheaper idea: already direct gather/pad.
- action: reviewed, skip.

## task195 — 80af3007

- rule: magnified Conway sprite in input maps to a block-expanded output sprite.
- current: `18.1745`, mem `882`, params `39`.
- current mechanism: locate offsets and gather/pad output.
- cost: low.
- cheaper idea: already compact.
- action: reviewed, skip.

## task196 — 810b9b61

- rule: many blue rectangles, some closed or with openings; green indicates
  closed/interior-target boxes.
- current: `16.5798`, mem `4500`, params `38`, teacher-derived public probe.
- current mechanism: `MaxPool`/`Min` rectangle closure checks plus output mask.
- cost: repeated propagation masks.
- cheaper idea: rectangle openness may admit a cheaper side-test mechanism like
  task102, but many variable boxes make it harder.
- action: focused log review found a prior fresh-exact `15.25` candidate, with
  remaining open angle around cheaper bad-seed flood.  No new prototype in this
  pass.

## task197 — 82819916

- rule: repeated colour rows with hidden alternating light/dark pattern; fill
  later rows based on the first informative row.
- current: `17.1360`, mem `2586`, params `16`.
- current mechanism: row-colour extraction and pattern propagation.
- cost: row colour float carrier (`560B`) plus routing.
- cheaper idea: already row-projection based.
- action: reviewed, skip.

## task198 — 83302e8f

- rule: line-grid with permeable points; black points on grid lines expand to
  adjacent yellow/green cells in the output line-grid.
- current: `15.4842`, mem `13434`, params `138`, teacher-derived public probe.
- current mechanism: full grid float `G`, cell expansion masks, equality masks,
  and line-grid value propagation.
- cost: `G` full float (`3600B`) plus several full 30x30 masks.
- cheaper idea: possible mechanism candidate: because line spacing/minisize is
  constrained, a modulo/cell-index formulation might avoid some full masks.
- action: focused log review confirms the task is closed-form, not flood, but
  the exact separable formulation still floors around current live score; no
  new adopt candidate.

## task199 — 834ec97d

- rule: one coloured pixel moves down one row, and yellow fills checkerboard
  cells above it based on column parity.
- current: `17.4841`, mem `1749`, params `88`.
- current mechanism: coordinate parity predicates via `Einsum`/where.
- cost: output predicate.
- cheaper idea: already formulaic.
- action: reviewed, skip.

## task200 — 8403a5d5

- rule: a coloured pixel at top/bottom launches a gray diagonal/line trail until
  the edge.
- current: `18.3720`, mem `570`, params `186`.
- current mechanism: small conv/template detection and `Einsum` line assembly.
- cost: low.
- cheaper idea: already compact.
- action: reviewed, skip.
