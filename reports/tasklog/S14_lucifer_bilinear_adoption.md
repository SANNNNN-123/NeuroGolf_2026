# S14 — bilinear/shared-operand Einsum adoption (89 nets, +17.70)

Source: lucifer19/tinyonnx-golf-forge (public LB 7221.43) = base64 ONNX dump.
Merge: kept our better net per task, adopted their cheaper net where BOTH cheaper
AND passing our strict gate (arc-gen 262 + train+test, fail=0). All 89 verified:
source-built (src/custom/taskNNN.py via live_to_exact_source) scores bit-parity
with the adopted ONNX, fail=0. Submitted 54360209 → public **7232.24** (new best).

MECHANISM (shared across the 89) — one Einsum vs FREE input:
1. operand reuse: one low-rank factor referenced N× (P×2 t292, R×2·C×2 t83, m×4 t142);
   exploits row=col / encode=decode symmetry → params halve vs distinct-but-equal.
2. input twice (`input,input,...`): pairwise/auto-correlation features, ZERO counted
   intermediates (t142/t100/t52) — replaces Conv-autocorr/GridSample/RoiAlign.
3. spatial-op collapse: Conv/QLinearConv/ConvTranspose/ScatterND → Einsum; kills the
   big activation plane so grader MEMORY drops even when params rise (t325 2010→518).
See memory neurogolf-bilinear-einsum-lever; per-task deltas reports/lucifer7221_adopt_targets.md.
Backup of pre-adoption nets/sources: reports/backup_pre_lucifer/.

Adopted:
- task003: adopted (m+p → 261), +19.435pts net
- task005: adopted (m+p → 6801), +16.175pts net
- task010: adopted (m+p → 1012), +18.080pts net
- task011: adopted (m+p → 1460), +17.714pts net
- task014: adopted (m+p → 4135), +16.673pts net
- task017: adopted (m+p → 7004), +16.146pts net
- task018: adopted (m+p → 27616), +14.774pts net
- task025: adopted (m+p → 10424), +15.748pts net
- task029: adopted (m+p → 5299), +16.425pts net
- task039: adopted (m+p → 455), +18.880pts net
- task040: adopted (m+p → 190), +19.753pts net
- task044: adopted (m+p → 9126), +15.881pts net
- task046: adopted (m+p → 2285), +17.266pts net
- task049: adopted (m+p → 345), +19.156pts net
- task051: adopted (m+p → 1744), +17.536pts net
- task052: adopted (m+p → 194), +19.732pts net
- task059: adopted (m+p → 549), +18.692pts net
- task063: adopted (m+p → 1873), +17.465pts net
- task076: adopted (m+p → 16557), +15.285pts net
- task083: adopted (m+p → 210), +19.653pts net
- task089: adopted (m+p → 6772), +16.179pts net
- task096: adopted (m+p → 8371), +15.967pts net
- task100: adopted (m+p → 361), +19.111pts net
- task104: adopted (m+p → 238), +19.528pts net
- task105: adopted (m+p → 2943), +17.013pts net
- task107: adopted (m+p → 3813), +16.754pts net
- task109: adopted (m+p → 1650), +17.591pts net
- task111: adopted (m+p → 311), +19.260pts net
- task118: adopted (m+p → 19492), +15.122pts net
- task121: adopted (m+p → 328), +19.207pts net
- task128: adopted (m+p → 136), +20.087pts net
- task141: adopted (m+p → 1383), +17.768pts net
- task142: adopted (m+p → 90), +20.500pts net
- task146: adopted (m+p → 267), +19.413pts net
- task152: adopted (m+p → 90), +20.500pts net
- task156: adopted (m+p → 1564), +17.645pts net
- task157: adopted (m+p → 7243), +16.112pts net
- task161: adopted (m+p → 2551), +17.156pts net
- task167: adopted (m+p → 146), +20.016pts net
- task174: adopted (m+p → 4563), +16.574pts net
- task176: adopted (m+p → 210), +19.653pts net
- task183: adopted (m+p → 1048), +18.045pts net
- task184: adopted (m+p → 2160), +17.322pts net
- task191: adopted (m+p → 11882), +15.617pts net
- task198: adopted (m+p → 12806), +15.542pts net
- task203: adopted (m+p → 355), +19.128pts net
- task205: adopted (m+p → 11209), +15.676pts net
- task211: adopted (m+p → 270), +19.402pts net
- task213: adopted (m+p → 1858), +17.473pts net
- task218: adopted (m+p → 651), +18.521pts net
- task222: adopted (m+p → 6203), +16.267pts net
- task229: adopted (m+p → 200), +19.702pts net
- task238: adopted (m+p → 1947), +17.426pts net
- task249: adopted (m+p → 261), +19.435pts net
- task250: adopted (m+p → 2542), +17.159pts net
- task252: adopted (m+p → 180), +19.807pts net
- task253: adopted (m+p → 511), +18.764pts net
- task254: adopted (m+p → 687), +18.468pts net
- task255: adopted (m+p → 10809), +15.712pts net
- task257: adopted (m+p → 400), +19.009pts net
- task259: adopted (m+p → 894), +18.204pts net
- task260: adopted (m+p → 1581), +17.634pts net
- task268: adopted (m+p → 3464), +16.850pts net
- task274: adopted (m+p → 175), +19.835pts net
- task275: adopted (m+p → 1356), +17.788pts net
- task284: adopted (m+p → 3649), +16.798pts net
- task292: adopted (m+p → 80), +20.618pts net
- task293: adopted (m+p → 1270), +17.853pts net
- task296: adopted (m+p → 314), +19.251pts net
- task300: adopted (m+p → 528), +18.731pts net
- task319: adopted (m+p → 17143), +15.251pts net
- task325: adopted (m+p → 1484), +17.698pts net
- task327: adopted (m+p → 640), +18.539pts net
- task336: adopted (m+p → 1820), +17.493pts net
- task342: adopted (m+p → 1093), +18.003pts net
- task353: adopted (m+p → 244), +19.503pts net
- task368: adopted (m+p → 6193), +16.269pts net
- task369: adopted (m+p → 1393), +17.761pts net
- task374: adopted (m+p → 2284), +17.266pts net
- task375: adopted (m+p → 327), +19.210pts net
- task377: adopted (m+p → 8209), +15.987pts net
- task379: adopted (m+p → 9133), +15.880pts net
- task384: adopted (m+p → 509), +18.768pts net
- task388: adopted (m+p → 2215), +17.297pts net
- task389: adopted (m+p → 130), +20.132pts net
- task394: adopted (m+p → 2780), +17.070pts net
- task396: adopted (m+p → 4944), +16.494pts net
- task399: adopted (m+p → 60), +20.906pts net
- task400: adopted (m+p → 1227), +17.888pts net

## S14 frontier probe (post-adoption): bilinear lever → un-mined tasks
Ran 8 opus agents on high-cost tasks NEITHER we nor public-7221 optimized
(85/74/162/177/196/279/333/382), applying bilinear/shared-operand/collapse.
Result: 7/8 GENUINE FLOOR, 1 tiny win.
- task382: +0.0073 (int32 count plane → fp16, count≤20 exact). ADOPTED, verified fail=0, 5648→5606 mem.
- FLOOR (do not re-probe): 279 (nonlinear erosion→flood chain), 196 (Laplacian seed+5× flood dilation, k=5 min),
  85 (color-rank-1 Conv but reduce adds plane → mem 4500→11700), 74 (max-over-symmetry-orbit = non-bilinear,
  6 uint8 orbit planes + fp32 decode floor), 162 (sliding-window detector = per-position Conv floor),
  177 (RoiAlign collapse possible but 7696>3621 — fp32 30-wide selectors lose to native 8×8 crop),
  333 (operand-reuse already applied; deriving matrices converts cheap params→pricey activation).
KEY LESson: memory (activation bytes) is the floor on these; any reduction that MATERIALIZES a plane loses
(params cost 1/elem, fp32 activation 4/elem). The 89-net harvest captured the bilinear wins; remaining
high-cost tasks are memory-dominated floors (confirms structural-ceiling). Further gains need a NEW mechanism.
