# task042 — 22233c11

## 2026-06-29 m1 direct-conv probe

Rule: a 10x10 grid contains one or two green two-block motifs at magnification
1..3; output adds cyan blocks on the opposite diagonal ends.

Current source score: 16.957622 @ mem 2792 params 318.  Largest tensors are the
full-canvas `cyan30` bool plane (900 B), the 10x10 green slice (400 B), and five
10x10 fp16/bool score planes.  The source already uses separate m=1, m=2, and m=3
template machinery.

Tried replacing the m=1 pair-detector + `ConvTranspose` shift path with a single
5x5 fp16 Conv that ORs the four possible two-green offset pairs.  Result: invalid
semantic rewrite.  Stored eval dropped to 172/266 and fresh failed immediately:
m=2/m=3 blocks partially satisfy the m=1 offset-pair positions, producing false
cyan pixels.  The current m=1 two-stage detector is therefore not just historical
slack; it prevents cross-scale false positives.

`onnxsim` gave no score change.  Re-open only with a cross-scale separating code,
not a plain summed template kernel.

## 2026-06-29 semantic uint8 detector probe

Re-read the generator: each instance uses one `magnify in {1,2,3}` and one or two
green diagonal `m x m` block pairs; cyan is placed at the opposite diagonal
off-grid-adjacent positions.  A candidate semantic integer detector tried to
enumerate `m=3,2,1`, require the two green `m x m` blocks, and reject scale
confusion by requiring the local diagonal `2m x 2m` bounding box to contain
exactly `2*m*m` green cells.

Result: **rejected before ONNX**. Python reference scored only stored `2/4` and
fresh `311/1000`. The simple bounding-box exactness predicate misses valid
generator placements and still does not encode the full off-grid cyan/anchor
semantics. Do not implement this as QLinearConv. A viable rewrite needs an
explicit cross-scale separating code that matches the generator's anchor
geometry, not just required-green template occupancy.

Follow-up: adding the missing clipped-cyan semantics and requiring each green
`m x m` block to be an exact independent component (all block cells green, no
orthogonally adjacent green on its border) produced a Python reference with
stored `4/4` and fresh `20000/20000`.

However the source-owned ONNX version was not competitive. It used per
scale/flip `QLinearConv` positive-count and border-count detectors plus fp16
`ConvTranspose` stamp kernels. Stored eval passed `266/266`, but scored only
`16.28068262449363` at `memory=5170`, `params=950`, far worse than the current
`16.957622` at `memory=2792`, `params=318`. The exact integer semantics are real,
but the expanded detector bank is more expensive than the current compact fp16
template graph.

## 2026-07-03 S12 — mech-16 (runtime-parameterized stamp) KILL (이중 반증)
m∈{1,2,3} 전역 스칼라 검출은 성립 (green-cell count ReduceSum 스칼라 프로브, 범위
{2,4}/{8,16}/{18,36} 분리). 그러나: ① 파라메트릭 패밀리 부재 — 인컴번트 3필터는
개별 피팅된 matched filter지 스케일 패밀리가 아님 (m=1 커널의 Kronecker×m 업스케일
= fresh 1500/1500 전패; +1 8-offset 클린 conv = 1399/2000 실패 — 페어링 강제 불가).
② 비용 플로어 — 올바른 빌드는 2-스테이지(anchor+stamp) 필요, 신규 중간텐서 ≈ 뱅크
절감분(~992B). 최소 mech-16 구조는 2927<3110로 싸지만 bundled 24/266 실패로 사망.
지배 비용(green_f 400B + cyan30 900B)은 구조적. 산물: reports/candidates/task042_rts.py.
⭐경계: tasklog의 "template bank"가 피팅된 필터 집합이면 mech-16 부적용 — 룰이
기하학적 스케일 복제일 때만 성립. 재탐사 금지.
