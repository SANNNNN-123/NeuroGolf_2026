# task370 — e8dc4411

## 2026-06-29 QLinear repeat-bank screen

Rule: a black sprite is shown once on a coloured background, with one hint pixel
of the foreground colour indicating the diagonal repeat direction.  The output
copies the sprite repeatedly along that direction, recoloured to the hint colour,
until it leaves the grid.  The generator uses width/height 10..20, sprite length
3..4 in fresh samples, while stored examples include additional stress shapes.

Current source/live graph:

- score `15.751016004431612`
- memory `9127`
- params `1267`
- tags: `conv_heavy`, `qlinear`, `scan`, `scatter`

Current mechanism:

- Slice the black channel to a 20x20 active grid, then cast to uint8.
- Infer hint colour and direction signs from the black sprite centroid vs hint
  pixel.
- Build a bank of repeated-sprite masks with exact uint8 `QLinearConv` for
  candidate spacings/footprints (`repeat_d5`, `repeat_d4`, `repeat_d3`,
  `repeat_d2`).
- Select the first valid candidate and emit `Where(repeat_mask, hint_color,
  input)` so the 10-channel output copy is free.

Probe:

| question | result |
|---|---|
| Can active canvas shrink below 20x20? | No. Fresh generator spans all width/height pairs 10..20. |
| Can sprite/repeat candidate bank shrink from stored bounds? | No clear safe cut. Stored/fresh black footprint ranges roughly 4..17 cells. |
| Can only the selected repeat mask be computed? | Not in ONNX without control-flow; all candidate branches must be materialised before selection. |

Adopt decision: **no rewrite**.  This graph already uses the important
source-owned mechanism: uint8 `QLinearConv` repeat masks plus free output `Where`.
The remaining cost is the fixed candidate bank and 20x20 repeat planes, which
match the generator/stored bounds.

Transferable negative insight: when a repeat/stamp task has several possible
step sizes or footprint banks, QLinearConv makes each candidate cheap, but ONNX
still pays every branch.  Do not spend time trying to remove candidate branches
unless the generator proves a strict stored+fresh subset.

## 2026-06-30 — S6-deep WIN: dedup mirrored dilated-conv weight table
- bank_d2 stored two 19×19 dilated-conv weight planes (722 params); plane1 == plane0
  mirror-flipped, each just a 9-elem diagonal. Replaced with d2_zero(361)+static
  d2_idx_all(72)+d2_updates_bank(36) and Gather(sign)→ScatterND reconstruction. Key:
  keep INDICES static, make UPDATES the sign-gathered tensor (18B uint8 intermediate)
  so mem rises 18 while params drop 253.
- Verified vs REAL networks/task370.onnx: 0 divergence on 3000 fresh + bundled 266/266.
- **mem+params 9904→9669, pts 15.799→15.823 (+0.024). ADOPTED (custom:task370).**

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 8645/15.82; iterative diagonal-stamp; black20 1600B forced by uint8 QLinearConv (can't come from reduction); both levers already applied. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-3 opus agent; see agent report in submission_log context). Do not re-attempt without a new mechanism.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — output = arbitrary sprite stamped along a ray (QLinearConv stamp, non-separable); recolour already free-routed via Where. Cost = fp32 entry crop + 4-dilation candidate bank + mux chain + 900B routing mask. Only conceivable lever = runtime-parameterized stamp (does not exist).

## 2026-07-03 S12 — ⭐ ADOPTED: runtime-parameterized stamp (playbook mechanism 16)
S11 "runtime-parameterized stamp (does not exist)" 반증. 4-후보 dilated-conv 뱅크
(repeat_d2..d5 4×400B + mux 1200B + w/idx 뱅크) → d 선-검출(clamped GatherND 프로브
4회 → ReduceMax(valid·[2,3,4,5]), 스칼라만) + 파라메트릭 ScatterND 커널 조립
(base_idx·d → zeros[31,31], 방향은 탭 좌표에 베이크, OOB 탭은 off-diagonal trash셀에
update 0으로 클램프) + QLinearConv 1회. dilation은 static attr라 물리 커널 필요 →
centered half-size C=15 (= generator 최대 스탬프 오프셋; C=14는 fresh 2/2500 실패로
경계 확인). 기각 변형: one-sided+8flip평면 10796, 41×41 10358.
- mem 8645→7477, params 1024→1094, 비용 9669→8571, pts 15.823→15.944 (+0.1205)
- 게이트(오케스트레이터 직접 재측정): bundled 266/266 fail=0; fresh 2000
  vs networks 인컴번트 bit-identical 발산 0 + truth 실패 0; 채택 후 src↔networks
  raw 발산 0/800. 백업: reports/candidates/task370_incumbent_s11.onnx + git.
- ⭐TRANSFERABLE: 뱅크-구체화 스탬프 넷 전반 (센서스 #3 버킷). 잔여 비용: black20
  1600B(fp32 entry crop, 구조적), w_runtime 961B, rvb30 900B(라우팅 마스크).


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.056)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 8104 (mem 7331 + params 773), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
