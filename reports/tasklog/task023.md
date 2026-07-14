# task023 — 150deff5

## 2026-06-29 last-pass simplification probe

Rule: small 8..9 by 9..11 grid contains gray toys: 2x2 boxes become cyan, 1x3
or 3x1 sticks become red.  Shapes can touch in ways that create false local 2x2
evidence, so a naive `part-of-any-2x2 -> cyan else red` local rule is not exact
(`1851/10000` fresh in Python reference).

Current live/source graph is a QLinearConv peel program: repeatedly classify and
remove unambiguous square/horizontal/vertical toy evidence, then output square
accumulator as cyan and the residual gray as red.  Current score:
`15.517040433249083`, mem `12843`, params `291`.

Probe: prune the graph to fewer peel passes and/or simplify the final pass.

| variant | stored | mem | params | result |
|---|---:|---:|---:|---|
| keep `Sacc1` | 193/266 | 5715 | 291 | reject |
| keep `Sacc2` | 248/266 | 8091 | 291 | reject |
| keep `Sacc3` | 264/266 | 10467 | 291 | reject |
| final pass square-only residual (`Sadd4_sel <- sqpos4`) | 266/266 | 12051 | 291 | fresh failed 12/13 |

Adopt decision: **reject**.  The final ambiguity gate is not stored-only
overhead; fresh generation can still create residual false-square cases after
three peel rounds.  Do not remove the last H/V disambiguation unless a stronger
residual invariant is proven.

Transferable negative insight: for small toy-shape decomposition, apparent
stored-safe "last pass only needs boxes" pruning is unsafe under fresh generator
instances.  Fresh verification must be large before adopting peel-depth changes.

## 2026-06-30 S1 — LANDED (behaviour-preserving crop, fresh-gated)
mem 12843→11603, params 291, pts 15.517→15.6162 (+0.099). Bundled fail=0; fresh 1500
candidate==incumbent (diff 0). Generator places gray only at cols 1..8 (cols=randint(1,6),
max stick col 6+2=8) → never col 0. Shifted peel crop start [5,0,0]→[5,0,1] (region
[8,9]→[8,8]) + output pad spad to re-place at cols 1–8. Dropping an always-empty col is a
no-op (conv already zero-pads) → ~150 working planes shrink 72→64B (≈1240B). teacher→custom:task023.

## 2026-06-30 — FLOOR re-confirmed (golf re-check)
Re-attacked at mem 11603/params 291/pts 15.616. Restored the landed crop incumbent
(an in-flight session had reverted task023.py to HEAD's 12843 b64-init form; node graph
identical, only `ns`/`spad` inits differed — re-applied [5,0,1]+spad to recover 11603).
Findings: (1) iteration depth IS required — K=4 fails 2/266 bundled, K≤3 worse (matches
prior peel-depth rejections). (2) No dead tensors / unused inits (graph already pruned;
iter-4 H/V/asg tail already eliminated — a naive K=5 rebuild measures 12371). (3) Local
rule "gray cell→box iff in any fully-gray 2×2" fails 220/266 bundled & 16288/20000 fresh
→ classification is intrinsically non-local; the 5-round greedy is necessary. (4) fp32
slices (black_f 396B + nz_f 256B) and the 990B pre-pad carrier are structural (input is
fp32; one-hot output needs the 10-ch carrier). Verdict: FLOOR. mem unchanged 11603.

# (appended) S8 2026-07-02 — round/node golf + residual majority rule (+0.618) ADOPTED
The 56-QLinearConv bank = SEQUENTIAL 5-round exact-cover unit propagation on 8×8 u8 (64B
planes — walk-einsum fp32 loses, but the "floor" verdict was wrong: the lever was round/node
golf). 3 rounds × 11 nodes: type-agnostic unique-cover plane funi=(coverage==1); Div folded
into QLC i32-bias saturation; confirm fused per type via 5u+funi threshold QLC (Σ≥21);
Wnt kernel reused as counter AND removal scatter; asymmetric-pad 3×3s replace 5×5s.
NEW residual majority rule (sat_u8(2·nSq−nL)≥1, weight zero_point=128 = signed-in-u8) resolves
most fixpoint-ambiguous blobs the incumbent defaulted to red: fresh fail 147-156/2500 → 50-60.
6163+249 vs 11603+291 → 15.616→16.234. Candidate failure set ⊂ incumbent's (div only on
inc-failing instances). TRAPS: bundled has row-0 shapes (window can't shrink; F=3 forced by
one bundled example); sanitize_model needs globally-unique node output names.
TRANSFERABLE: "spend the incumbent's fresh-failure budget on a cheaper-but-more-accurate
residual rule" for unrolled disambiguation nets.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k7(cost 6412): 514k 패치, 145k viols. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
