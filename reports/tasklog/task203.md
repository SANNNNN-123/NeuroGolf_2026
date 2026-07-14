# task203 — FLOOR (2026-07-01)

mem/params at structural floor. [10,10] pairwise count-match plane (Equal100+Cast100+ArgMax80) irreducible; ORT rejects ArgMax(bool) so bool->u8 Cast required.

No source change.

## S11 (2026-07-03) — ADOPTED: cnt16 fp16 cast (+0.0605)
18.986 → 19.047 (408B → 384B mem, params 1). dtype_overpay_scan flagged maxc/pair/target/
target_col as U8 (bundled max 72) but per-colour counts reach ~900 on fresh 30×30 → uint8
would overflow (the fresh gate would have caught it); fp16 (integers ≤2048 exact) is the
correct width. One Cast off the PRODUCER_BOUND cnt plane; downstream chain fp16.
⭐TRANSFERABLE: bundled-observed max is NOT the dtype bound — size u8/fp16 to the GENERATOR
max. Gates: bundled fail=0, fresh 2000 divergence 0. Backup: reports/retired_networks/task203_pre_s11_recast.onnx.
