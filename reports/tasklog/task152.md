
## S9 (2026-07-03) — mechanism-14 separable-remap einsum (+0.934) ADOPTED
Single 5-operand Einsum 'ra,ai,zcij,bj,sb->zcrs', mem=0: mirror-tile 3x3->6x6 symmetric U/S shared, 458->180.
Gates: stored fail=0; uncached fresh 2000+600: 0/0/0 (bit-identical). No TopK.
NOTE: scan projection was ~8x optimistic — output axis must span the FULL 30 (grading
tensor [1,10,30,30]), so U tables are [30,K] not [out,K]. Backup task152_pre_s9.onnx.
