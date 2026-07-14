# task269 — FLOOR (2026-07-01)

mem/params at structural floor. 360B native 3x3x10 read floor; 4x4 Pad required for crop_and_resize to span full 3*scale extent (tested: dropping Pad breaks geometry, 5x5 vs 6x6).

No source change.
