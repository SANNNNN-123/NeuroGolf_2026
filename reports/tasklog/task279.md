# task279 — b2862040

## 2026-06-29 erosion-depth sweep

Rule: closed blue boxes should be restored to cyan while open blue boxes and
barnacles stay blue.  Current semantic source uses a 16x16 crop, eight repeated
3x3 degree QLinearConv erosions to find closed-box cores, then two cross-neighbour
growth steps inside the original blue mask.

Stored sweep:

| erosion count | stored | mem | params | result |
|---:|---:|---:|---:|---|
| 4 | 20/266 | 4484 | 47 | reject |
| 5 | 82/266 | 4740 | 47 | reject |
| 6 | 180/266 | 4996 | 47 | reject |
| 7 | 242/266 | 5252 | 47 | reject |
| 8 | 266/266 | 5508 | 47 | current |

Adopt decision: **no change**.  The eighth erosion is not slack; generator
instances at the 16x16 bound and 3..5 box sizes still need the full depth to
separate closed boxes from open boxes plus barnacles.  This remains a compact
source-owned semantic QLinearConv solution, but not a 20+ single-conv candidate.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-1/2 opus agents; occupancy/max-semiring reductions or sub-400B u8 banks). Do not re-attempt without a new mechanism.
