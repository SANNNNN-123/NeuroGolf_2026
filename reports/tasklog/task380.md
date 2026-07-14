# task380 — rot90 of a fixed 3×3 grid

**Rule:** every example is a 3×3 grid; output = `rot90(input, 1)`, i.e. `out[i,j] = in[j, 2-i]`
(transpose composed with a row flip). Off-grid (rows/cols ≥3) is all-zero background.

**Incumbent (live, S8 = current best):** mem 0, params 99, **20.40 pts**.
Single `Einsum('ncjk,it,tu,ku->ncij', input, selector, reverse3, selector)`:
- `selector[30,3]` (90) places the 3 active components onto the 30-wide output axis AND
  compresses the 30-wide input col axis back to 3 — the SAME matrix does double duty, so it is
  counted once.
- `reverse3[3,3]` (9) is the anti-diagonal flip core.
- the transpose half is FREE: input row index `j` passes straight through to the output column.
- writes directly to the FREE `output` → **mem 0**.

## Verdict: OPTIMAL (at floor) — no landable reduction found

min_stat lists `floor=64`, but that is a hardcoded FIXED_TRANSFORM class constant, NOT an
achievable bound. The real minimum for this graph is the incumbent's 99:

1. **Placement is irreducible at 90.** The output row axis must be size 30, so any matrix
   producing it is `[30, rank]`. The embedded flip has rank 3 → `[30,3]` = 90 elements. params
   counts zeros too, and `sparse_initializer` is a proven 0-pt dead-end, so the sparsity can't help.
2. **The 9-param flip core is mathematically necessary.** A shared-selector form gives
   `M[i,k] = Σ_t selector[i,t]·selector[k,t]`, which is always symmetric PSD. The flip matrix
   `M[i,k]=1 ⟺ k=2−i` is symmetric but has eigenvalues {1,1,−1} — NOT PSD — so it cannot be written
   as `S·Sᵀ`. The non-PSD part requires an explicit core (`reverse3`), hence the +9.
3. **No zero-param routing exists.** `Transpose→output` would handle the transpose for free, but the
   remaining row-flip of a 3×3-embedded-in-30×30 still needs either an Einsum matrix (params) or a
   Slice/Gather (which materialises a full-plane intermediate → mem ≫ 99). Slice-reverse flips the
   whole 30-axis (rows→29−row), not the 3×3 block, so it's wrong anyway.

Conclusion: incumbent 99/0 is at the genuine structural floor for an embedded fixed permutation.
The +0.44 "headroom" vs the heuristic 64 is illusory. No change landed.
