# Answer

The reconstructed method is a fixed-column-sparsity Johnson-Lindenstrauss distribution.

For fixed-vector accuracy `epsilon` and failure probability `delta`, choose

```text
ell = Theta(log(1/delta))
s   = Theta(epsilon^-1 ell)
k   = Theta(epsilon^-2 ell)
```

Construct a `k x d` matrix whose column `j` has exactly `s` nonzero entries, each equal to `+1/sqrt{s}` or `-1/sqrt{s}`. The support can be chosen as `s` distinct rows, or as one row in each of `s` equal blocks. Signs and locations only need enough independence for the `ell`-moment proof.

For unit `x`, exact column sparsity makes the diagonal contribution equal to one. The entire error is

```text
Z = ||Sx||_2^2 - 1
  = (1/s) sum_r sum_{i != j}
      eta_{r,i} eta_{r,j} sigma_{r,i} sigma_{r,j} x_i x_j.
```

The proof is a high-moment analysis of this signed collision polynomial. Expanding `E[Z^ell]`, sign independence kills every monomial whose collision multigraph has an odd-degree vertex. The remaining even-degree graphs are counted, while the sparse support variables contribute controlled collision probabilities. With `s = Theta(epsilon^-1 log(1/delta))` and `k = Theta(epsilon^-2 log(1/delta))`, Markov's inequality gives the JL tail bound.

This is more than recombining JL with hashing. Hashing gives sparse updates, but CountSketch-style median aggregation is nonlinear and does not produce one Euclidean embedding. The decisive additions are exact per-column sparsity, no self-collision noise, direct concentration of cross-collisions, and a limited-independence proof matched to moment order `Theta(log(1/delta))`.

The resulting application time is

```text
O(s ||x||_0) = O(epsilon^-1 log(1/delta) ||x||_0)
```

while the target dimension remains on the dense JL order. A reference block-support implementation is in `code/sparse_jl_block_reference.py`, and the theorem artifact is in `refs/final_artifact/sparse_jl_theorem.md`.
