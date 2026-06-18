# Singular Value Thresholding (SVT), distilled

SVT is a first-order, low-memory algorithm for nuclear-norm matrix completion: recover a
low-rank matrix `M` from a few observed entries by iterating two cheap steps — soft-threshold
the singular values of a running matrix, then add back the data residual on the observed set.
It is Uzawa's dual-ascent method applied to a proximal surrogate of the nuclear-norm program,
and it scales to matrices with hundreds of millions of entries because the working matrix stays
sparse and the iterates stay low rank.

## Problem it solves

Given observations `P_Omega(M)` of an unknown (approximately) low-rank `M ∈ R^{n1×n2}` on a
random set `Omega` of `m << n1 n2` entries, solve the convex program

```
minimize    ||X||_*          (nuclear norm = sum of singular values)
subject to  P_Omega(X) = P_Omega(M),
```

at a scale where interior-point SDP solvers fail (they top out near `n = 100`, form huge Newton
systems, and ignore the low rank of the solution).

## Key idea

Work with a *proximal* version of the program,

```
minimize    f_tau(X) = tau||X||_* + (1/2)||X||_F^2
subject to  P_Omega(X) = P_Omega(M),
```

and solve it by Uzawa / dual ascent. The Lagrangian is `L(X,Y) = f_tau(X) + <Y, P_Omega(M-X)>`.
Alternating the inner minimization over `X` with a dual step on `Y` gives

```
X^k = D_tau(Y^{k-1}),
Y^k = Y^{k-1} + delta_k * P_Omega(M - X^k),     Y^0 = 0,
```

where `D_tau` is the **singular value shrinkage operator**: for `Y = U Σ V^*`,

```
D_tau(Y) = U diag((sigma_i - tau)_+) V^*,        t_+ = max(t, 0).
```

Two facts make this both principled and cheap:

- **`D_tau` is the proximity operator of the nuclear norm**:
  `D_tau(Y) = argmin_X { (1/2)||X - Y||_F^2 + tau||X||_* }`. So the inner step is exact, not a
  heuristic. (Proof: check `Y - D_tau(Y) ∈ tau ∂||D_tau(Y)||_*` using
  `∂||X||_* = {UV^* + W : U^*W = 0, WV = 0, ||W||_2 <= 1}`.)
- **Sparsity + low rank.** Since `Y^0 = 0` and the update only adds a residual supported on
  `Omega`, every `Y^k` is sparse (`m` nonzeros). Large `tau` makes the shrink kill most singular
  values, so each `X^k` is low rank — only the singular triplets above `tau` are needed, via a
  partial (Lanczos) SVD. Storage and per-iteration cost stay far below `n1 n2`.

As `tau -> ∞`, the solution of the proximal program converges to the minimum-Frobenius-norm
nuclear-norm minimizer (`X_tau -> X_∞`), so a large `tau` recovers the genuine nuclear-norm
solution — and large `tau` is also exactly what keeps the iterates low rank and sparse.

## Convergence and parameters

**Convergence.** `f_tau` is strongly convex: for `Z ∈ ∂f_tau(X)`, `Z' ∈ ∂f_tau(X')`,
`<Z - Z', X - X'> >= ||X - X'||_F^2`. With `r_k = ||P_Omega(Y^k - Y^*)||_F`,

```
r_k^2 <= r_{k-1}^2 - (2 delta_k - delta_k^2) ||X^k - X^*||_F^2,
```

so if `0 < inf delta_k <= sup delta_k < 2` then `r_k` is nonincreasing and
`||X^k - X^*||_F -> 0`. The iterates converge to the unique solution of the proximal program.

**Step size.** `delta < 2` is safe but slow. Using the near-isometry
`||P_Omega(A)||_F^2 ≈ p ||A||_F^2` (`p = m/(n1 n2)`), a much faster practical choice is

```
delta = 1.2 / p = 1.2 * n1 n2 / m.
```

(Heuristic: `X^* - X^k` depends on `Omega`, so the near-isometry is not rigorous here, but it
converges empirically and takes far larger steps.)

**Threshold.** Pick `tau` large enough that `tau||X||_*` dominates `(1/2)||X||_F^2`. For
square `n×n` matrices generated as `M = M_L M_R^*` with Gaussian factors,
`||M||_F ≈ n√r` and `||M||_* ≈ n r`, so
`tau||M||_* / ((1/2)||M||_F^2) ≈ 2 tau/n`; setting this ≈ 10 gives

```
tau = 5 n.
```

For a rectangular `n1×n2` implementation, the same calibration is usually written
`tau = 5 sqrt(n1 n2)`; this reduces to `5n` in the square case.

**Warm start.** While `k delta ||P_Omega(M)||_2 <= tau`, the shrink yields `X^k = 0` and
`Y^k = k delta P_Omega(M)`. Skip those trivial steps: define `k_0` by
`tau / (delta ||P_Omega(M)||_2) ∈ (k_0 - 1, k_0]` and start at `Y^0 = k_0 delta P_Omega(M)`.

**Stopping.** From the KKT conditions (`X = D_tau(Y)` holds by construction; the remaining
condition is `P_Omega(X - M) = 0`), stop when the relative residual on the observed set is small:

```
||P_Omega(X^k - M)||_F / ||P_Omega(M)||_F <= epsilon   (e.g. 1e-4).
```

By the near-isometry this tracks the true relative reconstruction error
`||X^k - M||_F / ||M||_F`.

## Algorithm

```
Input: P_Omega(M), step delta, threshold tau, tolerance eps, increment l, max iters k_max
Set Y^0 = k_0 * delta * P_Omega(M)    (k_0 from the warm-start rule); r_0 = 0
for k = 1, 2, ... , k_max:
    compute the singular triplets of Y^{k-1} with sigma > tau   (partial SVD)
    X^k = sum_{sigma_j > tau} (sigma_j - tau) u_j v_j^*          (= D_tau(Y^{k-1}))
    if ||P_Omega(X^k - M)||_F / ||P_Omega(M)||_F <= eps: break
    Y^k = Y^{k-1} + delta * P_Omega(M - X^k)                     (sparse, supported on Omega)
output X^k
```

## Working code

Filling the `recover` slot of the strategy harness, with a dense SVD (fine for moderate `n`;
swap in a partial Lanczos SVD for large scale):

```python
import math
import torch


class NuclearNormSVT:
    """Singular Value Thresholding for matrix completion."""

    def __init__(self, tau_factor=5.0, delta_factor=1.2, tol=1e-4):
        self.tau_factor = float(tau_factor)      # tau = tau_factor * n   (=> 5n)
        self.delta_factor = float(delta_factor)  # delta = delta_factor / p  (=> 1.2/p)
        self.tol = float(tol)                    # relative-residual stop on Omega

    @torch.no_grad()
    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        mask = observed_mask.to(device).to(torch.float32)
        M_obs = observed_values.to(device).to(torch.float32)      # P_Omega(M)
        n_observed = max(int(mask.sum().item()), 1)
        p = n_observed / float(n * n)                             # observation ratio

        tau = self.tau_factor * float(n)                          # tau = 5n
        delta = self.delta_factor / max(p, 1e-6)                  # delta = 1.2 / p

        # Warm start: jump over the trivial steps where the shrink kills everything.
        norm_proj_m = float(torch.linalg.matrix_norm(M_obs, ord=2).item())
        k0 = max(1, math.ceil(tau / (delta * max(norm_proj_m, 1e-6))))
        Y = (k0 * delta) * M_obs                                  # Y^0 = k_0 * delta * P_Omega(M)

        X = torch.zeros_like(M_obs)
        norm_obs = max(float(M_obs.norm().item()), 1e-6)
        log_every = max(int(log_iters), 1)
        for it in range(1, max_iters + 1):
            # X^k = D_tau(Y^{k-1}): soft-threshold the singular values.
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            S_thresh = torch.clamp(S - tau, min=0.0)             # (sigma_i - tau)_+
            X = (U * S_thresh) @ Vh

            # Stop on the same relative residual used by Algorithm 1 and SVT.m.
            residual = (M_obs - X) * mask
            rel_res = float(residual.norm().item() / norm_obs)
            train_mse = float(residual.pow(2).sum().item() / float(n_observed))
            if it == 1 or it % log_every == 0 or it == max_iters or rel_res <= self.tol:
                print(
                    f"TRAIN_METRICS iter={it} rel_res={rel_res:.6e} "
                    f"train_mse={train_mse:.6e}",
                    flush=True,
                )
            if rel_res <= self.tol:
                break

            # Dual / residual update, supported on Omega.
            Y = Y + delta * residual

        return X.detach().cpu()


def build_strategy():
    return NuclearNormSVT(tau_factor=5.0, delta_factor=1.2, tol=1e-4)
```

For large matrices, replace the dense `torch.linalg.svd` with a partial SVD (Lanczos
bidiagonalization, e.g. PROPACK) that returns only the singular triplets above `tau`, requesting
`s_k = rank(X^{k-1}) + 1` values and incrementing until one falls below `tau`; the sparse `Y^k`
multiplies vectors cheaply, keeping each iteration far below `O(n^3)`.
