# Context: large-scale low-rank matrix recovery (circa 2008-2010)

## Research question

We observe a small subset of the entries of an unknown matrix `M` and want to fill in the
rest. Concretely, `M ∈ R^{n1×n2}` is observed only on a set `Omega` of `m` index pairs, with
`m` far smaller than `n1·n2`, and we must produce a complete matrix that agrees with the data
and is "right" on the unseen entries. With fewer samples than unknowns this is hopelessly
ill-posed in general — infinitely many completions fit the data — so something must pin down
the answer. The thing that does is a structural assumption: the matrix of interest has low
rank (or is close to low rank). Recommender systems (the Netflix problem), structure-from-
motion in vision, system identification in control, and multi-class learning all produce
matrices believed to be governed by a few latent factors, hence (approximately) low rank.

The computational problem is to recover the low-rank `M` by solving this convex program
at a scale that matters in practice: matrices with `n` in the thousands to tens of thousands,
hundreds of millions to a billion entries, of which only a fraction of a percent are observed.

## Background

The structural fact that makes recovery possible is that, under an incoherence assumption on
`M`'s singular vectors (its row and column spaces are not aligned with the canonical basis),
most rank-`r` matrices are determined by a number of random entries that is small relative to
`n^2`. Candès and Recht (2008, arXiv:0805.4471) proved that if the number of observed entries
obeys `m >= C n^{6/5} r log n`, then with high probability `M` is the *unique* matrix of
minimum nuclear norm consistent with the data — so a convex program recovers it exactly. The
nuclear norm `||X||_* = sum_i sigma_i(X)` (the sum of singular values, i.e. the `l1` norm of
the spectrum) is the natural convex surrogate for rank: Fazel (2002) showed it is the convex
envelope of `rank(X)` over the spectral-norm ball `{||X||_2 <= 1}`, the tightest convex
relaxation of the NP-hard rank-minimization problem, and that it can be cast as a semidefinite
program. So the object to solve is the convex program

```
minimize    ||X||_*
subject to  P_Omega(X) = P_Omega(M),
```

where `P_Omega` is the orthogonal projection that keeps entries in `Omega` and zeros the rest.

Several pieces of machinery sit ready to be used.

**The subdifferential of the nuclear norm.** For `X = U Σ V^*` in reduced SVD form,

```
∂||X||_* = { U V^* + W : U^*W = 0, W V = 0, ||W||_2 <= 1 },
```

with two facts that follow immediately: every subgradient `Z0` obeys `||Z0||_2 <= 1`, and
`<Z0, X> = ||X||_*` (the nuclear and spectral norms are dual). This is the matrix analogue of
the `l1` subgradient `sign(x)`.

**Soft-thresholding as a proximal operator.** In compressed sensing and image processing the
recurring primitive is the scalar/vector soft-thresholding rule
`S_tau(x) = sign(x)·(|x| - tau)_+`, where `t_+ = max(t,0)`. It is exactly the proximity
operator of the `l1` norm: `S_tau(x) = argmin_u { (1/2)(u - x)^2 + tau|u| }`, obtained by
solving `0 ∈ u - x + tau ∂|u|` case by case. Iterative soft-thresholding schemes and
*linearized Bregman* iterations (Osher, Yin, Goldfarb and collaborators, 2007-2008) use such
thresholding sweeps to find the minimum-`l1` solution of an underdetermined linear system, and
the closely related proximal forward-backward / iterative shrinkage methods (Combettes & Wajs
2005) solve unconstrained problems of the form `lambda||x||_1 + (1/2)||Ax - b||^2` via the
fixed point `x = S_{lambda δ}(x + δ A^*(b - Ax))`.

**Lagrange duality and Uzawa's method.** For a convex program with equality constraints,
strong duality identifies the solution with a saddle point of the Lagrangian, and Uzawa's
algorithm (Arrow-Hurwicz-Uzawa 1958) reaches it by alternating an inner minimization over the
primal variable with a step that moves the dual variable along a (sub)gradient of the dual
function. This is the standard frame for turning a constrained problem into a sequence of
cheap updates.

**A near-isometry of the sampling operator.** Under the same incoherence assumption, for a
*fixed* matrix `A` and a uniformly random `Omega` of size `m`, with high probability

```
(1 - ε) p ||A||_F^2 <= ||P_Omega(A)||_F^2 <= (1 + ε) p ||A||_F^2,   p := m / (n1 n2),
```

so the energy of `A` restricted to the observed entries is about `p` times its total energy.
This is the matrix-completion analogue of the restricted-isometry property from compressed
sensing.

**Random-matrix scale facts.** For `M = M_L M_R^*` with `M_L, M_R` having i.i.d. Gaussian
entries (the standard synthetic generator), the Frobenius norm concentrates near `n√r` and the
nuclear norm near `n r`, facts used to calibrate any scale-dependent parameters.

At the time, the convex program above was solved with general-purpose interior-point SDP
solvers (SDPT3, SeDuMi), which form and solve large linear systems for the Newton direction
at every step and are accurate on small problems.

## Baselines

**Interior-point semidefinite programming (SDPT3 / SeDuMi).** Recast the nuclear-norm program
as an SDP and solve with a primal-dual interior-point method. Core idea: follow the central
path by taking Newton steps on a barrier-augmented system.

**Iterative soft-thresholding / proximal forward-backward splitting (PFBS) for the unconstrained
relaxation** (Combettes & Wajs 2005; and the `l1` imaging literature). Relax the hard equality
constraint into a penalty,

```
minimize  lambda ||X||_* + (1/2) ||P_Omega(X) - P_Omega(M)||_F^2,
```

and iterate the proximal fixed point. Written with an intermediate matrix this is

```
X^k = D_{lambda δ_{k-1}}(Y^{k-1}),
Y^k = X^k + δ_k P_Omega(M - X^k),
```

where `D` soft-thresholds singular values.

**Linearized Bregman iteration for `l1` recovery** (Osher, Yin, Goldfarb, Darbon et al.,
2007-2008). For the vector problem `min ||x||_1 s.t. Ax = b`, alternate a soft-thresholding of
an auxiliary variable with a residual update; provably solves a quadratically-perturbed `l1`
problem.

**Explicit nonconvex factorization / alternating minimization.** Parameterize `X = A B^*` with
`A, B` of fixed rank and minimize the squared error on observed entries by alternating least
squares. Cheap per step and naturally low rank.

## Evaluation settings

The natural yardsticks already in use:

- **Synthetic low-rank recovery.** Generate `M = M_L M_R^*` with `M_L, M_R ∈ R^{n×r}` having
  i.i.d. Gaussian entries; sample `Omega` uniformly at random at observation ratio
  `p = m/(n1 n2)`. Sweep `n` (50 up to tens of thousands), rank `r`, and `p`. The "degrees of
  freedom ratio" `m / (r(2n - r))` and the sampling ratio relative to `n` characterize
  difficulty.
- **Metric.** Relative recovery error in Frobenius norm,
  `||X_out - M||_F / ||M||_F`, reported on the full matrix (equivalently, on the unobserved
  entries, since the observed ones are nearly fit). Also tracked: the rank of the output,
  number of iterations, wall-clock time, and storage.
- **Protocol.** Fixed step size `δ_k = δ` across iterations; a relative-residual stopping rule
  on the observed entries; a fixed threshold parameter; a maximum iteration cap. Recovery is
  declared when the relative error falls below a small tolerance (e.g. `1e-4`).
- Scale targets range from desktop-size (`1000×1000`) up to nearly a billion entries
  (`30000×30000`, rank ~10, ~0.4% observed).

## Code framework

The solver plugs into a fixed driver that hands it the observed entries and a budget, and
expects back a full `[n, n]` matrix. Nothing about the recovery rule itself is settled —
that rule is exactly what is to be designed — so the substrate is only the generic pieces that
already exist: the masked-observation data, dense linear algebra (an SVD routine), and the
projection onto the observed set. The single empty slot is the iteration that turns the masked
data into a completed low-rank matrix.

```python
import torch


class MatrixRecoveryStrategy:
    """Interface the fixed driver calls. Subclass and implement `recover`."""

    def recover(
        self,
        observed_values: torch.Tensor,   # M masked to zero off Omega, shape [n, n]
        observed_mask: torch.Tensor,     # bool [n, n], True on the observed set Omega
        n: int,
        rank_hint: int,                  # true rank (may be used or ignored)
        device: torch.device,
        max_iters: int,
        log_iters: int,
    ) -> torch.Tensor:                   # return completed matrix, shape [n, n]
        raise NotImplementedError


class Strategy(MatrixRecoveryStrategy):
    """Generic recovery strategy. Has access to the masked observations, the
    projection onto the observed set, and a dense SVD. Must stay cheap: exploit
    sparsity of the observations and (expected) low rank of the answer."""

    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        mask = observed_mask.to(device).to(torch.float32)
        M_obs = observed_values.to(device).to(torch.float32)   # P_Omega(M)
        p = float(mask.sum().item()) / float(n * n)            # observation ratio

        X = torch.zeros_like(M_obs)
        for it in range(1, max_iters + 1):
            # TODO: the recovery iteration we will design.
            #       From the masked observations M_obs (and any state we keep),
            #       compute the next completed estimate X and decide when to stop.
            pass
        return X.detach().cpu()


def build_strategy() -> "MatrixRecoveryStrategy":
    """Return the recovery strategy the fixed driver runs."""
    return Strategy()
```

The driver supplies the masked observations and the budget; `recover` is where the iteration
that completes the matrix will live.
