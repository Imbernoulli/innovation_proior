# Context: recovering an underdetermined matrix without choosing the fit by hand

## Research question

We are handed a small set of linear measurements of an unknown matrix and asked to fill in the
rest. Write the recovery problem as a quadratic least-squares fit over a matrix
`X ∈ R^{n×n}`,

```text
min_X  F(X) = ||A(X) - y||_2^2,
```

where `A : R^{n×n} -> R^m` is linear, `A(X)_i = <A_i, X>`, and `y ∈ R^m`
contains the observations. Matrix completion is the special case where each measurement reads a
single entry, so `y` is the list of observed entries.

The hard regime is `m << n^2`. Then `A(X) = y` is badly underdetermined: an affine family of
matrices fits the data exactly. For completion, setting all unobserved entries to zero is already
a zero-training-error solution. The core question is which exact fit an algorithm selects and
whether that selection generalizes to the unobserved entries.

## Background

The classical structural assumption for matrix recovery is low rank. A rank-`r` matrix has only
`O(nr)` degrees of freedom rather than `n^2`, so a small number of measurements can be enough in
principle. Direct rank minimization,

```text
min rank(X)  subject to A(X) = y,
```

is NP-hard in general, so the standard convex replacement is the nuclear norm
`||X||_* = sum_k sigma_k(X)`. Recht, Fazel, and Parrilo establish nuclear-norm minimization as a
convex surrogate for affine rank minimization, including the compressed-sensing analogy and RIP-
style recovery conditions. Candes and Recht show that, under random sampling and incoherence,
minimum-nuclear-norm matrix completion can exactly recover many low-rank matrices from far fewer
than `n^2` entries. Srebro-Shraibman and related generalization results explain why a low nuclear
norm is also a useful prediction complexity measure.

Another line of work factorizes a positive semidefinite matrix as `X = U U^T`, with
`U ∈ R^{n×d}`. This is the Burer-Monteiro route: it replaces a constrained problem over `X` by
a smooth non-convex problem over factors. If `d < n`, the factorization imposes a hard rank cap.
If `d` is large enough, no-spurious-local-minimum results such as Journée et al. show that local
minima of the factorized problem correspond to global minima of the original least-squares
problem, and Lee et al. explain why gradient descent avoids strict saddles almost surely.

## Baselines

**Explicit nuclear-norm minimization.** Solve

```text
min_X ||X||_*  subject to A(X) = y
```

or a softened version of the same convex program. This gives the reference low-rank-promoting
solution and has the cleanest recovery theory.

**Gradient descent on `X` directly.** Treat `F(X) = ||A(X)-y||_2^2` as a convex quadratic and
run gradient descent in matrix space from `X = 0`. The gradient is always
`A*(A(X)-y)`, a linear combination of the measurement matrices, so the iterates stay in
`L = {A*(s) : s ∈ R^m}`. If the run reaches zero error, the KKT conditions for

```text
min_X ||X||_F^2  subject to A(X) = y
```

are satisfied: `A(X) = y` and `X = A*(ν)`. Thus this baseline selects the minimum-Frobenius-norm
fit.

**Low-rank factorized descent.** Choose a small `d`, optimize
`min_U ||A(UU^T)-y||_2^2`, and rely on `rank(UU^T) <= d`. This can be effective when the true
rank is known or well tuned.

## Evaluation settings

The natural evaluation is to compare both reconstruction quality and the complexity of the
selected fit.

For planted matrix sensing, generate random symmetric Gaussian measurement matrices, choose a
positive semidefinite `X*`, and set `y = A(X*)`. Vary the planted spectrum: exactly low rank,
smoothly decaying spectrum, and a non-reconstructible regime with too few measurements. Track
relative reconstruction error `||X-X*||_F / ||X*||_F`, training loss, and `||X||_*`.

For matrix completion, sample entries from `X*` either uniformly or under skewed sampling, then
fit only those entries and evaluate both the held-out entries and the nuclear norm of the
completion. A real recommendation dataset such as MovieLens supplies the same kind of masked
least-squares problem without a synthetic ground-truth matrix.

Useful reference lines are the explicit minimum-nuclear-norm solution, direct gradient descent on
`X`, and low-rank constrained factorizations at several values of `d`. Sweeping initialization
scale and step-size policy is important because the optimizer, not only the loss surface,
determines which point in the feasible set is selected.

## Code framework

The scaffold is a PyTorch recovery interface: it provides observed values, a Boolean observation
mask, the matrix size, a rank hint that a method may ignore, a device, and an iteration budget.
The method must return a dense recovered matrix.

```python
import torch
import torch.nn as nn


class MatrixRecoveryStrategy:
    """Interface contract; subclass and implement recover."""

    def recover(
        self,
        observed_values: torch.Tensor,   # observed entries, zero elsewhere, [n, n]
        observed_mask: torch.Tensor,     # bool [n, n], True on observed entries
        n: int,
        rank_hint: int,
        device: torch.device,
        max_iters: int,
        log_iters: int,
    ) -> torch.Tensor:                   # recovered [n, n]
        raise NotImplementedError


class Strategy(MatrixRecoveryStrategy):
    # TODO: choose a parameterization, initialization, optimizer, stopping rule,
    # and returned end-to-end matrix using only the observed-entry loss.
    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        pass


def masked_mse(estimate: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    residual = (estimate - target) * mask
    denom = max(int(mask.sum().item()), 1)
    return residual.pow(2).sum() / denom


def build_strategy() -> MatrixRecoveryStrategy:
    # TODO: return the completed recovery strategy.
    pass
```
