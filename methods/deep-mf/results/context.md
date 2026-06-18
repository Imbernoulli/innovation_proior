# Context: low-rank completion and implicit bias before the answer

## Research question

Given only a small, randomly chosen subset of the entries of an unknown matrix `W* in R^{d x d'}`,
recover the missing entries. Fitting the observed entries is underdetermined: the constraints define a
large affine set of interpolating matrices, and most interpolants need not match the unseen entries.
The useful prior is that `W*` is low-rank, but the desired mechanism is not an explicit rank cap or an
added penalty. The question is how to parameterize the completion so that ordinary gradient-based
optimization, started near the origin and run to small training error, implicitly selects a low-rank
interpolant.

This is a deliberately clean test bed for the broader implicit-regularization problem in deep learning.
Each matrix entry can be viewed as a regression example; observed entries are training data; recovery
quality measures generalization to entries not used for fitting. The empty slot is the parameterization
and initialization of the reconstruction matrix.

## Background

The standard convex surrogate for low rank is the nuclear norm
`||W||_* = sum_r sigma_r(W)`, the sum of singular values. Under incoherence and enough observations,
minimum-nuclear-norm completion can recover a low-rank matrix exactly. This makes nuclear norm the
natural reference point: it is tractable, convex, and tightly connected to rank, but it is still only a
surrogate.

The non-convex neural version begins with a full-dimensional factorization `W = W_2 W_1` and squared
loss on the observed entries. If the hidden dimension is capped, rank is constrained explicitly; the
interesting case leaves the hidden dimension full and relies on optimization. Prior work observed that
small-step gradient descent from near-zero initialization tends to produce low-rank completions even
without a rank cap. Gunasekar et al. formalized the leading conjecture: in the small-initialization,
small-step limit, full-dimensional depth-2 factorization behaves like minimum nuclear norm; in a
commuting symmetric positive-semidefinite sensing setting, that statement can be proved.

A separate ingredient is the end-to-end dynamics of linear networks of arbitrary depth. For a product
`W = W_N W_{N-1} ... W_1`, gradient flow from a balanced initialization
`W_{j+1}^T W_{j+1} = W_j W_j^T` makes the product matrix obey

```text
dot W = - sum_{j=1}^N [W W^T]^((j-1)/N) grad ell(W) [W^T W]^((N-j)/N).
```

This formula is a pre-existing tool, not yet a completion algorithm. It says that factorization changes
the product-space dynamics by preconditioning the gradient in a way that depends on the current singular
spectrum of `W`. Balancedness holds exactly for identity-style initialization and is a good approximation
for small random initialization.

## Baselines

**Minimum nuclear norm.** Solve `min ||W||_*` subject to matching the observed entries, or a softened
version of the same objective. This is convex and has strong recovery guarantees in the well-sampled
regime. Its limitation is that, when observations are too scarce, the nuclear-norm interpolant can differ
from the true low-rank completion.

**Full-dimensional depth-2 factorization.** Optimize `W = W_2 W_1` on observed-entry squared loss, using
small initialization and small steps. This is the shallow neural-network baseline. It is empirically
low-rank-biased and theoretically tied to nuclear norm in restricted settings, but the mechanism remains
ambiguous outside those settings.

**Direct optimization over entries.** Treat `W` itself as the parameter and run gradient descent on the
observed-entry squared loss. In an underdetermined problem this has no structural preference for a
low-rank interpolant; it is the no-factorization control.

## Evaluation settings

The synthetic completion task uses a random rank-`r` matrix, generated as `U V^T` with i.i.d. Gaussian
factors and, in the canonical code, normalized so `||W*||_F = n` for square `n x n` instances. Observed
entries are sampled uniformly without replacement. Representative settings include `100 x 100` matrices
with ranks `5` and `10`, sweeping the number of observed entries; matrix sensing replaces observed
entries by random Gaussian linear measurements.

The training loss is the mean squared error on the observed entries or provided projections. Recovery is
reported as normalized Frobenius error against the ground-truth matrix, with nuclear norm and effective
rank used as diagnostics for the returned singular spectrum. MovieLens-100K provides a non-synthetic
check by fitting a uniformly sampled subset of observed ratings from the `943 x 1682` user-movie matrix.

The protocol holds the parameterization full-dimensional unless a baseline explicitly imposes rank. It
sweeps small learning rates and small zero-centered initialization scales, runs to a tiny observed-entry
training loss or an iteration budget, and compares the returned product matrix to the convex and direct
optimization controls.

## Code framework

The recovery harness supplies observed values, an observed-entry mask, and a budget. The unresolved part
is the model returned by `_build`, the map from model parameters to the completed matrix, and the optimizer
appropriate for the chosen parameterization.

```python
import torch
import torch.nn as nn


class MatrixRecoveryStrategy:
    def recover(
        self,
        observed_values: torch.Tensor,
        observed_mask: torch.Tensor,
        n: int,
        rank_hint: int,
        device: torch.device,
        max_iters: int,
        log_iters: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class Strategy(MatrixRecoveryStrategy):
    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def _build(self, n: int, device: torch.device):
        # Choose the trainable parameterization and its initialization.
        raise NotImplementedError

    def _reconstruct(self, model) -> torch.Tensor:
        # Map trainable parameters to the full [n, n] completed matrix.
        raise NotImplementedError

    def _optimizer(self, model):
        # Choose the optimizer used to fit observed entries.
        raise NotImplementedError

    def recover(self, observed_values, observed_mask, n, rank_hint, device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = self._optimizer(model)
        mask = observed_mask.to(device)
        target = observed_values.to(device)
        denom = max(int(mask.sum().item()), 1)

        for _ in range(max_iters):
            recon = self._reconstruct(model)
            loss = ((recon - target) * mask).pow(2).sum() / denom
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            return self._reconstruct(model).detach().cpu()
```
