## Research question

Given a sparse, uniformly-random subset of the entries of an unknown rank-$r$ matrix
$M^*$, design the **matrix-recovery strategy** that reconstructs $M^*$ as accurately as possible on
the *unobserved* entries. The single thing being designed is the algorithm inside `recover(...)` — how
the recovered matrix is parameterized (depth, initialization, regularization) and how it is fit (which
update, which optimizer, which stopping rule). Everything else — how $M^*$ is sampled, how the mask is
drawn, how the estimate is scored — is fixed by the driver.

## Prior art / Background / Baselines

Low-rank matrix completion is fundamental in collaborative filtering, signal processing, and
statistics. The methods below agree that the answer should be a low-rank matrix fitting the observed
entries, and differ in how they reach it.

- **Nuclear-norm convex relaxation.** Replace the rank objective with nuclear-norm minimization
  $\|X\|_* = \sum_i \sigma_i(X)$, a convex surrogate that exactly recovers incoherent rank-$r$ matrices
  from enough entries.
- **Explicit low-rank factorization.** Write $X = UV^\top$ with a fixed inner dimension $d$ and fit the
  observed entries by gradient descent or alternating minimization.
- **Implicit regularization from depth-2 factorization.** Use a full-dimensional factorization with
  small initialization and small-step gradient descent; the dynamics bias the solution toward low rank
  without an explicit penalty.

## Fixed substrate / Code framework

The driver is frozen and must not be touched. Per top-level seed it samples one ground-truth matrix and
one initialization, then calls `strategy.recover(...)` once under a hard iteration budget.

- **Ground truth.** A random rank-$r$ matrix $M^* = (UV^\top)/\sqrt{r}$ with $U, V \sim \mathcal N(0,1)$
  i.i.d. of shape $[n, r]$, rescaled to $\|M^*\|_F = n$ (the Arora et al. 2019 construction).
- **Observations.** A uniformly random subset of the $n^2$ entries at the configured observation rate;
  the unobserved entries are zeroed before they reach the strategy, so the truth cannot be peeked at.
- **Metric.** Relative Frobenius error on the *unobserved* entries,
  $\texttt{test\_rel\_fro} = \|(\hat M - M^*)[\lnot\text{mask}]\|_F / \|M^*[\lnot\text{mask}]\|_F$,
  lower is better. A secondary `full_rel_fro` over the whole matrix is also reported.

The driver hands the strategy helpers it may use: `set_global_seed`, the masked-MSE and
relative-Frobenius utilities, and `torch` / `torch.nn`. It also emits `TRAIN_METRICS iter=...
train_mse=...` lines so the agent loop sees training progress.

## Editable interface

Exactly one region of `pytorch-examples/synth_matrix_completion/custom_strategy.py` is editable (the
`EDITABLE` block): `build_strategy()`, the `MatrixRecoveryStrategy` interface, and one or more strategy
classes. The contract is the single method

```
recover(observed_values, observed_mask, n, rank_hint, device, max_iters, log_iters) -> torch.Tensor
```

returning a full $[n, n]$ estimate of $M^*$. The driver supplies `observed_values` ($M^*$ masked to
zero off the observation set), `observed_mask` (bool $[n,n]$), `n`, `rank_hint` (the true rank — may be
used or ignored), `device`, `max_iters` (a hard budget), and `log_iters` (the suggested print
interval).

The starting point is the scaffold default shown below: a depth-3 parameterization fit by Adam on the
masked squared error.

```python
# EDITABLE region of custom_strategy.py — scaffold default fill
def build_strategy() -> "MatrixRecoveryStrategy":
    """Return the matrix-recovery strategy used by the fixed driver."""
    return DeepMatrixFactorization(depth=3, init_scale=1e-3, lr=1e-3)


class MatrixRecoveryStrategy:
    """Interface contract; subclass and implement `recover`."""

    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters) -> "torch.Tensor":
        raise NotImplementedError


class DeepMatrixFactorization(MatrixRecoveryStrategy):
    """End-to-end depth-d parameterization M = W_d ... W_1, full hidden dim."""

    def __init__(self, depth=3, init_scale=1e-3, lr=1e-3, train_thres=1e-7):
        if depth < 2:
            raise ValueError("DeepMatrixFactorization requires depth >= 2.")
        self.depth = depth
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n, device):
        layers = [nn.Linear(n, n, bias=False) for _ in range(self.depth)]
        scale = (self.init_scale ** (1.0 / self.depth)) * (n ** (-0.5))
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model):
        out = None
        for layer in model:
            out = layer.weight.t() if out is None else layer(out)
        return out

    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mask = observed_mask.to(device)
        target = observed_values.to(device)
        denom = max(int(mask.sum().item()), 1)

        for it in range(1, max_iters + 1):
            e2e = self._e2e(model)
            residual = (e2e - target) * mask
            loss = residual.pow(2).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if it == 1 or it % log_iters == 0 or it == max_iters:
                print(f"TRAIN_METRICS iter={it} train_mse={loss.item():.6e}", flush=True)
                if loss.item() <= self.train_thres:
                    break

        with torch.no_grad():
            return self._e2e(model).detach().cpu()
```

## Evaluation settings

Three configurations span the difficulty range, each scored on `test_rel_fro` (lower better):

- **rank3-50**: $n=50$, rank $3$, $30\%$ observed (easiest — generous samples relative to the rank).
- **rank5-100**: $n=100$, rank $5$, $20\%$ observed (the canonical Arora et al. 2019 setup).
- **rank10-200**: $n=200$, rank $10$, $10\%$ observed (hardest — large and severely undersampled).

The release configuration evaluates a single top-level seed (42), one ground-truth matrix and one
initialization per environment, under a per-run wall-clock cap (under 30 minutes on one H100, bounded
by `max_iters`). The task-level score is the geometric mean of normalized `test_rel_fro` terms across
the three environments.
