**Problem.** Recover a low-rank matrix from a sparse random subset of its entries. Fitting the
observations is trivial and useless — infinitely many matrices interpolate the masked entries, almost
all garbage off the observation set. The completion is decided entirely by *which* interpolant the
algorithm lands on, so the design question is the implicit bias of the recovery procedure.

**Key idea (the floor).** Do not bolt on a penalty: parameterize the estimate as a full-dimensional
depth-2 product $M = W_2 W_1^\top$ (two square bias-free linear layers, hidden dim $n$, *no* rank cap)
and descend the masked squared error on the factors from a near-zero Gaussian init. Descending on the
factors changes the dynamics of the product from the flat $\dot X = -\nabla F$ (which gives the
useless minimum-Frobenius / impute-zeros fit) to the multiplicative $\dot X = -\nabla F\,X - X\,\nabla F$
— a rich-get-richer flow that, from near-zero init, biases toward the minimum-nuclear-norm (low-rank)
solution for free.

**Why it is the weakest rung.** Depth-2 is exactly the case the implicit-bias theory says yields only
the *nuclear-norm-strength* surrogate: the per-mode growth rate $\propto(\sigma_r^2)^{1-1/N}$ is just
$|\sigma_r|$ at $N=2$, a gentle power-law gap with no saturating cap on small modes. Combined with an
adaptive optimizer and a not-tiny step (a coarse approximation to the gradient flow), this should win
where samples are generous, leave error on the canonical undersampled setup, and fail in the deeply
data-poor regime.

**Hyperparameters.** depth $=2$; `init_scale = 1e-3` (per-layer Gaussian std
$=\texttt{init\_scale}^{1/2}\cdot n^{-1/2}$, so the end-to-end product starts near zero);
optimizer Adam, `lr = 5e-3`; masked-MSE loss normalized by the number of observations; early stop at
train MSE $\le$ `1e-7`.

```python
# EDITABLE region of custom_strategy.py — step 1: shallow (depth-2) matrix factorization
def build_strategy() -> "MatrixRecoveryStrategy":
    """Return the matrix-recovery strategy used by the fixed driver."""
    return ShallowMatrixFactorization(init_scale=1e-3, lr=5e-3)


class MatrixRecoveryStrategy:
    """Interface contract; subclass and implement `recover`."""

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


class ShallowMatrixFactorization(MatrixRecoveryStrategy):
    """End-to-end depth-2 parameterization M = W2 @ W1^T, full hidden dim."""

    def __init__(
        self,
        init_scale: float = 1e-3,
        lr: float = 5e-3,
        train_thres: float = 1e-7,
    ) -> None:
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n: int, device: torch.device) -> nn.Sequential:
        layers = [nn.Linear(n, n, bias=False) for _ in range(2)]
        scale = (self.init_scale ** 0.5) * (n ** (-0.5))
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model: nn.Sequential) -> torch.Tensor:
        out = None
        for layer in model:
            out = layer.weight.t() if out is None else layer(out)
        return out

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
                print(
                    f"TRAIN_METRICS iter={it} train_mse={loss.item():.6e}",
                    flush=True,
                )
                if loss.item() <= self.train_thres:
                    break

        with torch.no_grad():
            return self._e2e(model).detach().cpu()
```
