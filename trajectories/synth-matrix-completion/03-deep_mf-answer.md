**Problem.** Both lower rungs top out at nuclear-norm strength — depth-2 approximates it (badly, via
Adam), SVT targets it (cleanly, via the convex program) — and neither beats it. But nuclear norm still
leaves a quarter of rank3-50 wrong, because minimum-nuclear-norm and minimum-rank part ways exactly in
the data-poor regime. The fix is not to approximate or target nuclear norm but to make the implicit bias
*stronger than* it.

**Key idea (deep matrix factorization).** Parameterize the estimate as the end-to-end product
$M = W_3 W_2 W_1$ of three full-dimensional bias-free linear layers (no rank cap) and descend the masked
squared error from a near-zero Gaussian init. The singular-value dynamics carry a depth factor
$N(\sigma_r^2)^{1-1/N}$ (exponent $2-2/N$): for $N=2$ this is the gentle power-law gap (nuclear-norm
strength), but for $N\ge3$ the small modes *saturate at a finite cap* while the dominant ones run off —
a hard low-rank bias that no single matrix norm captures and that beats nuclear norm where data is poor.

**Why it is the top rung.** Depth $3$ is the cheapest depth that switches the bias from power-law to
saturating ($N=4$ is indistinguishable from $N=3$). Where the convex baseline left recoverable
structure on the generously-sampled environments, the saturating bias should reclaim it — near-exact
recovery where the truth is identifiable at all.

**Diff from the depth-2 floor.** Identical scaffold, identical Adam masked-MSE loop and `_e2e` fold;
only `depth=3` (per-layer std $=\texttt{init\_scale}^{1/3}\cdot n^{-1/2}$, spreading the small scale
across three factors) and `lr=1e-3` (smaller step — three layers compose into a more sensitive product,
and the smaller step tracks the gradient flow the bias lives in). No penalty, no rank cap.

**Hyperparameters.** depth $=3$; `init_scale = 1e-3`; Adam, `lr = 1e-3`; masked-MSE normalized by the
number of observations; early stop at train MSE $\le$ `1e-7`. This is the scaffold default.

```python
# EDITABLE region of custom_strategy.py — step 3: deep (depth-3) matrix factorization
def build_strategy() -> "MatrixRecoveryStrategy":
    """Return the matrix-recovery strategy used by the fixed driver."""
    return DeepMatrixFactorization(depth=3, init_scale=1e-3, lr=1e-3)


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


class DeepMatrixFactorization(MatrixRecoveryStrategy):
    """End-to-end depth-d parameterization M = W_d ... W_1, full hidden dim."""

    def __init__(
        self,
        depth: int = 3,
        init_scale: float = 1e-3,
        lr: float = 1e-3,
        train_thres: float = 1e-7,
    ) -> None:
        if depth < 2:
            raise ValueError("DeepMatrixFactorization requires depth >= 2.")
        self.depth = depth
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n: int, device: torch.device) -> nn.Sequential:
        layers = [nn.Linear(n, n, bias=False) for _ in range(self.depth)]
        scale = (self.init_scale ** (1.0 / self.depth)) * (n ** (-0.5))
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
