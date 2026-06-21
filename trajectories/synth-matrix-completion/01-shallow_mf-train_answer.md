I am handed a tiny uniformly-random subset of the entries of a rank-$r$ matrix $M^\*$ and asked for the rest, scored on the entries I never saw. The first honest thing to say is that fitting the data is trivial and useless: there are $n^2$ unknowns and far fewer observations, so the masked squared error hits exactly zero on an enormous family of interpolants — I could set every unobserved entry to zero and have a perfectly valid zero-training-loss "solution" that recovers nothing. The question was never "can I fit"; it is *which* fitting matrix I land on, and that is decided entirely by the algorithm inside `recover`, not by the objective. What makes the problem well-posed is the one assumption the driver hands me — $M^\*$ is genuinely low rank, $M^\* = (UV^\top)/\sqrt r$ rescaled to $\|M^\*\|_F = n$ — so the good completion is the low-rank one, and I even get `rank_hint = r`. The classical move is to put that bias in by hand: relax rank to its convex envelope, the nuclear norm $\|X\|_\* = \sum_i \sigma_i(X)$, and minimize it subject to the observations. I will do that on a later rung. For the floor I want the cheaper, stranger phenomenon — generalization *without* any explicit regularizer — so I deliberately reach for the method that uses no penalty and no rank cap, and let it tell me how far that gets me.

The reason I cannot just descend on the estimate $X$ directly is worth stating, because it sets up everything. The objective $F(X) = \|(X - M^\*)\odot\text{mask}\|_F^2$ is convex with gradient supported only on the observed entries, so from $X = 0$ every iterate stays in the $m$-dimensional span of the observation-indicator directions and the limit is the *minimum-Frobenius-norm* fit — and the minimum-Frobenius completion of a masked matrix is exactly the impute-zeros matrix. Frobenius norm is the wrong complexity measure here. So whatever biases the search toward low rank cannot be descent on $X$; I have to change *which* norm gets implicitly minimized, and the lever that does it is the parameterization.

I propose a *shallow (depth-2) matrix factorization* with no explicit regularizer. Instead of optimizing $X$, write the estimate as a product of two square, bias-free linear layers,
$$X = W_2 W_1^\top,$$
with the hidden dimension taken to be the *full* $n$, and descend the masked squared error on the factors $W_1, W_2$ from a near-zero Gaussian initialization. Full hidden dimension is the crucial choice: the factorization imposes *no* explicit rank constraint — $W_2 W_1^\top$ can represent any $[n,n]$ matrix, so I am back to the same underdetermined family of fits — and yet the *dynamics* of descending on the factors are different from descending on $X$. Track what gradient flow on the factors does to the product. In the symmetric lift $X = UU^\top$ (to which the asymmetric $W_2 W_1^\top$ reduces), the flow on the product is not the flat slide $\dot X = -\nabla F(X)$ that gives min-Frobenius, but
$$\dot X = -\nabla F(X)\,X - X\,\nabla F(X),$$
the same residual direction now multiplied by the current $X$ on both sides. That multiplicative $X$ factor is everything: the velocity is suppressed wherever $X$ is small, so starting near zero the flow can barely move in directions $X$ has not already grown into. This is a self-reinforcing, rich-get-richer growth that prefers a few dominant directions — low rank — and it is a property of the flow, not of a penalty I added.

To see that this lands specifically on the *nuclear-norm* fit, integrate the flow in the solvable case — commuting measurements, init $\alpha I$, $\alpha\to0$. The solution has the form $X_t = \exp(s_t A)\,X_0\,\exp(s_t A)$ with $s_t$ the integrated residual; the two-sided exponential is a power-iteration-like amplifier, and to reach a finite-scale fit from a vanishing start it must drive $|s_\infty|\to\infty$, at which point $\exp(s_\infty A)$ collapses onto the top eigenspace of $A$. Reading the limit eigenvalue by eigenvalue gives exactly the complementary-slackness conditions of the minimum-nuclear-norm SDP — on every direction in the support the rescaled dual eigenvalue is $1$, off the support it stays below $1$ — so both nuclear-norm KKT conditions hold and the limit minimizes $\|X\|_\*$. The single-entry indicators of completion do not commute, so this is a theorem in the commuting case and a well-motivated conjecture in general, but the mechanism I am betting on — small init, amplifier collapses onto the top spectrum — is exactly what carries over.

That argument also pins down why each ingredient must be what it is, and the harness exposes a knob for each. I factorize at all because descending on $X$ gives min-Frobenius (impute-zeros) and factorizing swaps the implicit norm to the rank-promoting nuclear norm. I take the full hidden dimension $d=n$ so the low-rank preference comes entirely from the dynamics rather than from a hard cap I would have to know the rank to set. I initialize close to zero because the theorem is a statement about $\lim_{\alpha\to0}$ — reaching a finite-scale fit from a vanishing start is precisely what forces the amplifier onto the top eigenspace; start large and the depth-2 flow behaves like flat descent on $X$, a generic high-nuclear-norm optimum. And I take small steps because the selecting manifold is *curved* — its tangent space differs at every point — so only near-infinitesimal steps stay on it; momentum and large steps walk off. In the fill the per-layer Gaussian std is $\texttt{init\_scale}^{1/2}\cdot n^{-1/2}$: the $1/\text{depth} = 1/2$ power spreads the small overall scale across the two multiplied factors so the *end-to-end* product starts a hair above zero, and the $n^{-1/2}$ is fan-in normalization so the product's magnitude is governed by `init_scale` and not by $n$. The end-to-end product is formed by `_e2e`, folding the layers left to right — the first layer contributes its weight transposed, $W_1^\top$, then layer two is applied, giving $W_2 W_1^\top$.

Two harness specifics depart from the textbook recipe and bound what I should expect. First, the optimizer is Adam, not vanilla small-step gradient descent, with `lr = 5e-3` — larger than the `1e-3` a deeper default would use. Adam's per-coordinate adaptivity is *not* the gradient flow the implicit-bias theory is built on; it does the fitting, but the bias lives in the parameterization plus the near-zero init, and Adam with a not-tiny step is a coarse approximation to that flow. Second, and more fundamentally, depth-2 is *exactly* the case the theory says yields only nuclear-norm strength. The singular-value dynamics make this concrete: for a depth-$N$ factorization the per-mode growth rate carries a factor $(\sigma_r^2)^{1-1/N}$, which at $N=2$ is just $|\sigma_r|$ — a gentle power-law gap between leading and trailing singular values, *some* low-rank bias but no saturating cap on the small modes the way deeper factorizations get. So I should expect this rung to behave like a nuclear-norm-strength recovery at best, coarsened further by Adam and the larger step. Concretely: a clean recovery where samples are generous relative to the rank (rank3-50, $30\%$ of a rank-3 matrix), real residual error on the canonical undersampled rank5-100 where bias *strength* starts to matter, and near-total failure on rank10-200, which sits right at the information floor. The floor's whole job is to fix exactly how much the bare, penalty-free depth-2 fill buys, so the later rungs have a number to beat.

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
