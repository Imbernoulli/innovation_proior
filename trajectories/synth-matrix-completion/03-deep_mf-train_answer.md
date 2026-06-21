The convex baseline answered its question sharply. On rank3-50 explicit nuclear-norm minimization landed `test_rel_fro = 0.250` against depth-2's `0.452`, and on rank5-100 `0.311` against `0.596` — decisive wins, confirming that depth-2's deficit on the well-sampled environments was a *bias-strength* problem, not a data problem. But look at what nuclear norm could *not* do: on rank3-50 it still leaves `0.250`, a quarter of the matrix wrong, when $30\%$ of a rank-3 matrix is plenty of information to recover it nearly exactly. So both lower rungs top out at nuclear-norm strength — depth-2 approximates it badly via Adam, SVT targets it cleanly via the convex program — and neither beats it, because neither tries to. Minimum-nuclear-norm and minimum-rank *part ways* exactly in the data-poor regime that matters, and the convex surrogate leaves recoverable structure on the table. The natural next move is the direction I deferred: do not approximate or target nuclear norm — make the implicit bias *stronger than* it. The lever is depth.

I propose **deep (depth-3) matrix factorization**: parameterize the estimate as the end-to-end product $W = W_3 W_2 W_1$ of three full-dimensional bias-free linear layers — no rank cap — and descend the masked squared error from a near-zero Gaussian init. The first tempting story is that if depth-2 corresponds to nuclear norm (Schatten-1), then depth-$N$ corresponds to Schatten-$p(N)$ for some $p$ shrinking toward $0$ as $N$ grows, since $\|X\|_{S_p}^p = \sum_r \sigma_r^p \to \text{rank}(X)$ as $p\to0$ — depth as a continuous knob interpolating the relaxation from nuclear norm down toward rank. That story is false, and seeing why is what reveals the real mechanism. In the solvable case — commuting matrix sensing, balanced near-zero init, gradient flow on the product — the depth-$N$ limit provably *agrees* with nuclear-norm minimization (the KKT conditions for $\min\langle I,W\rangle$ s.t. feasibility, $W\succeq0$, which for PSD $W$ is exactly nuclear-norm minimization), the same conclusion as depth-2. Worse, one can exhibit a min-nuclear-norm point $\bar W = \text{diag}(1,1,0,\dots)$ that is not even a local Schatten-$p$ minimizer for any $0<p<1$: perturbing $\varepsilon$ into the off-diagonal keeps it feasible and PSD with eigenvalues $1\pm\varepsilon$, and $(1+\varepsilon)^p + (1-\varepsilon)^p < 2$ by strict concavity. The factorization goes to a point those quasi-norms want to move away from. No single matrix norm or quasi-norm captures the bias, because the commuting-sensing world is precisely where min-nuclear-norm and min-rank coincide. So I stop hunting for a norm and analyze the *dynamics*.

What makes the method work lives in the singular-value dynamics of gradient flow. Because the over-parameterization is analytic, $W(t)$ admits an analytic SVD $W = USV^\top$ I can differentiate, and projecting the flow onto the $r$-th singular direction gives $\dot\sigma_r = u_r^\top \dot W v_r$. Substituting the end-to-end dynamics — $WW^\top = US^2U^\top$, $W^\top W = VS^2V^\top$, the powers $[WW^\top]^{(j-1)/N}$ and $[W^\top W]^{(N-j)/N}$ picking out the $r$-th diagonals, exponents adding to $(N-1)/N$ independent of $j$ across $N$ identical terms — yields

$$\dot\sigma_r(t) = -N\,(\sigma_r^2(t))^{(N-1)/N}\,\big\langle\nabla\ell(W(t)),\,u_r(t)v_r(t)^\top\big\rangle.$$

There it is. Given the current $W$, depth $N$ enters the value dynamics *only* through the factor $N(\sigma_r^2)^{(N-1)/N}$, exponent $2-2/N$. For $N=1$ the factor is $1$ — flat, Frobenius-like, every mode treated alike. For $N\ge2$ it is a *power of the singular value's own magnitude* multiplying its velocity: large modes are amplified, small modes attenuated, and the gap sharpens as $N$ grows (the exponent climbs from $1$ at $N=2$ toward $2$). This is rich-get-richer on the spectrum, and crucially it is *not* a fixed functional of $W$ being minimized — the same $W$ reached along different trajectories gives different dynamics, which is exactly why no norm captures it. The vectors must settle for the value dynamics to decouple, and deriving $\dot U,\dot V$ shows that stationary singular vectors require $U^\top\nabla\ell(W)V$ diagonal — the singular vectors of $W$ align with those of $\nabla\ell(W)$ — so gradient flow rotates $W$ into alignment and the values then evolve by the decoupled scalar ODE.

Made quantitative on a single-measurement toy with aligned vectors, eliminating the shared time-factor between two modes and integrating gives the punchline. At $N=1$, linear coupling $\sigma_{r_1} = \alpha\sigma_{r_2} + c$: no bias, the weak mode grows in lockstep. At $N=2$, a power law $\sigma_{r_1} = c\,\sigma_{r_2}^\alpha$: the weak mode grows polynomially slower — *some* bias, the nuclear-norm-strength regime I already saw twice. And for $N\ge3$, $\sigma_{r_1} = (\alpha\,\sigma_{r_2}^{-(N-2)/N} + c)^{-N/(N-2)}$, so as $\sigma_{r_2}\to\infty$ the weak mode *saturates* at a finite asymptote, lower the larger $N$ is. That is the hard low-rank bias: depth $\ge3$ does not merely slow the small modes, it *caps* them — a few large singular values, a sharp shoulder, the rest frozen near zero. This is the structure nuclear norm cannot produce, and it is exactly what is needed to push past the $0.25$ ceiling SVT left on rank3-50.

Every design choice follows. **Depth $N\ge3$:** the bias exponent $2-2/N$ is monotone in $N$; $N=2$ gives only the power-law gap (the nuclear-norm regime the first two rungs lived in), $N\ge3$ gives the saturating cap that beats nuclear norm in the data-poor regime, and empirically $N=4$ is indistinguishable from $N=3$ — so depth $3$ is the cheapest depth buying the strong bias. **Near-zero init:** every $\sigma_r(0)$ must be tiny so every mode starts on the throttled plateau and the dynamics get to *select* which modes switch on (the data-aligned ones); initialize large and modes start off the throttle and cannot be separated. **Balanced init:** the clean end-to-end ODE requires $W_{j+1}^\top W_{j+1} = W_j W_j^\top$ at init, which near-zero Gaussian factors satisfy approximately. **Full hidden dimension $=n$:** I deliberately do *not* cap the inner dimension — capping it would be *explicit* low-rank factorization, and the whole point is that low rank emerges *implicitly* from the depth dynamics. **Small step, many iters:** Adam with a small step approximates the gradient flow and drives training error to interpolation; the bias supplies the rest.

On the literal scaffold the diff from the depth-2 floor is small and exact, which is the point. The fill is the same `nn.Sequential` of `nn.Linear(n,n,bias=False)` layers, the same `_e2e` left-to-right fold, the same Adam masked-MSE loop with the same `train_thres = 1e-7` early stop. Two things change. `depth=3` instead of `2` (guarded by `if depth < 2: raise`), so the per-layer initialization std is $\texttt{init\_scale}^{1/3}\cdot n^{-1/2}$ — the $1/N$ power spreading the small overall scale across *three* multiplied factors. And `lr=1e-3` instead of the depth-2 rung's `5e-3` — a smaller step, right both because three layers compose into a more sensitive product and because the smaller step tracks the gradient flow more faithfully, and the bias *is* the flow. There is no nuclear-norm penalty, no rank cap, no extra regularizer; the entire difference from the floor is one more layer and a smaller learning rate.

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
