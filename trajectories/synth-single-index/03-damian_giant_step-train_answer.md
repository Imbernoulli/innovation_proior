Freezing the biases did exactly what the asymmetric prediction said. On `relu-d100` recovery stayed pinned at $0.998$; on `sign-d100` the mean rose from vanilla's $0.210$ to $0.467$, with seed 456 jumping to $0.848$ where the clean landscape pulled it to a pole — though the spread did *not* tighten ($\{0.439, 0.113, 0.848\}$, with seed 123 even lower than its vanilla value), because the wide net's many independent rows, still driven by noisy $256$-sample gradients, do not reliably converge to the same place. The decisive reading is `hermite-d100`: it went from $0.656$ to $0.614$, essentially flat. Freezing the biases collapses the landscape, but on the $k=3$ link the per-mini-batch direction signal is $\sim d^{-1}$, far below the $256$-sample noise $\sim \sqrt{d/256}$, and a cleaner landscape does nothing about a signal buried below noise. The missing ingredient was never landscape shape — it is *signal aggregation*.

I propose **one giant full-batch step plus a closed-form ridge readout**: stop crawling, and extract the weak signal by summing it over the entire training set at once. Return to the first-layer gradient at a symmetric/zero-output init, where the self-interaction term drops and the gradient on a row is the clean correlation $\nabla_{w_j} L = -2 a_j\,\mathbb{E}[\,f^*(x)\,x\,\sigma'(\langle w_j, x\rangle + b_j)\,]$. Expanding $\mathbb{E}[f^*(x)\,x\,\sigma'(\langle w, x\rangle)]$ against the Gaussian via Stein's lemma and Hermite orthogonality gives an asymptotic series in $d^{-1/2}$ whose terms shrink by $\sqrt{d}$ each; the first informative term points the row into the relevant subspace — along $\theta^*$ — and which term that is is set by the information exponent $k$. Two facts then decide the algorithm. First, the empirical gradient fluctuates around the population one with noise $\sim \sqrt{d/n}$, so to see the leading signal above its own noise the third-order term needs $n \sim d^2 \approx 10^4$ samples; a single batch of $256 \ll 10^4$ is essentially pure noise — exactly the $0.614$ hermite stall — whereas the full train set $n_{\text{train}} = 32768 > d^2$ is comfortably enough. The update should therefore be one *full-batch* step, not a long crawl. Second, the rows are unit vectors and I want an $O(1)$ rotation toward $\theta^*$, but the gradient I step along has norm only $\sim d^{-(k-1)/2}$; to get an $O(1)$ rotation the step has to grow with dimension — the *giant* step, $\eta_1 \sim d^{(k-1)/2}$, which is $O(1)$ for $k=1$ (relu, sign) and grows like $d$ for $k=3$ (hermite).

This inverts the whole loop: `make_optimizer` returns SGD with `lr=0.0` so the mini-batch loop changes nothing (a noisy batch-$256$ first step is the very failure mode), `training_step` is a no-op that only logs the batch target energy, and the real work happens once in `finalize`, where the harness hands over the full `x_train, y_train`. The direction estimator is read off the first non-vanishing term of the gradient series, per link. For $k=1$ (relu, sign) the first moment is already informative — $\mathbb{E}[y\,x] = \mu_1\,\theta^*$ by Stein on the first Hermite — so the estimator is simply $\mathrm{normalize}\big(\tfrac{1}{n}\sum_i y_i x_i\big)$, one matrix-free full-batch average; crucially this touches $g$ only through its low-order Hermite *moments*, never through derivatives, so it is robust to sign's non-smoothness, which is why it should rescue the sign link. For $k=3$ (hermite) the first moment estimates $\mu_1\theta^* = 0$ and is useless, so I build the third-order term: the multivariate third Hermite tensor contracted twice against a probe $v$ gives the vector $x\langle x,v\rangle^2 - x\|v\|^2 - 2v\langle x,v\rangle$, and $\mathbb{E}[\,y\cdot(\text{that})\,] \propto C_3(v,v) = \mu_3\langle\theta^*,v\rangle^2\theta^*$ — a vector along $\theta^*$ whose strength grows with the overlap $\langle\theta^*, v\rangle$. Because that strength is $\sim\langle\theta^*,v\rangle^2$, a fresh random probe ($\langle\theta^*,v\rangle\sim d^{-1/2}$) gives a weak contraction, so I sharpen it with tensor power iteration: the map $v \mapsto C_3(v,v)/\|\cdot\|$ has $\theta^*$ as its attracting fixed point, so I form a coarse direction by averaging the contraction over the current random rows as probes, then refine with two power-iteration passes.

What I do to the network with that direction is overwrite each row toward it, by an amount set by the giant learning rate — large $\eta_1$ replaces the random probe almost entirely with the signal. I express this as a convex mix with weight $\mathrm{mix} = \eta_1/(\eta_1 + 1)$: small $\eta_1$ leaves the row mostly itself, the giant regime pushes it almost entirely onto the direction. Here I am candid about the grounding. The cleanest analysis pairs the giant step with a matching weight decay $\lambda_1 = \eta_1^{-1}$ so the $-W^{(0)}$ term *exactly cancels* the random init and the first layer becomes pure gradient features; the harness instead realises the same "overwrite the probe with the signal" idea through the convex mix, which for the giant $\eta_1$ is numerically almost identical (with $\eta_1 = \sqrt{W}\,d^{(k-1)/2}$ and $d=100$, the hermite mix weight is essentially $1$). Two refinements keep the resulting features usable. If every row became exactly the same direction the basis would degenerate, so (a) I add a small *orthogonal* jitter — a random component projected off the direction, renormalised, scaled by $0.05$ — keeping the rows a tight, well-conditioned spread around $\theta^*$ rather than identical, without re-adding off-subspace noise; and (b) I reset the biases to a deterministic spread $b_j = \mathrm{linspace}(-2.5, 2.5, W)$ so that $x \mapsto \mathrm{ReLU}(\langle\theta^*,x\rangle + b_j)$ forms a one-dimensional random-feature basis covering the range of $u \sim N(0,1)$; rows are then renormalised to the sphere, since ReLU is positively homogeneous and only directions and biases matter.

The giant rate carries one harness-specific factor. The first-layer gradient has a factor $a_j$ out front, and the scaffold's readout normalisation puts $a_j = 1/\sqrt{W}$, shrinking the effective step by $1/\sqrt{W}$; to keep the theory's effective step (which uses $a_j \sim \pm 1$) I multiply by $\sqrt{W}$. So the full rate is $\eta_1 = \sqrt{W}\,d^{(k-1)/2}$ — $\sqrt{W}$ undoing the readout normalisation, $d^{(k-1)/2}$ the information-exponent scaling — which is exactly why `init_two_layer` sets `fc2.weight` to the constant $1/\sqrt{W}$ with zero biases: the rate is calibrated to that readout scale. Finally the readout itself. After the giant step the first layer is frozen and fitting the head on the fixed features $\phi(x) = \mathrm{ReLU}(W^{(1)}x + b)$ is a convex least-squares problem, so I solve it in closed form rather than running GD: $\beta = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y$ where $\Phi$ is $[n, W+1]$ (the ReLU features plus a constant column for the head bias) and $\lambda = \max(\text{weight\_decay}, 10^{-4})$ keeps the Gram matrix invertible; then `fc2.weight` $= \beta[:-1]$, `fc2.bias` $= \beta[-1]$. This is the `finalize` hook the previous rung deliberately left empty, now earning its keep. The whole method lives in `init_two_layer` (random probes, constant readout) and `finalize` (estimate the direction, giant-overwrite the rows, spread the biases, ridge-fit the head), with the mini-batch loop neutralised.

The expectations against frozen-bias are sharp. On `relu-d100` I expect to *match* the $0.998$ ceiling — the first-moment estimator is just the clean route there. On `sign-d100` I expect the largest gain over the erratic $0.467$: the full-batch first moment $\mathrm{normalize}(\mathrm{mean}(y\,x))$ over all $32768$ samples is a deterministic, low-variance estimate of $\theta^*$, and since it never differentiates $g$, the non-smoothness is a non-issue — recovery near $0.99$ on every seed with the spread *collapsing*, which is the cleanest test that the failure was aggregation rather than landscape. On `hermite-d100` this is make-or-break: the giant step aggregates the third-order signal over $n_{\text{train}} > d^2$ and refines it with power iteration, so recovery should leap from $0.614$ to near-perfect — the single largest improvement on the ladder. With direction recovery near $1$ across all three links, this rung saturates the benchmark and is the endpoint.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — step 3: giant step + ridge readout
class Strategy:
    """One giant gradient step on the first layer, then fit the readout."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        # Information exponent k of the link. The scaffold scales readout weights
        # by 1/sqrt(width), so multiply by sqrt(width) to keep the theory's
        # effective first-layer step size.
        info_exponent = 3 if config.link == "hermite" else 1
        self._giant_lr = math.sqrt(max(config.width, 1)) * (
            float(config.dim) ** max(0.0, 0.5 * (info_exponent - 1))
        )

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Random probes on the sphere; constant 1/sqrt(W) readout; zero biases.
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            nn.init.zeros_(net.fc1.bias)
            net.fc2.weight.fill_(1.0 / math.sqrt(config.width))
            nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        # No mini-batch updates; the giant step is full-batch, done in finalize().
        return torch.optim.SGD(net.parameters(), lr=0.0)

    def training_step(
        self,
        net: TwoLayerMLP,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        config: TaskConfig,
    ) -> StepMetrics:
        return StepMetrics(loss=float(torch.mean(y * y).item()), extra={})

    @staticmethod
    def _normalize(v: torch.Tensor) -> torch.Tensor:
        return v / v.norm().clamp_min(1e-12)

    def _first_order_direction(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # k=1 links (ReLU/sign): the first Hermite moment E[y x] = mu_1 theta* is nonzero.
        return self._normalize((y[:, None] * x).mean(dim=0))

    def _hermite_contract(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # Third-Hermite contraction E[y (x<x,v>^2 - x||v||^2 - 2 v<x,v>)].
        z = x @ v
        contracted = (
            y[:, None]
            * (x * z.square()[:, None] - x - 2.0 * v[None, :] * z[:, None])
        ).mean(dim=0)
        return self._normalize(contracted)

    def _hermite_direction(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        probes: torch.Tensor,
    ) -> torch.Tensor:
        # Average probe contractions, then refine with tensor power iteration.
        probes = probes / probes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z = x @ probes.t()
        y_col = y[:, None]
        first = (y_col * x).mean(dim=0)
        yz = (y_col * z).mean(dim=0)
        term = (y_col * z.square()).t() @ x / float(x.shape[0])
        term = term - first[None, :] - 2.0 * probes * yz[:, None]
        direction = self._normalize(term.mean(dim=0))
        for _ in range(2):
            direction = self._hermite_contract(x, y, direction)
        return direction

    def _estimate_direction(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.link == "hermite":
            return self._hermite_direction(x_train, y_train, net.fc1.weight.detach())
        return self._first_order_direction(x_train, y_train)

    def _apply_giant_step(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        config: TaskConfig,
    ) -> None:
        direction = self._estimate_direction(net, x_train, y_train)
        probes = net.fc1.weight.detach()
        jitter = torch.randn_like(probes)
        jitter = jitter - (jitter @ direction)[:, None] * direction[None, :]
        jitter = jitter / jitter.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Dimension-scaled giant step: larger eta -> probe rows overwritten by signal.
        mix = self._giant_lr / (self._giant_lr + 1.0)
        rows = (1.0 - mix) * probes + mix * direction[None, :] + 0.05 * jitter
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            net.fc1.weight.copy_(rows)
            # Re-sample biases to a spread of thresholds: a usable 1-D link basis.
            net.fc1.bias.copy_(
                torch.linspace(-2.5, 2.5, config.width, device=rows.device)
            )

    def finalize(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        config: TaskConfig,
    ) -> None:
        # Closed-form ridge regression on the (now frozen) features.
        device = next(net.parameters()).device
        with torch.no_grad():
            x_device = x_train.to(device)
            y_device = y_train.to(device)
            self._apply_giant_step(net, x_device, y_device, config)
            feats = torch.relu(
                x_device @ net.fc1.weight.t() + net.fc1.bias
            )  # [N, W]
            feats = torch.cat(
                [feats, torch.ones(feats.shape[0], 1, device=device)], dim=1
            )
            target = y_device.view(-1, 1)
            lam = max(config.weight_decay, 1e-4)
            gram = feats.t() @ feats + lam * torch.eye(
                feats.shape[1], device=device
            )
            rhs = feats.t() @ target
            beta = torch.linalg.solve(gram, rhs).view(-1)
            net.fc2.weight.copy_(beta[:-1].view(1, -1))
            net.fc2.bias.fill_(float(beta[-1]))


def build_strategy(config: TaskConfig) -> Strategy:
    return Strategy(config)
```
