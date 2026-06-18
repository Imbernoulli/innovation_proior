**Problem.** Frozen biases collapse the landscape but cannot rescue the hard `k=3` link
(`hermite-d100` flat at `0.614`), because the third-order direction signal is `~ d^{-1}`, buried below
the per-mini-batch noise `~ sqrt(d/256)`; and `sign-d100` stays erratic (`0.467`, seeds {0.439, 0.113,
0.848}). The missing ingredient is signal aggregation, not landscape shape.

**Key idea (one giant full-batch step + ridge readout).** Neutralise the mini-batch loop and do the
work once, full-batch, in `finalize`. The first-layer population gradient points each row toward
`theta*`; aggregating it over all `n_train > d^2` samples surfaces a signal that 256-sample steps
cannot. Estimate the direction from the link's first non-vanishing Hermite moment — first moment
`E[y x]` for `k=1` (relu, sign), the third-Hermite contraction with tensor power-iteration for `k=3`
(hermite) — then take one *giant*, dimension-scaled step (`eta_1 = sqrt(W) d^{(k-1)/2}`) that overwrites
the random rows with that direction, re-spread the biases into a 1-D link basis, and close-form
ridge-fit the head on the now-frozen features.

**Why / harness grounding.** `make_optimizer` returns `lr=0.0` (the loop is a no-op — a noisy batch-256
first step is the failure mode); the giant step lives in `finalize`, which sees the full train set.
`sqrt(W)` undoes the scaffold's `1/sqrt(W)` readout normalisation (the gradient carries a factor `a_j`);
`d^{(k-1)/2}` is the information-exponent scaling. The harness realises the analysis's
weight-decay-cancelled init-erase through a convex mix `mix = eta_1/(eta_1+1)` (≈1 in the giant regime),
plus small orthogonal jitter so the aligned rows stay a well-conditioned spread; biases reset to
`linspace(-2.5, 2.5, W)`; ridge `lambda = max(weight_decay, 1e-4)`.

**What to watch.** `relu` matches the `0.998` ceiling; `sign` jumps near `0.99` with the seed spread
collapsing (proof the failure was aggregation, not landscape); `hermite` is the make-or-break, expected
to leap from `0.614` to near-perfect. This rung saturates every link, so it is the endpoint.

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
