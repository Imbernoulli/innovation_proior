**Problem.** Vanilla SGD recovers the direction on the easy `k=1` ReLU link but stalls on the hard
links: `hermite-d100` (`k=3`) at `0.656` recovery and `sign-d100` (`k=1`, non-smooth) at `0.210` with
huge seed spread. The first layer and the readout are trained jointly, so the high-dimensional
direction search and the one-dimensional link fit interfere, and the landscape has no benign structure
pulling every run to `theta*`.

**Key idea (frozen biases).** Split the two jobs by *what is trainable*. The hidden-neuron biases are
the random-feature sampling of the one-dimensional link kernel (a bank of thresholds `{ReLU(u - b_j)}`);
the directions are the high-dimensional search. Freeze the biases at a wide random init
(`Uniform(-1,1)`) and train only the directions and the head. Frozen biases do not depend on the
directions, so each row enters the loss only through the scalar overlap `m_j = <w_j, theta*>` — the
gradient on each row becomes colinear with `theta*` and the `d`-dimensional landscape collapses to a
benign scalar flow whose only critical points are the equator and the poles.

**Why / what the harness keeps.** The fixed model is a *wide* MLP, so this rung keeps only the one
essential move — the bias freeze — applied to the standard net; it does not tie the rows to a single
shared direction, schedule a two-phase search-then-fit, or do a closed-form ridge refit (those are the
next rung). First-layer rows are put on the unit sphere so the optimiser controls direction only; the
optimiser is built over `requires_grad` params, excluding the frozen biases.

**Hyperparameters.** Unit-sphere `fc1.weight`; `fc1.bias ~ Uniform(-1,1)`, frozen; uniform `fc2` in
`[-1/sqrt(W),1/sqrt(W)]`, `W=256`; SGD `base_lr=1e-2`, `momentum=0.9`, `weight_decay=0.0`, batch 256,
`max_steps=8000`; no finalize.

**What to watch.** Asymmetric prediction: `relu` unchanged at the ceiling; `sign` improved and
*tightened* across seeds (the `k=1` direction signal just needed a clean landscape); `hermite` roughly
flat (the freeze collapses the landscape but does not aggregate the weak `k=3` signal). A stuck hermite
forces full-batch signal aggregation at step 3.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — step 2: frozen-bias shallow net
class Strategy:
    """Frozen-bias shallow network: biases sampled once and never trained."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Random first-layer rows on the unit sphere; biases sampled uniformly
        # in [-1, 1] and FROZEN -- the key move.
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            net.fc1.bias.uniform_(-1.0, 1.0)
        net.fc1.bias.requires_grad_(False)

        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        params = [p for p in net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def training_step(
        self,
        net: TwoLayerMLP,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        config: TaskConfig,
    ) -> StepMetrics:
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return StepMetrics(loss=float(loss.item()), extra={})

    def finalize(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        config: TaskConfig,
    ) -> None:
        return


def build_strategy(config: TaskConfig) -> Strategy:
    return Strategy(config)
```
