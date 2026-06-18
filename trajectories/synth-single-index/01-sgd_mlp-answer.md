**Problem.** Recover the hidden direction `theta*` (and fit the link `g`) of a single-index model
`y = g(<theta*, x>) + noise` with a fixed two-layer ReLU MLP, across links of information exponent
`k = 1` (ReLU, sign) and `k = 3` (`He_3`). The recipe controls only init, optimiser, the per-step
update, and an optional refit.

**Key idea (the control rung).** Do nothing single-index-aware: Kaiming-normal first layer, small
uniform readout, plain SGD with momentum on the *whole* network, mean-squared-error mini-batch
updates, no post-hoc refit. The direction is left entirely to whatever the joint SGD dynamics extract
from 256-sample gradients.

**Why this is the floor.** The first-layer gradient carries the direction only through the link's first
non-vanishing Hermite moment. For `k=1` that moment is `O(1)` and present every step, so SGD recovers
the direction well. For `k=3` the signal is `~ d^{-1}`, far below the per-mini-batch noise `~ sqrt(d/256)`,
so the rows barely move toward `theta*`; and because every row is trained jointly with the head, the
direction search and the link fit interfere. Nothing decouples them and nothing aggregates the weak
signal across the dataset — exactly the two levers the later rungs pull.

**Hyperparameters.** `base_lr = 1e-2`, `momentum = 0.9`, `weight_decay = 0.0`, batch 256,
`max_steps = 8000`; Kaiming-normal `fc1`, uniform `fc2` in `[-1/sqrt(W), 1/sqrt(W)]`, `W = 256`.

**What to watch.** A clean `k`-split: `relu-d100` near-perfect, `sign-d100` middling and high-variance,
`hermite-d100` the weakest. That failure on the hard link forces a landscape fix at step 2.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — step 1: vanilla SGD MLP
class Strategy:
    """Vanilla SGD on a two-layer ReLU MLP (reference baseline)."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Kaiming-normal first layer; small uniform second layer.
        nn.init.kaiming_normal_(net.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(net.fc1.bias)
        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            net.parameters(),
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
