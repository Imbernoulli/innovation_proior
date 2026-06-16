**Problem.** The online default cracked `N = 32` (mean `0.771`, high variance from a solved/unsolved
mixture, near-full `mean_steps`) but left slack: many seeds crossed the amplification threshold only
just within the `30_000`-step budget, and `N = 50, 64` stayed at chance against the `N^{K-1}` step wall.
The remaining lever lives at `N = 32`, where faster amplification would tip marginal seeds over.

**Key idea (no weight decay).** Online there is no finite table to memorize, so the decoupled decay
`theta <- (1 - lr*wd) theta` no longer earns its keep eroding a dense memorizer — it can only act on the
feature signal itself. The relevant first-layer weights start near init and climb under a faint drift
`gamma ~ N^{-(K-1)/2}`; a constant per-step shrink is a steady headwind that delays the threshold-crossing,
and it shrinks the signal-carrying weights at the same rate as the dead coordinates. Set `wd = 0`,
keeping the maximal one-pass dataset and standard init unchanged, so the only changed variable from the
default is the decay knob.

**Why / limits.** Irrelevant coordinates have zero population drift (pure `sqrt(t)` diffusion, sub-linear
and already damped by AdamW's per-coordinate normalization), while relevant coordinates drift linearly in
`t` — so the diffusion-capping role of decay is not binding at a `30_000`-step horizon, and removing it
is net positive. The change only helps where amplification was marginal (`N = 32`); `N = 50, 64` remain
chance-bound by the budget, which decay never affected.

**Hyperparameters.** Dataset size `num_examples = max_train_examples` (one-pass). Init: Xavier-uniform
(ReLU gain on layer 0), zero biases. AdamW: `lr = 1e-3`, **`wd = 0.0`**, `(beta1, beta2) = (0.9, 0.999)`.

```python
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP without using the hidden secret."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def make_dataset(
    secret: tuple[int, ...],
    config: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a maximal random dataset to induce one-pass training."""
    generator = torch.Generator().manual_seed(seed)
    num_examples = config.max_train_examples
    x = torch.randint(
        low=0,
        high=2,
        size=(num_examples, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    y = parity_labels(x, secret)
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return AdamW hyperparameters with no weight decay."""
    return {
        "lr": 1e-3,
        "wd": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
    }
```
