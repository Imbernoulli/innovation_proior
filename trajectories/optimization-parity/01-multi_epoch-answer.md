**Problem.** A fixed two-layer MLP must learn a hidden `8`-sparse parity under a frozen loop that
trains for `30_000` steps at batch 128, reshuffling each epoch. Sparse parity is statistically trivial
(`Theta(K log N)` labels suffice) but SQ-hard (`N^{Omega(K)}` correlational work). The frozen loop ties
the number of gradient steps to the number of fresh examples whenever the dataset is consumed once, so
single-pass online training spends the entire budget without ever paying the compute the floor demands.

**Key idea (multi-epoch reuse).** Separate compute from evidence. Return a *small* fixed dataset
(`m = 10_000`) instead of a maximal random one. Because the loop re-passes and reshuffles whenever the
data is exhausted, this converts the fixed step budget into many epochs over the same examples:
`epochs = max_steps * batch_size / m`. Each step is now an estimate of the *empirical* gradient of one
informative sample, optimized repeatedly, rather than one pass over a fresh stream. Initialization and
AdamW settings are left at the scaffold defaults so the only changed variable is dataset size.

**Why it could work / why it is fragile.** Reuse manufactures no new evidence; it only spends compute on
a fixed empirical distribution. It helps only if the sample already carries the sparse Fourier trace
(`gap ~ N^{-(K-1)/2}`, faintest at large `N`) and the default weight decay (`wd = 1e-2`) tilts the
memorization-vs-generalization competition toward the low-norm sparse rule. The risk is that a width-512
network memorizes `10_000` points fast, trips the loop's early-stop on *training* accuracy, and halts in
the memorization phase before generalization arrives.

**Hyperparameters.** Dataset size `m = min(10_000, max_train_examples)`; fresh i.i.d. uniform binary
`x`, parity labels from the provided secret. Init: Xavier-uniform (ReLU gain on layer 0), zero biases.
AdamW: `lr = 1e-3`, `wd = 1e-2`, `(beta1, beta2) = (0.9, 0.999)` — all defaults.

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
    """Use a smaller, configurable random dataset to allow multi-epoch reuse."""
    generator = torch.Generator().manual_seed(seed)
    train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
    num_examples = min(train_examples, config.max_train_examples)

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
    """Return AdamW hyperparameters for the fixed training loop."""
    return {
        "lr": 1e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
    }
```
