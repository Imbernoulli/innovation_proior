**Problem.** The multi-epoch rung confirmed that a small fixed dataset is memorized: at `N = 32` it sat
at `0.510` with `mean_steps ~ 18_600` (early-stop on *training* accuracy fired on the interpolated
table), and at `N = 50, 64` it was dead chance with even shorter runs. The finite table gave the network
a shortcut — fit the sample, trip the stop — that beat the faint sparse signal every time.

**Key idea (single-pass online).** Remove the table. Return a *maximal* random dataset (up to
`max_train_examples`) so that within the `30_000`-step budget the loop never re-passes: every step sees
fresh i.i.d. data and the minibatch gradient is an unbiased estimate of the *population* gradient. With
no idiosyncratic labels to fit, the only persistent signal in the gradient stream is the Fourier gap of
majority's spectrum — degree-`(K-1)` mass at the relevant coordinates dominates degree-`(K+1)` mass at
the rest by `gamma ~ N^{-(K-1)/2}` — so the relevant first-layer weights drift up and amplify, and the
classifier snaps once they overtake the irrelevant ones. Init and AdamW stay at defaults, so the only
changed variable from the previous rung is dataset size (small -> maximal).

**Why / limits.** Online makes training accuracy track population accuracy, so the early-stop cannot
fire on memorization; runs use the full budget unless they truly solve the rule. But online ties steps
to fresh samples, and resolving the gap needs `~N^{K-1}` steps. The budget covers this only at `N = 32`
(helped by 512 parallel neurons); `N = 50, 64` need orders of magnitude more steps and stay at chance.

**Hyperparameters.** Dataset size `num_examples = max_train_examples` (one-pass). Init: Xavier-uniform
(ReLU gain on layer 0), zero biases. AdamW: `lr = 1e-3`, `wd = 1e-2`, `(beta1, beta2) = (0.9, 0.999)`.

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
    """Return AdamW hyperparameters for the fixed training loop."""
    return {
        "lr": 1e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
    }
```
