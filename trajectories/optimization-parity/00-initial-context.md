## Research question

A fixed two-layer MLP must learn a hidden `k`-sparse parity: draw a uniform binary vector
`x in {0,1}^N`, label it by `y = (sum_{i in S} x_i) mod 2` for an unknown subset `S` of size `K = 8`.
The architecture (`Linear(N, 512) -> ReLU -> Linear(512, 1) -> Sigmoid`), the optimizer family
(AdamW), the batch size (128), the binary-cross-entropy loss, the training loop, and the evaluation
protocol are all frozen. The only design freedom is in three knobs: **model initialization**, **the
training dataset that gets constructed**, and the **AdamW hyperparameters** (`lr`, `wd`, `beta1`,
`beta2`). The question is whether those three knobs alone can move a phase transition that, for the
default settings, sits at or beyond the step budget — i.e. whether one can raise the mean held-out
test accuracy above chance on `N = 32, 50, 64`.

Sparse parity is the canonical "feature-learning" problem precisely because it is statistically trivial
and computationally hard. Distinguishing one subset among `C(N, K)` candidates needs only
`Theta(K log N)` labels, but every correlation-based learner faces a wall: the parities are orthonormal
under the correlation inner product, so a wrong subset that overlaps `K-1` of the right indices has the
same (zero) correlation with `y` as one that overlaps none. There is no "getting warmer," and the
statistical-query lower bound formalizes this into an `N^{Omega(K)}` work floor for any noise-tolerant,
aggregate-statistic learner — gradient descent included, because a stochastic gradient is an
expectation estimated to a tolerance set by its noise.

## Prior art before the first rung (the feature-learning lineage)

The default fill the first rung reacts to is the setup that the following line of work converged to.

- **Statistical-query hardness (Kearns 1998; Blum, Furst, Jackson, Kearns, Mansour, Rudich 1994).**
  Casts learning as bounded correlational queries answered to tolerance `tau`; a Fourier/Parseval
  argument shows `T` queries can each single out the target for only a `1/tau^2` fraction of possible
  parities, so `T / tau^2 >= Omega(N^K)`. Gap: it is a *floor*, telling you the best achievable cost
  but not whether ordinary gradient descent reaches it or merely stalls.
- **Gradient-variance pessimism over all parities (Shalev-Shwartz, Shamir, Shammah 2017; Abbe, Sandon
  2020).** Over the dense family of all `2^N` parities, the variance of the per-target gradient is
  bounded by `G^2 / 2^N`, exponentially small, so the gradient is target-blind and "no gradient method
  should work." Gap: the bound is about the *dense* family; it answers the wrong question for a problem
  that fixes the sparsity to a small constant `K`.
- **Adaptive optimization with weight decay (Kingma & Ba 2015; Loshchilov & Hutter 2019).** AdamW is the
  frozen optimizer family: a per-coordinate adaptive step with a decoupled weight-decay shrink applied
  outside the `sqrt(v)` normalization. Gap: the defaults (`lr = 1e-3`, `wd = 1e-2`) were never tuned
  for the slow-amplification regime parity lives in — the decay can erode a feature signal that starts
  out polynomially faint.

## The fixed substrate

A single training-and-evaluation driver is frozen and must not be touched. It builds the fixed MLP via
`build_model`, calls the editable `init_model`, reads the editable `get_optimizer_config` to construct
`torch.optim.AdamW(..., betas=(beta1, beta2), weight_decay=wd)`, and trains with `nn.BCELoss` at batch
size 128 for up to `max_steps = 30_000` per run. Each epoch reshuffles the returned training set with a
per-order generator (`torch.randperm`) and steps through it in minibatches; when the dataset is
exhausted it starts a fresh shuffled pass, so **the number of epochs is not an independent knob** — up
to last-batch rounding, `epochs = max_steps * batch_size / num_examples`. The loop logs running-window
loss/accuracy and early-stops a run once a windowed train accuracy of `0.999` holds for four logging
windows past `min_steps_before_stop = 1_000`. The driver also provides the helpers a fill may use:
`parity_labels(x, secret)` (the sum-mod-2 labeler), `make_test_set` (a held-out `16_384`-example test
set per secret), and the `TaskConfig` fields (`n_features`, `secret_size`, `hidden_width`,
`max_train_examples = 12_800_000`, etc.). Held-out accuracy is computed with `evaluate_accuracy`.

## The editable interface

Exactly one region of `pytorch-examples/optimization_parity/custom_strategy.py` is editable: the three
functions `init_model`, `make_dataset`, and `get_optimizer_config`. Every method on the ladder is a fill
of this same contract. The contract:

- `init_model(model, config)` mutates the MLP's parameters in place and **must not depend on the hidden
  secret** (it only sees `model` and `TaskConfig`).
- `make_dataset(secret, config, seed)` may use the hidden `secret` and must return `(x, y)` or
  `{"x": x, "y": y}`, with `x` of shape `[num_examples, N]` binary, `y` of shape `[num_examples]` (or
  `[num_examples, 1]`) binary, and `num_examples <= config.max_train_examples = 12_800_000`. The size it
  returns is what sets the epochs-per-budget ratio above.
- `get_optimizer_config(config)` must return a dict with `lr`, `wd`, `beta1`, `beta2`.

The starting point is the scaffold default fill below. Each rung replaces exactly these three definitions
and nothing else.

```python
# EDITABLE region of custom_strategy.py — default scaffold fill
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
    """Return a reproducible training dataset for one hidden secret."""
    generator = torch.Generator().manual_seed(seed)
    num_examples = 4_096
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

## Evaluation settings

The benchmark is evaluated on three configurations — `(N=32, K=8)`, `(N=50, K=8)`, `(N=64, K=8)`, all
with width `W = 512` — across seeds `{42, 123, 456}`. For each configuration the driver samples hidden
secrets and random epoch-orderings and reports the **mean held-out test accuracy** across all runs as
the metric `test_accuracy` (also emitted as `score`); higher is better. The `mean_steps` field records
how many steps each run took before early-stopping or hitting the budget.
