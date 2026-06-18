## Research question

A single-index model says the target depends on a `d`-dimensional input only through one hidden
projection: `y = g(<theta*, x>) + noise`, with `x ~ N(0, I_d)`, `theta*` an unknown unit direction,
and `g` an unknown univariate link. The thing being designed is the **training recipe for a fixed
two-layer ReLU MLP** — how to initialise its two layers, what per-step update to run, and whether to
do a post-hoc closed-form refit — so that the network *recovers the direction* `theta*` (and, through
it, fits the link `g`) across link families of increasing difficulty. The architecture, the data
generator, the link functions, the evaluator, and the training driver are all fixed; only the recipe
is free.

The difficulty is set by one quantity per link: the **information exponent** `k`, the index of the
first non-vanishing Hermite coefficient of `g`. The benchmark spans three links — `ReLU` (`k=1`,
easy), `He_3/sqrt(6)` (`k=3`, beyond the kernel regime), and `sign` (`k=1` but non-smooth, a
robustness probe) — all at `d=100` with Gaussian inputs.

## Prior art before the first rung

The first rung is the textbook recipe — Kaiming init, vanilla SGD with momentum on the whole network,
mini-batches drawn fresh each step. It is the control every later method reacts to. The lineage that
sets the stage for *why* that control is the natural starting point and *where* it is expected to
break:

- **Random-feature / kernel models (Rahimi & Recht 2007; the NTK regime, Jacot et al. 2018).** Freeze
  the first layer at its random init and fit only the head: the features point in random directions
  that do not adapt to `theta*`, so a kernel needs `n = Theta(d^p)` samples to fit a degree-`p`
  polynomial even one that secretly lives on a single direction. Gap: cannot move its features, so it
  cannot learn the direction — the curse of dimensionality is structural, not an accident.
- **Teacher-student single-neuron analyses (Saad & Solla 1995; Goldt et al. 2019).** Study hidden-
  direction recovery under gradient descent, but with the student activation *set equal to the link*
  `g` — if you already know `g`, the non-parametric half evaporates and only finding `theta*` remains.
  Gap: assumes the link is known; not our setting.
- **Information exponent and online-SGD thresholds (Ben Arous, Gheissari, Jagannath 2021).** For online
  SGD on a single neuron, weak recovery of `theta*` costs `n = Theta(d^{k-1})` samples, set by the
  information exponent `k` alone — `O(d)` for `k=1`, `Theta(d^{k-1})` for `k>=3`. The difficulty is
  *escaping the flat equator*: a random init starts at correlation `m = <theta,theta*> ~ 1/sqrt(d)`,
  where the gradient signal scales like `m^{k-1}` and competes with empirical noise `~ sqrt(d/n)`. Gap:
  the analysis is for a fitted single neuron with a known link, and predicts that for hard links the
  per-mini-batch signal is essentially noise.
- **Classical semi-parametrics (projection pursuit, sliced inverse regression, the Hermite estimators
  of Dudeja & Hsu 2018).** Recover `theta*` (and `g`) at `n = O(d^s)` with bespoke estimators built
  around individual Hermite moments — beautiful, but not a network trained end to end; direction
  recovery and link fitting are done by separate, hand-engineered machinery. Gap: not a gradient
  method on the fixed MLP.

The control rung is exactly "run plain SGD on the whole net and hope the first layer moves on its
own"; the methods that follow are increasingly deliberate about *making* the first layer move toward
`theta*`.

## The fixed substrate

A single-run driver is frozen and must not be touched. For each replica it samples a fresh
`theta* ~ Uniform(S^{d-1})`, draws `n_train = 32768` Gaussian inputs with noisy targets
(`noise_std = 0.1`) and a noise-free test set, builds the network, calls the recipe's four callbacks,
loops `max_steps = 8000` mini-batches of size 256 (fresh i.i.d. index each step), evaluates every 500
steps, then calls `finalize` once and evaluates again. The fixed pieces:

- **Model.** `TwoLayerMLP = Linear(d, W) -> ReLU -> Linear(W, 1)` with `W = 256`, scalar output.
  `fc1` and `fc2` both carry a bias.
- **Data.** `x ~ N(0, I_d)`, `d = 100`; `y = g(<theta*, x>) + eps`, `eps ~ N(0, 0.1^2)` on train,
  noise-free on test.
- **Direction estimator (fixed, used by the scorer).**
  `theta_hat = normalize(sum_j |a_j| w_j)`, where `w_j` are the rows of `fc1.weight` and `a_j` the
  entries of `fc2.weight` — the readout-weighted first-layer rows.
- **Config handed to the recipe** (`TaskConfig`): `dim`, `width`, `link in {relu, hermite, sign}`,
  `noise_std`, `n_train`, `batch_size`, `max_steps`, `base_lr = 1e-2`, `weight_decay = 0.0`,
  `momentum = 0.9`.

## The editable interface

Exactly one region is editable — the `Strategy` class and `build_strategy(config)` in
`pytorch-examples/synth_single_index/custom_strategy.py` (lines 176–239). Every method on the ladder
is a fill of this same contract:

| Callback | Contract |
|---|---|
| `init_two_layer(net, config)` | Initialise `fc1`, `fc2` in place; may freeze params via `requires_grad_` |
| `make_optimizer(net, config) -> Optimizer` | Return a `torch.optim.Optimizer` over the trainable params |
| `training_step(net, opt, x, y, step, config) -> StepMetrics` | One update on the given mini-batch |
| `finalize(net, x_train, y_train, config)` | Optional global refit (e.g. closed-form ridge) after the step budget |

The starting point is the scaffold default — vanilla SGD on the whole net — shown below. Each later
method replaces exactly this block and nothing else.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — default fill: vanilla SGD MLP
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

    def make_optimizer(self, net: TwoLayerMLP, config: TaskConfig) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            net.parameters(),
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def training_step(self, net, optimizer, x, y, step, config) -> StepMetrics:
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return StepMetrics(loss=float(loss.item()), extra={})

    def finalize(self, net, x_train, y_train, config) -> None:
        return


def build_strategy(config: TaskConfig) -> Strategy:
    return Strategy(config)
```

## Evaluation settings

Three environments, each a link family at `d = 100`: `relu-d100` (`k=1`), `hermite-d100` (`k=3`), and
`sign-d100` (`k=1`, non-smooth). Three top-level seeds {42, 123, 456}, one internal replica per seed
(distinct `theta*` and data draws), trained from scratch; under 30 minutes per (baseline x env) on one
H100. Per environment we report `test_mse_<env>` (noise-free test MSE, lower is better),
`direction_recovery_<env> = |<theta_hat, theta*>|` (higher is better, 1 == perfect), and
`score_<env> = direction_recovery_<env>` (the primary signal).
