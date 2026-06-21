## Research question

A single-index model says the target depends on a `d`-dimensional input only through one hidden projection: `y = g(<theta*, x>) + noise`, with `x ~ N(0, I_d)`, `theta*` an unknown unit direction, and `g` an unknown univariate link. The design task is the **training recipe for a fixed two-layer ReLU MLP** — how to initialise its layers, what per-step update to run, and whether to do a post-hoc closed-form refit — so that the network recovers the direction `theta*` across link families of increasing difficulty. The architecture, data generator, link functions, evaluator, and training driver are fixed; only the recipe is free.

Difficulty is set by the **information exponent** `k`, the index of the first non-vanishing Hermite coefficient of `g`. The benchmark spans three links — `ReLU` (`k=1`), `He_3/sqrt(6)` (`k=3`), and `sign` (`k=1` but non-smooth) — all at `d=100` with Gaussian inputs.

## Prior art / Background / Baselines

- **Textbook recipe (control).** Kaiming init and vanilla SGD with momentum on all parameters, with fresh mini-batches each step. Gap: within the standard budget it often fails to align the first-layer directions with `theta*`, so direction recovery and test MSE stay poor on harder links.
- **Random-feature / kernel models (Rahimi & Recht 2007; Jacot et al. 2018).** Freeze the first layer at its random init and fit only the head. Gap: the features point in fixed random directions that do not adapt to `theta*`, so sample complexity scales polynomially with `d` even for a link that lives on a single direction.
- **Teacher-student single-neuron analyses (Saad & Solla 1995; Goldt et al. 2019).** Study hidden-direction recovery under gradient descent when the student activation equals the true link `g`. Gap: assumes the link is known, removing the non-parametric part of the problem.
- **Information-exponent thresholds for online SGD (Ben Arous, Gheissari & Jagannath 2021).** Characterize online-SGD sample complexity for a single neuron as `n = Theta(d^{k-1})`, driven by the information exponent `k`. Gap: the analysis applies to a single neuron with known link and predicts that for `k >= 3` the per-mini-batch signal is very weak.
- **Classical semi-parametric estimators (Dudeja & Hsu 2018).** Recover `theta*` and `g` via moment-based estimators built around individual Hermite moments. Gap: direction recovery and link fitting use separate, hand-engineered machinery rather than end-to-end gradients on the fixed MLP.

## Fixed substrate / Code framework

A single-run driver is frozen. For each replica it samples a fresh `theta* ~ Uniform(S^{d-1})`, draws `n_train = 32768` Gaussian inputs with noisy targets (`noise_std = 0.1`) and a noise-free test set, builds the network, calls the recipe's four callbacks, loops `max_steps = 8000` mini-batches of size 256 (fresh i.i.d. index each step), evaluates every 500 steps, calls `finalize` once, and evaluates again. The fixed pieces:

- **Model.** `TwoLayerMLP = Linear(d, W) -> ReLU -> Linear(W, 1)` with `W = 256`, scalar output. `fc1` and `fc2` both carry a bias.
- **Data.** `x ~ N(0, I_d)`, `d = 100`; `y = g(<theta*, x>) + eps`, `eps ~ N(0, 0.1^2)` on train, noise-free on test.
- **Direction estimator (fixed, used by the scorer).** `theta_hat = normalize(sum_j |a_j| w_j)`, where `w_j` are the rows of `fc1.weight` and `a_j` the entries of `fc2.weight`.
- **Config handed to the recipe** (`TaskConfig`): `dim`, `width`, `link in {relu, hermite, sign}`, `noise_std`, `n_train`, `batch_size`, `max_steps`, `base_lr = 1e-2`, `weight_decay = 0.0`, `momentum = 0.9`.

## Editable interface

Exactly one region is editable — the `Strategy` class and `build_strategy(config)` in `pytorch-examples/synth_single_index/custom_strategy.py` (lines 176–239). Every recipe fills this same contract:

| Callback | Contract |
|---|---|
| `init_two_layer(net, config)` | Initialise `fc1`, `fc2` in place; may freeze params via `requires_grad_` |
| `make_optimizer(net, config) -> Optimizer` | Return a `torch.optim.Optimizer` over the trainable params |
| `training_step(net, opt, x, y, step, config) -> StepMetrics` | One update on the given mini-batch |
| `finalize(net, x_train, y_train, config)` | Optional global refit (e.g. closed-form ridge) after the step budget |

The default fill is vanilla SGD on the whole net, shown below. The editable recipe replaces exactly this block and nothing else.

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

Three environments, each a link family at `d = 100`: `relu-d100` (`k=1`), `hermite-d100` (`k=3`), and `sign-d100` (`k=1`, non-smooth). Three top-level seeds {42, 123, 456}, one internal replica per seed (distinct `theta*` and data draws), trained from scratch; under 30 minutes per (baseline x env) on one H100. Per environment we report `test_mse_<env>` (noise-free test MSE, lower is better), `direction_recovery_<env> = |<theta_hat, theta*>|` (higher is better, 1 == perfect), and `score_<env> = direction_recovery_<env>` (the primary signal).
