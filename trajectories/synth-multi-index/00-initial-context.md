## Research question

A *fixed* two-layer ReLU MLP — `Linear(d, 256) → ReLU → Linear(256, 1)`, `d = 128`, bias on both
layers — has to fit a multi-index target

```
y = g(U*ᵀ x),   x ~ N(0, I_d),   U* ∈ R^{d×r} column-orthonormal (Haar),   r ∈ {2, 3, 4},
g(z) = (1/√r) Σ_{i=1..r} He₃(z_i),   He₃(z) = z³ − 3z.
```

All label structure lives in the unknown `r`-dimensional teacher subspace `span(U*)`; the input is
isotropic. The only design freedom is the **training recipe** for this frozen architecture, through
four hooks: weight initialization, training data construction, optimizer hyperparameters, and the
per-step update rule. The architecture, width, batch size (128), training budget (8000 mini-batch
steps), and evaluation are fixed. The task is to recover both the subspace `span(U*)` and the link
`g`. The link has **information exponent 3**: the lowest Hermite degree at which `g` correlates with
a single direction is 3, so gradient signal toward an unfound teacher direction is weak at a random
start.

## Prior art / Background / Baselines

Three baselines frame the difficulty.

- **Kernel / random-feature methods (fixed-feature regime).** Freeze the first-layer features and fit
  only a linear readout. A degree-3 target costs `Ω(d³)` samples under this regime.
- **Single-index information-exponent analysis.** For a target reading one direction with information
  exponent `s`, one-pass SGD on the first layer needs roughly `d^{s-1}` samples/steps; the gradient
  pull toward the true direction is `O(overlap^{s-1})` and vanishes at small overlap.
- **Multi-direction saddle-to-saddle SGD.** For several directions, SGD climbs saddle-to-saddle,
  picking up Hermite components in increasing order; a direction leaves the equator only after the
  directions it is coupled to are already found.

The starting point is vanilla joint SGD on the fixed MLP, and every subsequent rung modifies the same four-hook interface.

## Fixed substrate / Code framework

A driver builds the MLP, calls `init_model`, builds an SGD/AdamW optimizer with separate parameter
groups for the inner (`d → W`) and outer (`W → 1`) layers from `get_optimizer_config`, samples
`make_dataset` once per teacher, then iterates `training_step` for `max_steps = 8000` mini-batch
updates (batch 128, reshuffled each epoch). After training it evaluates on a fresh 8192-point test
set. It is frozen and may not be touched. Helpers the hooks may call: `teacher_outputs(x, teacher)`
(the label oracle `y = g(U*ᵀ x)`), `hermite3`, `link_function`, and the `TaskConfig` fields
(`n_features=128`, `rank`, `hidden_width=256`, `batch_size=128`, `max_steps=8000`,
`max_train_examples=200_000`). The optimizer is constructed *outside* `training_step`; the step
receives both the live `optimizer` and the immutable `OptimizerConfig` it was built from. The driver
also injects Langevin noise itself when `noise_std > 0` (it adds
`noise_std·√(2·lr)·N(0,I)` to each gradient before `optimizer.step()`), so a recipe can opt into
parameter noise purely through the config.

## Editable interface

Exactly one region of `pytorch-examples/synth_multi_index/custom_strategy.py` (lines 251–325) is
editable: four functions. Every rung on the ladder is a fill of this same contract.

- `init_model(model, config)` — initialize the two linear layers **without** looking at the teacher
  subspace (it only sees the model and `TaskConfig`).
- `make_dataset(config, teacher, seed)` — return `(x, y)` (or `{"x", "y"}`); `x` is `[n, d]`, `y` is
  `[n]`, `n ≤ max_train_examples`. `teacher` is the label oracle only.
- `get_optimizer_config(config)` — return a dict with `optimizer` (`"sgd"`/`"adamw"`), `lr_inner`,
  `lr_outer`, `wd_inner`, `wd_outer`, `momentum`, `noise_std` (inner = first layer, outer = readout).
- `training_step(model, optimizer, optimizer_config, batch_x, batch_y, step, config)` — exactly one
  optimizer update; return a dict with at least `loss`. A step may freeze a layer, project the
  first-layer rows onto the unit sphere, inject extra noise, or run a closed-form ridge solve for the
  readout.

The starting point is the scaffold default: standard Kaiming init, a fixed `n = 4096` Gaussian set,
plain SGD on both layers at `lr = 5e-2`, no decay, no noise, and a single joint squared-loss step.

```python
# EDITABLE region of custom_strategy.py — default fill (vanilla joint SGD)
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Default Kaiming-uniform init for both linear layers; no teacher info."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def make_dataset(config: TaskConfig, teacher: torch.Tensor, seed: int):
    """Fixed Gaussian training set of size n = 4096."""
    g = torch.Generator().manual_seed(seed)
    num_examples = 4_096
    x = torch.randn(num_examples, config.n_features, generator=g)
    y = teacher_outputs(x, teacher)
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, object]:
    """Plain SGD on both layers; no momentum, no weight decay, no noise."""
    return {
        "optimizer": "sgd",
        "lr_inner": 5e-2, "lr_outer": 5e-2,
        "wd_inner": 0.0, "wd_outer": 0.0,
        "momentum": 0.0, "noise_std": 0.0,
    }


def training_step(model, optimizer, optimizer_config, batch_x, batch_y, step, config):
    """Single squared-loss update applied to both layers jointly."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    preds = model(batch_x).view(-1)
    loss = ((preds - batch_y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
```

## Evaluation settings

The same scaffold runs at three teacher ranks — `r2` (d=128, r=2; easiest), `r3` (r=3), `r4` (r=4;
hardest, more directions to align) — each over three top-level seeds {42, 123, 456} (one teacher and
one data ordering per seed). Three metrics per run:

| Metric | Direction | Definition |
|--------|-----------|------------|
| `test_mse` | lower better | MSE on a fresh 8 192-point test set |
| `subspace_err` | lower better | `‖P_Û − P_{U*}‖_F`, `Û` = top-`r` right-singular subspace of the first layer's weights |
| `score` | higher better | `exp(−subspace_err² / r) · exp(−test_mse)`, in (0, 1] |

The leaderboard aggregates `score`, `test_mse`, `subspace_err` across the three ranks by a geometric
mean of per-rank weighted-mean composites. The two metrics that matter are coupled: a low
`subspace_err` (the first layer found `V*`) is the precondition for a low `test_mse` (the readout can
then fit the cubic), and `score` rewards both.
