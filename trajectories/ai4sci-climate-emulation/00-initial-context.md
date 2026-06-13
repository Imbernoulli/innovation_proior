## Research question

A global climate model splits the atmosphere into grid cells, but the processes that actually move heat
and water around — radiation, convection, cloud formation, turbulent mixing — live *below* the grid scale.
Run a high-resolution physics module inside each column and it produces, for every coarse atmospheric
state, the sub-grid tendencies that the coarse model would otherwise have to parameterize by hand. The
single thing being designed here is the **neural network architecture** that emulates that map: given a
556-dim column state, predict the 368-dim vector of sub-grid tendencies, with lower Normalized MSE than a
plain MLP. Everything else about the learning problem — the data, the normalization, the splits, the
optimizer, the loss, the multi-budget evaluation — is fixed. The architecture is the only free variable.

## Prior art before the first rung (column-emulation lineage)

The first rung reacts to the established way of emulating ClimSim-style sub-grid physics: a flat
fully-connected regressor trained by mean-squared error. These are the precedents the ladder climbs out of.

- **Flat deep MLP emulator (Gentine et al. 2018; Rasp, Pritchard & Gentine 2018).** Stack the whole column
  state into one vector, push it through several hundred-wide fully-connected layers, regress the tendency
  vector under MSE; it fits the convection map well and is even prognostically stable when coupled back
  into a GCM. Gap: it treats the 556 inputs as an unordered bag — it is blind to the fact that the multi-level
  variables are *profiles* sampled along an ordered vertical axis, so it must relearn the same local vertical
  interaction separately at every height, burning parameters and generalization.
- **Principal-component / linear reduced models (multi-cloud, QTCM lineage).** Project the state onto a few
  leading variance directions and regress on those — interpretable, cheap. Gap: the convection response is
  sharply nonlinear, so a linear projection throws away most of the predictable signal and underfits badly.
- **Deterministic point regression under squared error (the default everywhere above).** Whatever the
  architecture, training against MSE drives the output to the conditional *mean* of the targets. Gap: where
  the target is genuinely stochastic (boundary-layer convection), the mean-regressor squashes the variance,
  and it reports a single global error bar that is wrong everywhere the noise level varies with the state.

The ladder I climb here is exactly the set of architectures that fill the one editable slot below, ordered
weak to strong by their measured NMSE on this task's held-out test split.

## The fixed substrate

A supervised regression loop is frozen and must not be touched. Each example is one atmospheric column:
556 normalized input features (the first `9 * 60 = 540` are nine multi-level variables laid out on the
ordered vertical axis of 60 levels; the remaining ~16 are whole-column scalars), and a 368-dim normalized
target (the first `6 * 60 = 360` are six multi-level tendency profiles; the last 8 are single-level
diagnostics). The pipeline standardizes inputs and outputs with train-only statistics, holds out a
temporally-contiguous test split that training never touches, and trains with **AdamW + cosine-annealed
learning rate**, gradient-norm clipping at 1.0, batch size 1024, and a fixed **`nn.MSELoss`** objective,
with validation-loss early stopping (patience 10). The only per-method knobs exposed outside the
architecture are a `CONFIG_OVERRIDES` dict that may set `learning_rate`, `weight_decay`, or `patience`.

## The editable interface

Exactly one region is editable — the `Custom` model class in `ClimSim/custom_emulator.py` (lines 86–118),
optionally with a `CONFIG_OVERRIDES` fill (lines 173–175). Every method on the ladder is a fill of the same
contract: `__init__(self, input_dim, output_dim)`; `forward(x)` with `x` of shape `(batch_size, input_dim)`
returning shape `(batch_size, output_dim)`; the class name must stay `Custom` so the trainer finds it. The
loss is fixed MSE on the returned tensor, so any method that wants a different training objective must smuggle
it through the model (the trainer is not editable). The first `6 * 60 = 360` outputs are multi-level
tendencies; the last 8 are single-level diagnostics.

The starting point is the scaffold default: a plain 3-layer MLP on the flat vector. Each method on the ladder
replaces exactly this class.

```python
# EDITABLE region of ClimSim/custom_emulator.py (lines 86-118) — default fill: 3-layer MLP
class Custom(nn.Module):
    """Neural network for climate physics emulation.

    Default: simple 3-layer MLP baseline.
    Replace with a better architecture to improve prediction accuracy.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
               First 9*60=540 values are multi-level variables (9 vars x 60 levels),
               remaining values are single-level (scalar) variables.
        Returns:
            Predictions of shape (batch_size, output_dim).
            First 6*60=360 values are multi-level tendencies,
            last 8 values are single-level outputs.
        """
        return self.net(x)
```

## Evaluation settings

The held-out test split is scored by **Normalized MSE** (NMSE = MSE / Var(target), per variable, averaged
over non-constant variables; **lower is better**) as the primary metric. Secondary metrics are **R²**
(higher is better) and **RMSE** (lower is better), plus separate `ml_nmse` (the 360 multi-level tendencies)
and `sl_nmse` (the 8 single-level diagnostics) breakdowns so the two halves of the target can be read apart.
Every architecture is trained at three budgets — **short (30 epochs)**, **medium (100 epochs)**, and
**long (200 epochs)** — and an improvement is expected to be consistent across all three. The reported
baselines are run on a single seed (42) for the final metric; the leaderboard also records seeds {123, 456}.
