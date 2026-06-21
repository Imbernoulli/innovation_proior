## Research question

A global climate model divides the atmosphere into grid cells, but heat, water, radiation, convection, clouds, and turbulent mixing all operate below the grid scale. For each coarse column state, a high-resolution physics module can compute the sub-grid tendencies that the coarse model otherwise parameterizes by hand. The design problem here is the neural network architecture that emulates that map: given a 556-dimensional column state, predict the 368-dimensional vector of sub-grid tendencies with lower Normalized MSE than a plain MLP. The data, normalization, splits, optimizer, loss, and multi-budget evaluation are fixed; only the architecture changes.

## Prior art / Background / Baselines

Three standard approaches set the starting point.

- **Flat deep MLP emulator.** Stack the full column state into one vector and regress the tendency vector through several wide fully-connected layers under MSE. Gap: it treats the inputs as an unordered bag and ignores the fact that the multi-level variables form ordered vertical profiles, so it must relearn the same local vertical interaction separately at every height.
- **Principal-component / linear reduced models.** Project the state onto a few leading variance directions and regress linearly on those. Gap: the convection response is strongly nonlinear, so a linear projection discards most of the predictable signal and underfits.
- **Deterministic point regression under squared error.** Train the architecture to output a single point estimate and penalize squared error. Gap: where the target is genuinely stochastic, the conditional mean suppresses variance and a single global error bar misrepresents regions with state-dependent noise.

## Fixed substrate / Code framework

Each example is one atmospheric column: 556 normalized input features (540 multi-level values on 60 ordered vertical levels plus ~16 whole-column scalars) and a 368-dimensional normalized target (360 multi-level tendencies plus 8 single-level diagnostics). The pipeline standardizes inputs and outputs with train-only statistics, holds out a temporally-contiguous test split, and trains with AdamW, cosine-annealed learning rate, gradient-norm clipping at 1.0, batch size 1024, and fixed `nn.MSELoss`, with validation-loss early stopping (patience 10). The only per-method knobs outside the architecture are a `CONFIG_OVERRIDES` dict that may set `learning_rate`, `weight_decay`, or `patience`.

## Editable interface

Only the `Custom` model class in `ClimSim/custom_emulator.py` is editable. It must satisfy the contract `__init__(self, input_dim, output_dim)` and `forward(x)` returning a tensor of shape `(batch_size, output_dim)`. The class name must stay `Custom`, and the loss is fixed MSE on the returned tensor. The first 360 outputs are multi-level tendencies; the last 8 are single-level diagnostics.

The default fill is a plain 3-layer MLP on the flat vector.

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

The held-out test split is scored by Normalized MSE (NMSE = MSE / Var(target), per variable, averaged over non-constant variables; lower is better) as the primary metric. Secondary metrics are R² (higher is better) and RMSE (lower is better), plus separate `ml_nmse` (multi-level tendencies) and `sl_nmse` (single-level diagnostics). Every architecture is trained at three budgets: 30 epochs, 100 epochs, and 200 epochs. Improvements should be consistent across all three. Reported baselines use seed 42; the leaderboard also records seeds 123 and 456.
