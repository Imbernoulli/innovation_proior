**Problem.** The cosine baseline graded the predictor only on the *direction* of the latent feature
vectors and was invariant to their magnitude. Its detection AP held at the small and base widths
(0.5613 / 0.5475) but fell to 0.4664 at the large width — a monotone decline *with model width*, the
signature of a loss blind to magnitude the bigger model has more of. The detection probe reads
activation strength, so the untrained feature scale costs the most exactly where there is the most of
it. The prediction loss must match latents in both direction *and* magnitude.

**Key idea.** Use the squared Euclidean distance between the predicted and target channel vectors at
each location: the per-coordinate residual `r = state − predicted`, squared and averaged over the
`[B, C, T, H, W]` map. Unlike cosine, it keeps the norms — a prediction pointing the right way but the
wrong length now carries real cost — and its gradient `2(predicted − state)` pulls along the *full*
residual, training magnitude as well as direction. The per-coordinate cost `ρ(r) = r²` is the simplest
even, smooth, increasing-in-`|r|` function: a flat quadratic basin at the optimum (smooth gradient that
eases to zero, where cosine's stayed finite) and a full-residual pull that does not weaken for short
feature vectors (where cosine's `1/‖p‖` directional gradient did).

**Why it is principled, not arbitrary.** Squared error is the Gaussian maximum-likelihood match — it is
the right default when residuals are treated as roughly Gaussian with no specific reason to expect heavy
tails — and it is the loss the VICReg invariance term defaults to. It is the canonical
magnitude-sensitive feature-matching cost the JEPA lineage starts from, the obvious correction to a loss
that discarded magnitude. (The remaining bet — that no heavy-tailed roll-out residuals dominate the
gradient — is what a later rung tests.)

**Why mean, not sum.** A summed squared loss scales with `B·C·T·H·W`, so it grows with channel count
and resolution and would hand the large model a numerically larger objective purely for being wider —
coupling the effective learning rate to model width, the very across-width sensitivity cosine exposed.
Mean reduction makes the term per-coordinate and comparable across the three widths, every coordinate
weighted equally. This is exactly `F.mse_loss` with its default `reduction="mean"`.

**Hyperparameters.** None. Reduction = mean over all `[B, C, T, H, W]` coordinates; same code for all
three widths.

**Step-2 edit.** Replace `CustomPredictionLoss.forward` with the mean squared error. This is the literal
scaffold fill of the editable region (`custom_prediction_loss.py`, lines 36–54).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """MSE prediction loss for temporal JEPA.

    Mean squared Euclidean distance between the predicted and target latent feature
    maps, averaged over all [B, C, T, H, W] coordinates. Sensitive to both direction
    and magnitude (unlike cosine); gradient 2*(predicted - state) pulls along the full
    residual and trains feature scale. The Gaussian maximum-likelihood feature match.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.mse_loss(state, predicted)          # mean over [B, C, T, H, W]
```
