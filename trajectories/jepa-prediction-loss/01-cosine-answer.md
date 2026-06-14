**Problem.** The temporal-JEPA prediction loss must turn the residual between a predicted latent
feature map and the encoder's target feature map (`state`, `predicted`, both `[B, C, T, H, W]`) into
one scalar the predictor descends. The scaffold default returns a constant zero — no learning signal.
The first design decision is the most basic: should the loss match the *magnitude* of the latents or
only their *direction*?

**Key idea.** Bet on direction. Since the trainer already adds a Variance–Covariance regularizer that
governs feature *scale* (a variance hinge per dimension plus a covariance decorrelation), grade the
predictor purely on the *pattern* of activation across channels and let the regularizer own magnitude.
Concretely, at every spatial-temporal location L2-normalize the length-`C` channel vector of both
`state` and `predicted` onto the unit sphere, take their dot product (the cosine of the angle between
them), and use `1 − cosine` as the per-location cost; average over batch, time, and the spatial grid.
Normalizing along the channel axis (`dim=1`) before the dot product is what makes the loss invariant to
how *long* either vector is, so it is blind to magnitude and sensitive only to direction.

**Why it is the floor.** It commits to one extreme of the magnitude-vs-direction question. The bet has
a clean motivation — don't fight the regularizer over scale — but it is a real bet: the loss is exactly
invariant to rescaling either argument, so the predictor is never pushed to match feature *strength*,
and the directional gradient is the component of the target orthogonal to the prediction scaled by
`1/‖p‖`, which is weak precisely when features are short. The premise that the VC regularizer already
sets magnitude is only half true — it bounds each dimension's *spread* away from zero, not the *scale
agreement* between predicted and target — so any magnitude information the detection probe needs is
left untrained. It is a legitimate, well-posed self-supervised loss that gives the predictor a real
signal where the default gave none; it is the floor because of what it deliberately discards.

**Hyperparameters.** None. Normalization axis `dim=1` (channels); reduction = mean over all remaining
axes. Same code serves all three model widths.

**Step-1 edit.** Replace `CustomPredictionLoss.forward` with the per-location cosine loss. This is the
literal scaffold fill of the editable region (`custom_prediction_loss.py`, lines 36–54).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Cosine similarity prediction loss for temporal JEPA.

    Per spatial-temporal location, L2-normalize the length-C channel vectors of
    state and predicted onto the unit sphere and take their dot product (the cosine
    of the angle between them); the cost 1 - cosine is zero at perfect directional
    agreement. Invariant to the magnitude of either latent -- grades the predictor on
    direction only, leaving feature scale to the VC regularizer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        s = F.normalize(state, dim=1)               # unit channel vectors at each location
        p = F.normalize(predicted, dim=1)
        return (1 - (s * p).sum(dim=1)).mean()       # mean angular disagreement over B,T,H,W
```
