**Problem.** Uncertainty weighting recovered the capacity-scarce ResNet-20 but *regressed* on VGG-16-BN,
because it learns a single *static* per-task scale (fixed point tied to the loss *level*) that drifts only
slowly and cannot track the fine and coarse tasks learning at different *rates* over the 200-epoch schedule.
The lever left untouched is the *rate of descent*, not the level.

**Key idea (Dynamic Weight Average).** Weight each task by how *slowly* its loss is currently descending.
Form the scale-free per-task rate `r_k = L_k(now) / L_k(prev)` — a unit-free ratio that is comparable
across the 100-way and 20-way heads. A larger `r_k` means a slower (or worsening) learner, so it should get
*more* weight; map the two ratios to a normalized weight with a temperature softmax,
`w_k = K · softmax(r_k / T)` (`K = 2`, so weights sum to `K` and the mean weight stays 1, keeping the loss
scale fixed). `T = 2.0` is the conservative contrast: `T → ∞` recovers equal weighting, `T → 0` becomes a
hard argmax onto the single slowest task. Bootstrap to equal weights when there is no history yet.

**Why it should help here.** It directly targets uncertainty's failure mode: instead of a slow static scale,
the weight is a fast function of the live descent rate, so it backs off whichever task (e.g. the
quickly-mastered coarse head on VGG) has stopped improving and feeds weight to the lagging one. Bigger
*ratio* → more weight is the opposite push from uncertainty's bigger *loss* → less weight, by design.

**This task's implementation (per-batch, not per-epoch).** The paper's ratio is between *epoch-averaged*
losses; this interface runs per minibatch and can only cheaply keep the *previous call's* loss, so the
ratio is `L_k(this batch) / L_k(previous batch)` — a noisier per-batch rate, with `prev_losses` overwritten
every step. The `epoch == 0` guard pins weights to uniform for the whole first epoch; `T = 2.0` heavily
damps the per-batch noise. DWA registers **no learnable parameters** — it is pure state (a previous-loss
buffer) plus a softmax, and the weights are computed from *detached* losses, so gradient flows only through
the live `w_k · L_k` product, never through the weighting.

**Hyperparameters.** `T = 2.0` (temperature); weights bootstrap to `1` (epoch 0 / no history); `1e-8` floor
on the ratio denominator. No learnable parameters.

```python
# EDITABLE region of pytorch-vision/custom_mtl.py (lines 195-216) — step 3: Dynamic Weight Average
class MultiTaskLoss(nn.Module):
    """Dynamic Weight Average (Liu et al., 2019).

    Weights tasks by relative loss change rate with temperature.
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        self.prev_losses = None
        self.T = 2.0  # temperature

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = torch.stack([fine_loss, coarse_loss])
        if self.prev_losses is None or epoch == 0:
            weights = torch.ones(2, device=losses.device)
        else:
            ratios = losses.detach() / (self.prev_losses + 1e-8)
            weights = 2 * F.softmax(ratios / self.T, dim=0)
        self.prev_losses = losses.detach().clone()
        return (weights * losses).sum()
```
