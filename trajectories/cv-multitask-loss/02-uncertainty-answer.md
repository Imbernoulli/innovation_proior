**Problem.** PCGrad fixed only gradient *directions* and was blind to the fine/coarse *magnitude*
imbalance — the lever that actually decides fine accuracy on a capacity-scarce backbone. The obvious
fix, learning bare weights `w_i` in `Σ w_i L_i`, collapses: `∂L/∂w_i = L_i ≥ 0` drives every `w_i → 0`
and the model learns nothing. A principled, learnable, non-collapsing weighting is needed.

**Key idea (homoscedastic uncertainty weighting).** Read each task loss as a negative log-likelihood
with a per-task observation-noise scale `σ_i`, and learn `σ_i` jointly with the network. Maximizing the
joint likelihood yields, for each task, an **inverse-variance weight** `1/σ_i²` on its loss plus a
**logarithmic scale penalty** `log σ_i`. The inverse-variance weight down-weights the noisier/larger-scale
task; the `log σ_i` term is the anti-collapse regularizer the bare weights lacked (sending `σ_i → ∞` to
zero a task's weight drives `log σ_i → +∞`). For a softmax head the same shape follows via a temperature
`σ²` on the logits, with the messy log-sum-exp regularizer collapsing to `log σ` under an approximation
exact at `σ = 1`.

**Why it should help here.** It attacks exactly the axis PCGrad left untouched. The weighting is learned,
not grid-searched, and *dynamic*: at the fixed point `σ_i² = L_i`, so each task's weight tracks its
current loss — equal early (all losses large), then the easier 20-way coarse task's weight falls as its
loss drops. That adaptivity is what a capacity-scarce ResNet-20 trunk needs.

**Stability and init.** Train the **log-variance** `s_i := log σ_i²`, not `σ_i`: then `1/σ_i² = exp(-s_i)`
is always positive (no divide-by-zero) and `s_i` is unconstrained (plain SGD steps it). The per-task term
`exp(-s_i) L_i + s_i` is strictly convex in `s_i` with minimum at `s_i = log L_i`, so it is robust to
initialization. Initialize `s_i = 0` (`σ_i² = 1`, equal weighting) — a neutral start, no tuning.

**This task's implementation.** Pure loss-weighting, so unlike PCGrad it needs no graph walk and no
`torch.autograd.grad` — the interface hands exactly the two scalar losses it reads. Register one
log-variance `nn.Parameter` per task; the loop's optimizer (`model.parameters() + mtl_loss.parameters()`)
trains them with the network. The forward accumulates `exp(-s_i) · L_i + s_i` over the two tasks.

**Hyperparameters.** `log_vars` initialized to `0` (equal weighting); 2 learnable scalars total, no other
tuning.

```python
# EDITABLE region of pytorch-vision/custom_mtl.py (lines 195-216) — step 2: uncertainty weighting
class MultiTaskLoss(nn.Module):
    """Uncertainty weighting (Kendall et al., 2018).

    Learns per-task log-variance: loss_i / exp(log_var_i) + log_var_i.
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = [fine_loss, coarse_loss]
        total = sum(
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(2)
        )
        return total
```
