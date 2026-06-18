**Problem (from step 2).** The symmetric loss reversed the CIFAR-100 collapse but left `poison_fit` at `0.18`, because to protect convergence it kept the *full* cross-entropy term (`α = 1.0`) — its undiminished `1/f_y` factor still pours large updates into the confidently-wrong (poisoned) examples — and only counterweighted it with a separate RCE term. The robust term never removed the amplification that creates the memorization. The fix is to attenuate `1/f_y` *inside* one loss.

**Key idea.** Cross entropy's per-sample gradient weight is `f_y^{−1}`, MAE's is `f_y^{0}` — the two endpoints of a single exponent. Let the gradient weight be `f_y^{q−1}` for a free `q`; integrating back gives the loss `L_q = (1 − f_y^q)/q`, the negative Box-Cox transform. It interpolates: `q → 0` is cross entropy (recovered by L'Hopital), `q = 1` is MAE. Its gradient is `f_y^q ·` (cross-entropy gradient), so it is cross entropy with each sample down-weighted by `f_y^q ∈ [0,1]` — biting hardest on the low-confidence, likely-poisoned samples.

**Why it works.** The `f_y^q` factor removes the `1/f_y` amplification at its source (vs the symmetric loss, which kept it and pushed back). Bounding the symmetry defect `Σ_j L_q(f,e_j)` between `(K−K^{1−q})/q` and `(K−1)/q` and pushing it through the affine-risk argument gives a *graded* noise-tolerance bound with width `∝ K^{1−q} − 1`: zero at `q = 1` (MAE's exact tolerance), widening toward `q = 0`. So `q` continuously trades robustness (large `q`) against learnability (small `q`); the middle keeps both.

**Hyperparameters.** `q = 0.7` — robust enough to suppress overfitting at these noise rates while the `f_y^{q−1}` weight keeps the gradient curvature CIFAR-100's hard convergence needs (the harness fixes one global objective, so no per-dataset `q`). Floor the gathered probability at `1e-8` before the power to avoid underflow on confidently-wrong samples. The truncated/self-paced variant (which tightens the bound further) is unreachable here — it needs per-sample state across minibatches, which the contract does not expose. No `epoch` use.

```python
class RobustLoss:
    """Generalized cross-entropy for noisy labels."""

    def __init__(self):
        self.q = 0.7

    def compute_loss(self, logits, labels, epoch):
        probs = torch.softmax(logits, dim=1)
        p = probs.gather(1, labels[:, None]).clamp_min(1e-8)
        return ((1.0 - p.pow(self.q)) / self.q).mean()
```
