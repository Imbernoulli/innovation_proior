**Problem.** The naive MSE floor collapsed: 14.56 / 13.77 / 17.28 across ResNet-18/34/50, a tight band
near the 10-class chance floor. The bare invariance objective has the constant map as a global
minimizer, and the detached linear probe has nothing to separate. The fix must make the collapsed
configuration a *high*-loss state — and on this contract (two embedding tensors → one scalar, batch 256,
no negatives slot, no second network) the information-maximization family is the only clean fit.

**Key idea.** Barlow's redundancy-reduction principle: a good representation is invariant to the
augmentation *and* has decorrelated (non-redundant) dimensions. Standardize each view's embeddings
across the batch and form the D×D cross-correlation matrix `C = bn(z1).T @ bn(z2) / B`. Drive it to the
identity: `L = Σ_i (1 − C_ii)² + λ·Σ_{i≠j} C_ij²`. Diagonal→1 is invariance; off-diagonal→0 is
decorrelation. Standardization makes a zero-variance (constant) feature unable to self-correlate to 1,
so collapse is a high-loss state by construction; the off-diagonal term forbids copying one direction
into all features. No negatives, no predictor, no stop-gradient, no momentum encoder.

**Why this implementation, not the canonical large-scale form.** (1) Standardization is a `nn.LazyBatchNorm1d(
affine=False)` so the module registers in `__init__` but materializes D on the first forward (the loss
is constructed with no args). (2) `λ = 0.0051` (the standard redundancy weight; ~D² off-diagonals vs D
diagonals). (3) A `scale_loss = 0.1` multiplier on the whole loss — the raw summed loss is ~10³–10⁴, and
LARS' adaptive rescaling `‖p‖/(‖g‖+…)` starves under that gradient norm, leaving the diagonal stuck;
0.1 tames it. This is the CIFAR-scale recipe (proj `2048→2048`, batch 256, the three CIFAR
backbones), *not* the ImageNet-scale recipe (proj 8192, `scale_loss≈0.024`, batch 2048, 1000 epochs),
which at this budget would leave the diagonal stuck. No `CONFIG_OVERRIDES`.

**Hyperparameters.** `lambd = 0.0051`, `scale_loss = 0.1`, projector `2048 → 2048` (default).

**What to watch.** The two-digit band should vanish — high-80s on every backbone, recovering the bulk of
the ~75-point hole — and the cross-backbone order should invert (the larger backbones genuinely separate
classes rather than riding a chance cushion). A straggler stuck near 10% would mean the LARS-starvation
failure (scale too small / projector too wide). A point or two left on the table versus a per-branch
variance+decorrelation method is the opening for the next rung.

```python
class CustomRegularizer(nn.Module):
    """Barlow Twins redundancy-reduction regularizer.

    NB on scale_loss: the canonical 8192-projector recipe includes
    a `--scale-loss 0.024` multiplier. Without it the raw loss is on the
    order of 1e3-1e4, and LARS' adaptive rescaling
    (lars_lr = p_norm / (g_norm + ...)) starves the optimizer so the
    diagonal of the cross-correlation matrix never approaches 1.
    The 8192 projector uses scale_loss=0.024; our CIFAR setup below uses
    scale_loss=0.1.
    """

    def __init__(self, lambd=0.0051, scale_loss=0.1):
        super().__init__()
        self.lambd = lambd
        self.scale_loss = scale_loss
        # Use LazyBatchNorm1d so the module is registered in __init__
        # (with proper to(device)/dtype propagation) but the feature dim
        # is materialized on the first forward call.
        self.bn = nn.LazyBatchNorm1d(affine=False)

    @staticmethod
    def _off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        B = z1.shape[0]

        # Cross-correlation matrix.
        c = self.bn(z1).T @ self.bn(z2)
        c = c / B

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        total_loss = self.scale_loss * (on_diag + self.lambd * off_diag)

        return {
            "loss": total_loss,
            "on_diag": on_diag,
            "off_diag": off_diag,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Use the CIFAR-10 Barlow Twins recipe (proj=2048,
# scale_loss=0.1) instead of the ImageNet-scale recipe
# (proj=8192, scale_loss=0.024, batch=2048, epochs=1000). Our setup is
# CIFAR-10, batch=256, ResNet-{18,34,50}, LARS with eta=0.02 and
# clip_lr=True. The 8192 recipe needs epochs=1000 + batch=2048 to
# converge — at our 100-epoch budget it leaves the diagonal stuck
# (see logs from v3: rn34 only reaches 10%).
CONFIG_OVERRIDES = {}
```
