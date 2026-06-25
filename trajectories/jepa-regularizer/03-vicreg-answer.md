**Problem.** Barlow recovered the collapse hole — 89.12 / 90.28 / 90.62 across ResNet-18/34/50, monotone
with backbone scale, no straggler. But its whole objective rides one *coupled* cross-correlation matrix
(each entry mixes a dimension of branch A with one of branch B) and needs embedding *standardization* to
make the diagonal-to-1 target meaningful. Collapse-prevention is entangled across the two branches and a
normalization, and ResNet-18 sat lowest (89.12) — the opening for a cleaner member of the same family.

**Key idea.** Split collapse-prevention into two explicit terms applied to *each branch separately*, plus
plain-MSE invariance — no negatives, no cross-branch matrix, no embedding standardization. (1)
**Variance**: a hinge keeping each dimension's batch std above a floor γ, `mean_j max(0, γ − √(Var+ε))`.
Using the **standard deviation, not the variance**, is essential — `d√Var/dVar → ∞` as Var → 0, so the
restoring gradient survives at collapse, where the variance-in-the-hinge gradient vanishes. Forbids
*trivial* collapse. (2) **Covariance**: push every off-diagonal of each branch's covariance to zero,
decorrelating the dimensions. Forbids *informational* collapse. Variance alone permits copying one
direction into all dimensions; covariance alone collapses outright (the cheapest zero is a constant).
Together: variance gives the covariance term something to spread; the covariance, left *unnormalized*
(the variance term owns the scale), needs no standardization.

**Why this implementation, not the generic balanced form.** The harness leans on covariance and a narrowed
projector rather than the generic balanced weighting: `std_coeff = 1.0`, `cov_coeff = 100.0`,
`std_margin = 1.0`, and `CONFIG_OVERRIDES = {"proj_output_dim": 1024}`. The two interlock — a narrower
1024-wide embedding has fewer off-diagonal pairs, so a heavy cov weight can fully decorrelate it within
budget, and 1024 is the projector width the upstream "impact of the projector" comparison ranks best for
this method on CIFAR-10. Variance term holds the floor; the heavy covariance term does the spreading in a
deliberately narrower, more fully decorrelated space. (Unlike barlow, which kept the default 2048
projector.)

**Hyperparameters.** `std_coeff = 1.0`, `cov_coeff = 100.0`, `std_margin = 1.0`, invariance coeff 1.0,
`proj_output_dim = 1024`.

**What to watch.** Expect at-or-slightly-above barlow on the larger backbones (high-89s to low-91s); the
bet is ResNet-18 (barlow's weakest at 89.12) matching or edging past, since the decoupled per-branch
geometry in a 1024-wide space should transfer at least as well. A *lower* ResNet-18 would mean the
heavy cov weight over-decorrelates before the representation is informative. A near-tie with barlow on
aggregate confirms that the *kind* of anti-collapse term matters more than its exact formulation — which
points the final rung at pinning the *whole* embedding distribution to a target, not only its second
moments.

```python
class CustomRegularizer(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization."""

    def __init__(self, std_coeff=1.0, cov_coeff=100.0, std_margin=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_margin = std_margin

    def _std_loss(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cov_loss(self, x):
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        return self._off_diagonal(cov).pow(2).mean()

    def forward(self, z1, z2):
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self._std_loss(z1) + self._std_loss(z2)
        cov_loss = self._cov_loss(z1) + self._cov_loss(z2)
        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# The "impact of the projector" comparison ranks VICReg's best
# projector as 2048->1024 (90.12% on CIFAR-10 ResNet-18, 300 epochs).
CONFIG_OVERRIDES = {"proj_output_dim": 1024}
```
