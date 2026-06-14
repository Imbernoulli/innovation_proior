**Problem.** VICReg (90.24 agg) and barlow (90.01 agg) tied within ~0.2, but *neither* is stable across
backbones — VICReg slipped below barlow and below its own ResNet-18 on ResNet-34 (89.5), breaking the
monotone-with-scale. Both pin only the embedding's *second moments* (barlow zeros a cross-correlation;
VICReg floors per-dimension variance and zeros per-branch covariance), leaving every higher moment free —
which is where the per-backbone wobble lives. The fix: stop pinning second moments; pin the *whole*
embedding distribution to a target.

**Key idea — derive the target, then regularize toward it.**
- *Linear probe* (the leaderboard metric): ridge bias is worst along weak eigendirections for anisotropic
  Σ, and unregularized variance `σ²Σ_j 1/λ_j` is minimized (Jensen) at equal eigenvalues ⇒ **isotropy**.
- *Nonlinear probe*: integrated squared bias depends on Fisher information `J(p)=∫‖∇log p‖²p`. At fixed Σ,
  Cramér–Rao gives `J(p) ≥ tr(Σ⁻¹)` with equality **iff Gaussian**; minimizing `tr(Σ⁻¹)` under a scalar
  constraint forces Σ = sI ⇒ **isotropic Gaussian `N(0, I)`**. This subsumes decorrelation *and* equal
  variance *and* controls higher moments.
- *Match it cheaply by slicing.* Direct multivariate normality tests are O(B²) and batch-coupled.
  Cramér–Wold: `X =d Y` iff all 1-D projections match; for `N(0,I)` every unit projection is `N(0,1)`. So
  test many random 1-D projections for standard-normality, **average** over directions (the max gives
  sparse gradients), and resample directions so the sphere accumulates coverage. Per-slice test =
  **Epps–Pulley** (characteristic-function): differentiable average, identifiable, gradient/curvature-
  bounded — unlike moments (non-identifiable / explode) or CDF tests (need sorting).

**Why this implementation, not the full machinery.** The harness runs BCS (Batched Characteristic
Slicing), a stripped form: `num_slices = 256` directions seeded by an internal `step` counter (no cross-
device sync — single device); frequency grid `t = linspace(−3, 3, 10)` integrated over the *full*
symmetric grid with `torch.trapz` (not the evenness-halved 17-knot form); complex CF via `(1j*x_t).exp()`
(not the cos/sin-mean real form); window `exp(−t²/2)` as both target CF and weight. These trade quadrature
resolution and DDP-readiness for a compact single-device module without changing the algorithm. The
load-bearing config choice: `CONFIG_OVERRIDES = {"proj_output_dim": 128}` — Gaussianity-on-random-
projections concentrates better at low output dim (256 slices pin a 128-sphere far tighter than a 2048-
sphere), and the upstream recipe ranks 2048→128 top. SIGReg narrows hardest (128 vs VICReg 1024, barlow
2048).

**Hyperparameters.** `num_slices = 256`, `lmbd = 10.0`, grid `t_min=−3, t_max=3, n_points=10`,
`proj_output_dim = 128`.

**What to watch.** Expect low-90s on every backbone, clearing both prior rungs on aggregate, and — the
telling part — the *tightest* per-backbone spread (no ResNet-34-style dip), because the wobble was the
signature of un-pinned higher moments; the biggest margin should fall on ResNet-34 where second-moment
pinning was weakest. Merely *level* with VICReg would indict the coarse 10-knot full-grid quadrature in
the 128-dim space, not the derived target.

```python
class CustomRegularizer(nn.Module):
    """BCS (Batched Characteristic Slicing) regularizer for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def _epps_pulley(self, x, t_min=-3, t_max=3, n_points=10):
        t = torch.linspace(t_min, t_max, n_points, device=x.device)
        exp_f = torch.exp(-0.5 * t ** 2)
        x_t = x.unsqueeze(2) * t
        ecf = (1j * x_t).exp().mean(0)
        err = exp_f * (ecf - exp_f).abs() ** 2
        T = torch.trapz(err, t, dim=1)
        return T

    def forward(self, z1, z2):
        dev = z1.device
        with torch.no_grad():
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A = A / A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (self._epps_pulley(view1).mean() + self._epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2)
        total_loss = invariance_loss + self.lmbd * bcs
        return {
            "loss": total_loss,
            "bcs_loss": bcs,
            "invariance_loss": invariance_loss,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Paper sigreg.yaml uses 2048->128 — SIGReg's Gaussianity test on random
# projections concentrates better at low output dims (paper rank-1: 91.02%).
CONFIG_OVERRIDES = {"proj_output_dim": 128}
```
