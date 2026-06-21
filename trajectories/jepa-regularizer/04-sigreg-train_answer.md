VICReg landed where I bet it would and the numbers settle the family question: 89.85 on ResNet-18, 89.5 on ResNet-34, 91.38 on ResNet-50 — aggregate 90.24 against Barlow's 90.01, a near-tie. On ResNet-18 it edged Barlow (89.85 vs 89.12) and on ResNet-50 it pulled clearly ahead (91.38 vs 90.62), but on ResNet-34 it slipped below Barlow *and* below its own ResNet-18 (89.5), breaking the monotone-with-scale that both Barlow and the naive cushion showed. That non-monotonicity is the tell. The heavy `cov_coeff = 100` in the narrow 1024 space over-decorrelates ResNet-34 before its representation is fully informative — but the deeper lesson is not "VICReg beats Barlow." It is that two different second-moment formulations land within ~0.2 of each other on aggregate while *neither* is stable across all three backbones. Both pin only the embedding's *second moments* — Barlow zeros a cross-correlation, VICReg floors per-dimension variance and zeros per-branch covariance — and a method that controls only the first two moments leaves every higher moment free, which is precisely where the per-backbone wobble lives. So the opening is forced: stop pinning second moments and pin the *whole* embedding distribution to a target.

I propose **SIGReg** — sliced isotropic-Gaussian regularization — which first *derives* the target distribution from the probe worst case and then regularizes toward it directly, rather than picking a statistic and hoping the geometry transfers. The honest constraint is that I do not know the downstream task at pretraining time, so I should pick the embedding geometry best in the *worst case* over unknown tasks. For the linear probe this harness actually uses, fit ridge regression to unknown labels on the frozen features: the ridge bias along the *weakest* eigendirection is $\lambda/(\lambda_{\min}+\lambda)$ versus $\lambda/(\bar\lambda+\lambda)$ for an isotropic spectrum of the same energy, and since $\lambda_{\min}<\bar\lambda$ whenever the spectrum is not flat, anisotropy strictly hurts some task; and the unregularized variance $\sigma^2\sum_j 1/\lambda_j$ is minimized at fixed mean eigenvalue, by Jensen ($1/x$ convex), exactly when all eigenvalues are equal. So the linear probe says the embeddings should be **isotropic** — which already explains the wobble, since Barlow and VICReg push toward isotropy through second moments but never pin it, so each backbone lands at a different anisotropy.

Isotropy fixes only the covariance shape, and a worst-case-over-tasks argument should care about more than two moments. Push to a nonlinear probe, a radius-kNN or kernel estimator; its leading integrated squared bias depends on the Fisher-information functional $J(p)=\int\lVert\nabla\log p\rVert^2\,p\,dx$ of the embedding density. The identity estimator $T(X)=X$ is unbiased for the location of $p_\theta(x)=p(x-\theta)$ with covariance $\Sigma$, so Cramér–Rao gives $\Sigma\succeq I(\theta)^{-1}$, hence $I\succeq\Sigma^{-1}$, hence $J(p)=\mathrm{tr}\,I\geq\mathrm{tr}(\Sigma^{-1})$, with equality iff the score is affine in $x$ — i.e. iff $p$ is **Gaussian**. Among all densities of a given covariance, the Gaussian uniquely minimizes the functional the nonlinear-probe bias depends on. Combine with isotropy: minimizing $\mathrm{tr}(\Sigma^{-1})=\sum_i 1/\lambda_i$ under any scalar covariance constraint forces $\Sigma=sI$. The two axioms drop out together — the embeddings should be distributed as an **isotropic Gaussian** $\mathcal N(0,I)$. That target subsumes both prior rungs (isotropic covariance is decorrelated and equal-variance) *and* controls every higher moment, the part second-moment methods leave free.

The hard part is pushing a batch toward $\mathcal N(0,I)$ cheaply. A divergence (KL, MMD) means high-dimensional density estimation, exactly what I am trying to avoid; reframe it as a hypothesis test — is there evidence the embedding distribution differs from $\mathcal N(0,I)$? — pick a scalar test statistic of departure and *minimize* it. Off-the-shelf multivariate normality tests (Baringhaus–Henze, Henze–Zirkler) are $O(B^2)$ double sums over all sample pairs, the batch-coupled cost that hobbled contrastive learning. The escape is **slicing**: by Cramér–Wold, two random vectors are equal in distribution iff all their one-dimensional projections are, and for $\mathcal N(0,I)$ every unit projection $a^\top z$ is exactly $\mathcal N(0,1)$. So a $D$-dimensional Gaussianity test becomes a family of univariate "are these projected scalars standard normal?" tests, one per random direction, each on a scalar batch. For a *loss* I aggregate by *averaging* over directions, not taking the max — the max routes gradient only through the single worst direction (sparse, jumpy), while the average routes gradient through all sampled directions at once. The per-slice statistic is **Epps–Pulley** (characteristic-function): the empirical CF $\hat\varphi(t)=\mathrm{mean}_b\exp(itx_b)$ is a differentiable average, identifiable (full Fourier content, not a moment truncation), and in its Gaussian-weighted $L^2$ distance to the target CF has gradient and curvature bounded regardless of the input — unlike moment tests (non-identifiable / explode at high order) or CDF tests (Cramér–von Mises, Anderson–Darling, KS, all needing non-differentiable sorting). Project onto a direction, compare the empirical CF to $\exp(-t^2/2)$ in a Gaussian-weighted $L^2$ norm, integrate over a frequency grid, average over directions, add to the invariance term.

What this harness runs is the **BCS** (Batched Characteristic Slicing) form, a deliberately stripped version whose simplifications are load-bearing for both cost and numbers. I hold `num_slices = 256` random directions, drawn fresh each call from a `torch.Generator` seeded by an internal `step` counter so directions advance over training and the sphere accumulates coverage, normalized to unit norm, and project each view onto them. Per projection the Epps–Pulley statistic is computed directly: a small frequency grid `t = linspace(−3, 3, 10)`, the complex empirical CF `ecf = (1j * x_t).exp().mean(0)`, the Gaussian window $\exp(-t^2/2)$ serving as *both* target CF and weight, the weighted squared error $\exp(-t^2/2)\cdot|\mathrm{ecf}-\exp(-t^2/2)|^2$, and `torch.trapz` over $t$. The BCS loss is the mean over the two views' slices; the invariance term is `F.mse_loss(z1, z2)`; total is $\text{invariance} + \lambda\cdot\text{bcs}$ with `lmbd = 10.0`. Three things are *simpler* than the full machinery and worth naming: it integrates the full symmetric grid $[-3,3]$ with `torch.trapz` rather than exploiting evenness to halve it; it keeps complex arithmetic (`1j`) rather than the cos/sin-mean real form; and it seeds directions from a plain per-module `step` rather than a cross-device-synchronized counter with an `all_reduce` (single device here, so no sync needed) — and only 10 frequency knots, not 17. None change the algorithm; they trade a little quadrature resolution and DDP-readiness for a compact single-device module. The one configuration choice that *does* matter is `CONFIG_OVERRIDES = {"proj_output_dim": 128}`: the Gaussianity-on-random-projections test concentrates better at low output dim — fewer directions cover a 128-sphere than a 2048-sphere, so 256 slices pin the whole distribution far more tightly at $D=128$ — and the upstream recipe ranks $2048\to128$ top. So unlike Barlow (2048) and VICReg (1024), SIGReg narrows hardest, to 128, because its anti-collapse term *wants* a low-dimensional space to certify Gaussian.

The delta from VICReg is precise: where VICReg pinned only the embedding's second moments per branch — variance floor plus covariance decorrelation, leaving every higher moment free and wobbling per backbone — SIGReg pins the *whole* per-view embedding distribution to the derived $\mathcal N(0,I)$ target by slicing it into 256 one-dimensional Epps–Pulley Gaussianity tests in a deliberately narrow 128-dim space, averaged for dense gradients, added to plain-MSE invariance. Because the target is *derived* from the linear- and nonlinear-probe worst case rather than guessed, and because pinning the full distribution controls the higher moments the second-moment methods left free, I expect SIGReg to clear both prior rungs on aggregate — low-90s on every backbone — and, more tellingly, to be the most *stable across backbones*, with the spread tightening (no ResNet-34-style dip), beating VICReg's ResNet-34 (89.5) by the largest margin precisely where second-moment pinning was weakest. The risk is that the stripped 10-knot full-grid quadrature in 128 dimensions is too coarse to certify Gaussianity tightly — if so it would land merely *level* with VICReg rather than above, indicting the quadrature resolution and not the derived target.

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
