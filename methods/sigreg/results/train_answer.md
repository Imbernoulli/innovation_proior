I train an encoder $f_\theta$ from unlabeled data by joint embedding: take two related views of an input, embed both, and push the encoder so one view's embedding predicts the other's. The trouble is that this predictability objective, alone, has a trivial global minimum. If $f_\theta$ sends every input to the same constant vector, every view predicts every other perfectly, the loss is zero, and the representation carries nothing — complete collapse; the softer failure squashes all embeddings into a low-dimensional subspace, dimensional collapse, which is documented to track the covariance spectrum going rank-deficient. So a second term is needed whose only job is to keep the embeddings from degenerating, and the way the field builds that term today is a pile of heuristics: feature-whitening layers, batches of negatives, momentum teacher networks feeding students with a stop-gradient, centering and sharpening, asymmetric augmentations. Each patches a symptom and shares the same defects. They are under-specified — whitening's identity-covariance criterion, or VICReg's variance and covariance terms, or Barlow Twins' cross-correlation-to-identity, can all be driven to their minimum while the embeddings still sit in a degenerate configuration, because those criteria pin down only the first two moments and leave the rest of the distribution free, which is exactly where shortcut solutions hide. Several are quadratic in batch size or embedding dimension — anything comparing all pairs of samples, or forming and inverting a covariance, costs $O(N^2)$ or $O(K^3)$. They are brittle: EMA decay coupled to learning rate, temperatures and projector widths to retune, a recipe tuned in one setting collapsing in another. And there is no theory saying what the embeddings should look like — the mutual-information bounds were written after the fact to explain methods that already existed. I want to stop patching symptoms and answer two questions in order: what distribution *should* the embeddings have, and how do I enforce it cheaply and stably.

I propose SIGReg — Sketched Isotropic Gaussian Regularization. Its first half is a derivation of the target distribution rather than an assumption of one. Because downstream tasks are unknown at pretraining, I want the embedding geometry that is best in the worst case over unseen tasks. Take the simplest probe, ridge regression on frozen embeddings $Z$: $\hat\beta = (Z^\top Z + \lambda I)^{-1} Z^\top y$. With labels $y = Z\beta_{\text{true}} + \varepsilon$, the ridge bias collapses to $-\lambda (Z^\top Z + \lambda I)^{-1}\beta_{\text{true}}$, and along the eigendirection of the *smallest* Gram eigenvalue $\lambda_p$ the bias norm is $\frac{\lambda}{\lambda_p + \lambda}\|\beta_{\text{true}}\|$ — strictly larger than the isotropic $\frac{\lambda}{\bar\lambda + \lambda}\|\beta_{\text{true}}\|$ whenever the spectrum is not flat, so an adversarial task aligned with the weak direction always exists. The variance says the same: with $\lambda = 0$ the total is $\sigma^2 \sum_j 1/\lambda_j$, and since $1/x$ is strictly convex, Jensen makes this minimal at fixed mean eigenvalue exactly when all $\lambda_j$ are equal. The linear probe forces isotropy.

Isotropy fixes only the covariance shape, so I push to a nonlinear probe to pin the rest. For a radius-$k$-NN or kernel (Nadaraya–Watson) estimator, the small-bandwidth expansion gives a pointwise bias whose only distribution-dependent piece, after integrating the squared bias over queries $x\sim p$ under a neutral isotropic prior on the unknown task gradient ($\mathbb{E}[\nabla\eta]=0$, $\mathbb{E}[\nabla\eta\nabla\eta^\top]=\tau_g^2 I$), reduces through $\mathbb{E}[(\nabla\eta^\top v)^2] = \tau_g^2\|v\|^2$ with $v=\nabla\log p$ to $\tau_g^2 \int \|\nabla\log p(x)\|^2 p(x)\,dx$ — exactly the Fisher-information functional $J(p)$. So the controllable part of the downstream bias is $\tau_g^2 J(p)$, and I should minimize Fisher information at fixed covariance. The density that does so is the Gaussian, and the reason is the Cramér–Rao bound, not a hand-wave at maximum entropy. Fix covariance $\Sigma$ and the location family $p_\theta(x)=p(x-\theta)$; the identity estimator $T(X)=X$ is unbiased for the location with covariance $\Sigma$, so the matrix Cramér–Rao bound $\mathrm{Cov}(T)\succeq I(\theta)^{-1}$ reads $\Sigma\succeq I^{-1}$, hence $I\succeq\Sigma^{-1}$, and tracing gives
$$J(p) = \mathrm{tr}\, I \ge \mathrm{tr}(\Sigma^{-1}).$$
Equality holds iff the score is affine in $X-\theta$, i.e. the log-density is quadratic, i.e. $p$ is Gaussian. So among all densities with covariance $\Sigma$, $\mathcal{N}(0,\Sigma)$ uniquely minimizes Fisher information. Then minimizing $\mathrm{tr}(\Sigma^{-1})=\sum_i 1/\lambda_i$ under any scalar covariance constraint forces all eigenvalues equal — Cauchy–Schwarz under a trace constraint, AM–GM under a determinant constraint, the Lagrange conditions $\lambda_i^3 = \text{const}$ under a Frobenius constraint, monotonicity of $1/x$ under a spectral cap — so $\Sigma = sI$. The target is the isotropic Gaussian, WLOG $\mathcal{N}(0, I)$.

Now I must push the batch toward $\mathcal{N}(0,I)$ cheaply. The naive move — a KL or MMD divergence to the target — is exactly the high-dimensional density estimation I am avoiding. So I reframe: rather than measure a divergence, I ask whether there is *evidence* that the embedding law $P_\theta$ differs from $Q=\mathcal{N}(0,I)$. That is a hypothesis test $H_0: P_\theta = Q$; I pick a scalar statistic measuring evidence against the null and *minimize* it as the regularizer. The off-the-shelf multivariate normality tests (Baringhaus–Henze, Henze–Zirkler) are double sums over all sample pairs, $O(N^2)$, and couple every sample so they will not shard across data-parallel devices. The escape is slicing. The hyperspherical Cramér–Wold theorem says two vectors are equal in law iff all unit-direction projections are, and restricting to *unit* directions loses nothing: writing any $t = s\,u$ with $u$ on the sphere, $\varphi_X(t) = \mathbb{E}[e^{is\langle u,X\rangle}] = \mathbb{E}[e^{is\langle u,Y\rangle}] = \varphi_Y(t)$ because the scalar projections $\langle u,X\rangle$ and $\langle u,Y\rangle$ are equal in law and therefore share all characteristic-function values, including at $s$; by uniqueness $X=_d Y$. A $K$-dimensional Gaussianity test becomes a family of one-dimensional tests on the scalars $a^\top f_\theta(x)$, and the target is trivial since every unit projection of $\mathcal{N}(0,I)$ is exactly $\mathcal{N}(0,1)$. I sample a finite set $A$ of unit directions and aggregate. For a *test* the right object is $\max_{a\in A} T(a)$ — reject as soon as any direction separates, which gives consistency. But for a *loss* the max is fatal: its gradient flows only through the single arg-max direction, leaving the other hundreds with zero signal, a jumpy and flickering training objective. So I average, $\frac{1}{|A|}\sum_{a\in A} T(a)$, routing gradient through all directions at once.

Which univariate statistic $T$ goes on each slice decides whether the whole thing works, and walking the three classical families shows why only one survives. Moment-based tests (Jarque–Bera, extended to match the first four moments) are cheap but non-identifiable: a finite number of moments does not determine a distribution — the linear map from a $(K+2)$-point probability vector to its first $K$ moments has nontrivial kernel, so distinct non-Gaussian distributions match any finite moment set, which is exactly the moment-correct-but-collapsed shortcut I want to forbid. Sending the order $k\to\infty$ for identifiability detonates stability instead: the moment-objective gradient is a degree-$(k-1)$ polynomial in each sample, $\sim|X_i|^{k-1}$, unbounded for $k\ge 2$, with Monte-Carlo variance growing like $O(k^2 m_{2(k-1)})$. I cannot have both identifiability and bounded gradients with moments. CDF-based tests (Cramér–von Mises, Anderson–Darling, Watson, Kolmogorov–Smirnov, Shapiro–Wilk) are identifiable but built by *sorting*, which is non-differentiable (the empirical CDF's jumps move discontinuously) and breaks data-parallel SGD because order statistics need a global synchronization where an average needs only a cheap all-reduce; Kolmogorov–Smirnov is worse, its sup norm giving the same sparse gradient as the max. That leaves the characteristic function. The empirical CF $\hat\varphi_X(t) = \frac{1}{n}\sum_j e^{itX_j}$ is a plain average of bounded complex exponentials — differentiable, all-reduceable, no sorting — and identifiable, since the CF is the full Fourier content rather than a truncation. The Epps–Pulley statistic compares it to the target in a Gaussian-weighted $L^2$ norm,
$$\mathrm{EP}(a) = N\!\int \big|\hat\varphi_{a^\top Z}(t) - e^{-t^2/2}\big|^2\, w(t)\, dt,$$
with $w(t)=e^{-t^2/\sigma^2}$ down-weighting large $|t|$ so the integral converges on the informative low-frequency band (the canonical implementation reuses the target CF $e^{-t^2/2}$ as the window).

The real reason to choose Epps–Pulley is provable gradient *and* curvature boundedness, regardless of the input distribution — the property moments structurally cannot have. Writing $w_s(t)=e^{-s^2 t^2}$ (so $s^2 = 1/\sigma^2$) and $D = \int w_s |\hat\varphi_N - \varphi_G|^2\,dt$, the only sample dependence is through $\partial\hat\varphi_N/\partial X_i = \frac{1}{N} i t\, e^{itX_i}$, and using only $|\hat\varphi_N|\le 1$, $|\varphi_G|\le 1$, $|e^{itX}|=1$,
$$\Big|\frac{\partial D}{\partial X_i}\Big| \le \frac{4}{N}\int w_s(t)\,|t|\,dt = \frac{4}{N s^2} = \frac{4\sigma^2}{N}, \qquad \Big|\frac{\partial^2 D}{\partial X_i^2}\Big| \le \frac{C}{N}\int w_s(t)\,t^2\,dt = \frac{C\sqrt{\pi}\,\sigma^3}{2N},$$
using $\int e^{-s^2 t^2}|t|\,dt = 1/s^2$ and $\int e^{-s^2 t^2} t^2\,dt = \sqrt{\pi}/(2s^3)$. By the chain rule $\|\nabla_\theta D\| \le \frac{4\sigma^2}{N}\sum_i \|a^\top \nabla_\theta f_\theta(x_i)\|$, which never explodes no matter how degenerate or heavy-tailed the current embeddings are. Two more facts make me trust the sketch in high dimension. A few resampled directions suffice because the projected-CF-as-a-function-of-direction inherits the Sobolev smoothness $\alpha$ of the embedding density, and spherical-harmonic interpolation bounds the expected discrepancy over all directions by $C(K,\alpha)\,|A|^{-2\alpha/(K-1)}$ times a Sobolev norm — large $\alpha$ (smooth deep-network embeddings, infinitely smooth Gaussian target) makes $|A|=O(K)$ enough — and resampling a fresh $A$ each step accumulates coverage linearly in training time, so $|A|=16$ resampled beats a fixed $|A|$ of thousands. The minibatch statistic is biased but explicitly so: $\mathbb{E}[|\hat\varphi_n - \psi|^2] = |\varphi_\theta - \psi|^2 + (1-|\varphi_\theta|^2)/n$ (the ECF double sum's diagonal contributes $1$, the off-diagonal $|\varphi_\theta|^2$), an additive $O(1/N)$ term negligible even at $N=16$. Finally the frequency integral becomes a 17-knot trapezoidal quadrature over $[-3,3]$ or $[-5,5]$ (not $[-1,1]$, which throws away discriminating frequencies); evenness of $|\hat\varphi-\varphi|^2$ for real projections lets me integrate only $[0,t_{\max}]$ with doubled weights and half-weighted endpoints, and no complex arithmetic is needed since $|\hat\varphi - \varphi|^2 = (\text{cos-mean}-\varphi)^2 + (\text{sin-mean})^2$ with both means all-reduced across devices. Directions are Gaussian, normalized to unit columns, and seeded from a step counter synchronized across GPUs (an all-reduce MAX) so every device projects along the same axes. Added to the invariance term $\mathrm{MSE}(z_1,z_2)$ — the predictability half — with a single trade-off coefficient $\lambda$, this is the entire regularizer, with no stop-gradients, teacher–student networks, negatives, or whitening, at $O(N)$ cost.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def all_reduce_mean(x):
    """Global mean of a per-device batch statistic (identity on a single device)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization with the Epps-Pulley test.

    Projects embeddings onto `num_slices` random unit directions (resampled each
    step) and on each projection measures the weighted-L2 distance between the
    empirical characteristic function and exp(-t^2/2), integrated by trapezoid.
    """

    def __init__(self, num_slices=256, lmbd=10.0, n_knots=17, t_max=3.0, sigma=1.0):
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        t = torch.linspace(0.0, t_max, n_knots)            # positive half-grid (use symmetry)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2 * dt)           # doubled for the negative half
        weights[[0, -1]] = dt                              # trapezoid endpoints
        window = torch.exp(-(t ** 2) / (2 * sigma ** 2))   # target CF == Gaussian window
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _epps_pulley(self, proj):
        # proj: [B, M] -- batch projected onto M directions
        B = proj.size(0)
        x_t = proj.unsqueeze(-1) * self.t                  # [B, M, n_knots]
        cos_mean = all_reduce_mean(x_t.cos().mean(0))      # Re phi_hat(t)
        sin_mean = all_reduce_mean(x_t.sin().mean(0))      # Im phi_hat(t)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        return (err @ self.weights) * (B * world)          # [M]

    def _directions(self, z):
        with torch.no_grad():                              # directions are not learned
            step = self.step.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(step, op=torch.distributed.ReduceOp.MAX)
            g = torch.Generator(device=z.device)
            g.manual_seed(int(step.item()))
            A = torch.randn(z.size(1), self.num_slices, device=z.device, generator=g)
            A = A / A.norm(p=2, dim=0)                      # unit-norm columns
        return A

    def forward(self, z):
        with torch.no_grad():
            A = self._directions(z)
            self.step += 1
        return self._epps_pulley(z @ A).mean()             # average over directions

    def pair(self, z1, z2):
        with torch.no_grad():
            A = self._directions(z1)                       # same synced step for both views
            self.step += 1
        return 0.5 * (
            self._epps_pulley(z1 @ A).mean() +
            self._epps_pulley(z2 @ A).mean()
        )


class CustomRegularizer(nn.Module):
    """Two-view anti-collapse regularizer: SIGReg on each view + invariance."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.sigreg = SIGReg(num_slices=num_slices, lmbd=lmbd)
        self.lmbd = lmbd

    def forward(self, z1, z2):
        sigreg = self.sigreg.pair(z1, z2)
        invariance_loss = F.mse_loss(z1, z2)
        return {
            "loss": invariance_loss + self.lmbd * sigreg,
            "sigreg": sigreg,
            "invariance_loss": invariance_loss,
        }
```
