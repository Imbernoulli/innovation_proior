Training a deep network with stochastic gradient descent and momentum is slow, and the slowness is not in the cost of a step — each step is just a gradient — but in the quality of the *direction*. The loss surfaces of deep networks are wildly anisotropic: some directions have enormous curvature, others almost none, and the high- and low-curvature directions do not line up with the coordinate axes, so the curvature is densely correlated across parameters. Plain gradient descent rescales nothing and rotates nothing, so it crawls along low-curvature valleys and oscillates across high-curvature walls. The textbook cure is to divide the gradient by the curvature. For a network that outputs the parameters $z=f(x,\theta)$ of a predictive distribution $R_{y|z}$ with loss $L(y,z)=-\log r(y|z)$, training is maximum likelihood and the model defines a conditional distribution $P_{y|x}(\theta)$; the canonical notion of "how far did I move" is then not Euclidean distance in $\theta$ but the KL divergence between predictive distributions, and steepest descent under that metric is the natural gradient $F^{-1}\nabla h$, with $F=E[\nabla_\theta\log p\,\nabla_\theta\log p^\top]=E[\nabla\theta\,\nabla\theta^\top]$ the Fisher information. The natural gradient is invariant to reparameterization, which is exactly the property a curvature-aware optimizer should have, and when the loss is the negative log-likelihood of an exponential family with $z$ the natural parameters the Fisher coincides with the Generalized Gauss-Newton matrix, a positive-semidefinite stand-in for the Hessian — so $F$ is genuinely a curvature matrix and the whole quadratic-model toolbox (trust regions, damping, line searches) applies. The wall is size: $F$ is $(\#\mathrm{params})^2$, millions squared, so it can be neither stored nor inverted.

The existing options each fail on a different axis. Diagonal, per-unit, or low-rank curvature approximations are cheap and can be averaged online across many minibatches, but they discard exactly the off-diagonal, cross-parameter curvature that gives second-order methods their power — a diagonal preconditioner can only rescale axes, never rotate — so in practice they barely beat momentum. The Hessian-free / truncated-Newton approach keeps the full richness by never forming $F$ and instead solving $F\delta=-\nabla h$ with inner conjugate gradient using exact curvature-matrix-vector products; but each update runs CG for hundreds of mat-vecs (each as costly as a gradient), the curvature must be *frozen* across the CG iterations so each update sees only one small fixed minibatch — crippling in the stochastic regime — and the exact $F$ has no compact summary, so it cannot accumulate curvature over a long data history. What is wanted is all three at once: an approximation of $F$ that is rich (non-diagonal, non-low-rank), directly invertible in closed form (no inner solver), and summarizable in a small data-amount-independent structure (so it can be estimated by an exponential moving average over many minibatches). No prior method had all three.

I propose K-FAC, Kronecker-Factored Approximate Curvature, and the structural fact that makes it possible is the shape of a layer's gradient. With $s_i=W_i\bar a_{i-1}$, $a_i=\phi_i(s_i)$, and the bias folded in by appending a constant $1$ so $\bar a=[a;1]$ and the bias is the last column of $W$, backpropagation gives each layer's weight gradient as a single outer product $\nabla_{W_i}L=g_i\,\bar a_{i-1}^\top$, where $g_i=Ds_i$ is the backpropagated pre-activation gradient. Stacking parameters layer by layer makes $F=E[\nabla\theta\,\nabla\theta^\top]$ an $\ell\times\ell$ block matrix whose $(i,j)$ block is $E[\mathrm{vec}(\nabla W_i)\,\mathrm{vec}(\nabla W_j)^\top]$. Using the column-stacking identity $\mathrm{vec}(uv^\top)=v\otimes u$ gives $\mathrm{vec}(\nabla W_i)=\bar a_{i-1}\otimes g_i$, and then the Kronecker transpose and mixed-product rules collapse the block into

$$F_{i,j}=E\!\left[(\bar a_{i-1}\bar a_{j-1}^\top)\otimes(g_i g_j^\top)\right].$$

Each block is the *expectation* of a Kronecker product of an activation-covariance factor and a gradient-covariance factor. That expectation entangles the two factors, so it can be neither inverted nor stored compactly — unless I make the central approximation: move the expectation inside each factor,

$$F_{i,j}\approx E[\bar a_{i-1}\bar a_{j-1}^\top]\otimes E[g_i g_j^\top]=:\bar A_{i-1,j-1}\otimes G_{i,j}.$$

Now each block is a *single* Kronecker product of two small matrices — one the size of a layer's input, one the size of its output — which I can store and invert. This is a real approximation, not an identity, since $E[XY]\neq E[X]E[Y]$ in general, and it is worth knowing exactly what is assumed. Writing the exact fourth moment $E[\bar a^{(1)}\bar a^{(2)}g^{(1)}g^{(2)}]$ in cumulants and using the lemma that forward-pass quantities are uncorrelated with backward derivatives *under the model's own output distribution* — if $u$ is independent of $y$ given $f(x,\theta)$ and $v$ is a forward quantity then $E[u\,Dv]=0$, because the expected score $E_{y\sim\text{model}}[\partial\log r/\partial z]$ vanishes — kills the first cumulants $\kappa(g)=0$ and the cross cumulants $\kappa(\bar a,g)=0$. Of the fifteen partition terms, ten die, and the surviving second-and-lower-order pieces reassemble into exactly $E[\bar a^{(1)}\bar a^{(2)}]\,E[g^{(1)}g^{(2)}]$, so the error of the factoring is precisely $\kappa(\bar a^{1},\bar a^{2},g^{1},g^{2})+E[\bar a^{1}]\kappa(\bar a^{2},g^{1},g^{2})+E[\bar a^{2}]\kappa(\bar a^{1},g^{1},g^{2})$ — a fourth-order cumulant plus two third-order ones, all of which vanish for jointly Gaussian $(\bar a,g)$. The approximation is therefore exact insofar as $(\bar a,g)$ are jointly Gaussian, with the error governed by their higher cumulants. This lemma is also why targets must be sampled from the model, not taken from the training labels: with training labels the score's inner expectation is nonzero, the lemma collapses, one gets the empirical Fisher instead, and the Gauss-Newton equivalence is lost.

With each block a single Kronecker product, inversion is $(\bar A\otimes G)^{-1}=\bar A^{-1}\otimes G^{-1}$, and applying the inverse to a layer's gradient reshaped as a matrix $V_i$ uses $(A\otimes B)\mathrm{vec}(X)=\mathrm{vec}(BXA^\top)$, so the per-layer natural-gradient update is

$$U_i=G_{i,i}^{-1}\,V_i\,\bar A_{i-1,i-1}^{-1}$$

— left-multiply by $G^{-1}$ on the output side, right-multiply by $\bar A^{-1}$ on the input side ($\bar A,G$ are symmetric, so no transpose), no giant matrix ever formed, no CG. That handles within-layer structure, but the full approximate Fisher is a block matrix of Kronecker products (a Khatri-Rao product) and that structure does not survive inversion across layers. So I simplify the *inverse's* across-layer structure rather than the Fisher's. The crude choice is to make $F^{-1}$ block-diagonal, inverting each $\bar A_{i-1,i-1}\otimes G_{i,i}$ independently. This is justified by the precision-matrix picture: for a covariance $\Sigma$, the inverse $\Sigma^{-1}=D^{-1}(I-B)$ has small entries wherever a variable is not a useful linear predictor of another, and $F$ is a covariance ($E[\nabla\theta]=0$ by the same lemma). The most useful predictors of an entry of $\nabla W_i$ are the other entries of $\nabla W_i$ (shared forward and backward signals), so the diagonal blocks of $F^{-1}$ dominate — a statement about the *inverse* even though $F$ itself is dense. The next-most-useful predictors are the adjacent layers $\nabla W_{i\pm 1}$, since $\nabla W_i$ draws only on the layer below (forward) and above (backward), which motivates the richer block-*tridiagonal* inverse $\hat F^{-1}$: model $\nabla\theta$ as a linear-Gaussian chain $\mathrm{vec}(\nabla W_i)\sim N(\Psi_{i,i+1}\mathrm{vec}(\nabla W_{i+1}),\Sigma_{i|i+1})$, read the regression coefficients $\Psi_{i,i+1}=(\bar A_{i-1,i}\bar A_{i,i}^{-1})\otimes(G_{i,i+1}G_{i+1,i+1}^{-1})$ and conditional covariances $\Sigma_{i|i+1}$ off the tridiagonal blocks, and use the block-Cholesky factorization $\hat F^{-1}=\Xi^\top\Lambda\,\Xi$ — every $\Psi$ a Kronecker product so the $\Xi,\Xi^\top$ products stay cheap, with only the differences-of-Kronecker-products $\Sigma_{i|i+1}$ needing a Stein-equation solver $(A\otimes B\pm C\otimes D)^{-1}$ built from eigendecompositions of the whitened factors.

The approximate Fisher gives a *direction*; turning it into a real optimizer that can take a *step* needs damping, because any large step from a local quadratic model must guard against the model's breakdown — and a powerful second-order method needs *better* damping, not worse, like a fast car needing a better control system. Adding $(\lambda+\eta)I$ to a block (with $\eta$ the $\ell_2$ coefficient) breaks the single-Kronecker structure, since it becomes a sum of two Kronecker products. The fix is to damp each *factor* instead of the product,

$$\left(\bar A+\pi\sqrt{\lambda+\eta}\,I\right)\otimes\left(G+\tfrac{1}{\pi}\sqrt{\lambda+\eta}\,I\right),$$

which stays a single Kronecker product so every inversion formula still applies verbatim. Expanding it reproduces exact Tikhonov's $\bar A\otimes G+(\lambda+\eta)I\otimes I$ plus a residual $\pi\sqrt{\lambda+\eta}(I\otimes G)+\tfrac1\pi\sqrt{\lambda+\eta}(\bar A\otimes I)$; minimizing a triangle-inequality bound on that residual sets $\pi=\sqrt{\|\bar A\otimes I\|/\|I\otimes G\|}$, which with the trace norm becomes $\pi_i=\sqrt{[\mathrm{tr}(\bar A_{i-1,i-1})/(d_{i-1}+1)]/[\mathrm{tr}(G_{i,i})/d_i]}$, the square root of the ratio of the two factors' average eigenvalues, so neither factor is over- or under-damped. A single Tikhonov $\lambda$ cannot do everything, though: my $\bar A\otimes G$ carries a factoring error on top of the usual quadratic-model error, so the $\lambda$ that keeps the proposal trustworthy is large enough to wash out the small eigenvalues — the low-curvature directions a second-order method exists to exploit. The cure is to split the two jobs. First produce a candidate $\Delta=\tilde F^{-1}(-\nabla h)$ with light factored damping, then *rescale against the exact Fisher*: set $\delta=\alpha^\star\Delta$ with

$$\alpha^\star=-\frac{\nabla h^\top\Delta}{\Delta^\top F\Delta+(\lambda+\eta)\|\Delta\|^2},$$

which needs only one exact-$F$ matrix-vector product on the current minibatch, used to compute a couple of scalars, so a noisy estimate is fine (and $\Delta^\top F\Delta=(J\Delta)^\top F_R(J\Delta)$ halves the cost). With that scaling I can keep $\lambda$ small while a separate constant $\gamma$ (initialized to $\sqrt{\lambda+\eta}$) controls the factored damping of $\tilde F$: $\lambda$ is adapted by Levenberg-Marquardt from the reduction ratio $\rho=(h(\theta+\delta)-h(\theta))/(M(\delta)-M(0))$ — shrink it when $\rho>3/4$, grow it when $\rho<1/4$, cheaply since $M(\delta)-M(0)=\tfrac12\nabla h^\top\delta$ at the optimum — while $\gamma$ is adapted by a greedy three-point search $\{\gamma,\omega_2\gamma,\gamma/\omega_2\}$ on the already-computed rescaled $M(\delta)$. Momentum folds in parameter-free: instead of $\delta=\alpha\Delta$, search the 2-D subspace of the proposal $\Delta$ and the previous update $\delta_0$, solving a $2\times2$ minimization of the exact-$F$ quadratic for $(\alpha,\mu)$ in $\delta=\alpha\Delta+\mu\delta_0$ — literally preconditioned conjugate gradient in the deterministic-quadratic limit, with no momentum hyperparameter to tune. Finally, because $\bar A$ and $G$ are small fixed-size matrices, they are kept as exponential moving averages across minibatches, $\text{new}=\varepsilon\cdot\text{old}+(1-\varepsilon)\cdot\text{batch}$ with $\varepsilon=\min\{1-1/k,0.95\}$ (exponential, not flat, because the factors drift as $\theta$ moves and old estimates go stale), so the curvature draws on far more data than one minibatch while staying compact — exactly the requirement Hessian-free could never meet. As a bonus the construction is invariant: under a layerwise linear reparameterization the factors transform as $G_{i,j}^\dagger=\Phi_i^\top G_{i,j}\Phi_j$ and $\bar A_{i,j}^\dagger=\Omega_i\bar A_{i,j}\Omega_j^\top$, giving $\breve F^\dagger=J_\zeta^\top\breve F J_\zeta$, so K-FAC's path through distribution space is invariant to input rescaling and to sigmoid-vs-tanh; choosing the whitening transform $\Phi_i=G_{i,i}^{-1/2},\Omega_i=\bar A_{i,i}^{-1/2}$ makes $\breve F^\dagger=I$, showing block-diagonal K-FAC is ordinary gradient descent on a network whose activations and backpropagated gradients have been centered and whitened — what centering methods reached for, now achieved with the within-layer correlations accounted for and without skip connections.

```python
import math
import torch
import torch.optim as optim

KNOWN = {"Linear", "Conv2d"}  # layers whose gradient is an outer product g·āᵀ


def cov_a(a, layer):
    # A = E[ā āᵀ]; append a constant 1 so the bias is the last column of W
    b = a.size(0)
    if layer.bias is not None:
        a = torch.cat([a, a.new_ones(b, 1)], dim=1)
    return a.t() @ (a / b)


def cov_g(g, layer, batch_averaged):
    # G = E[g gᵀ]; g is the backprop pre-activation gradient (targets sampled from the model)
    b = g.size(0)
    return g.t() @ (g * b) if batch_averaged else g.t() @ (g / b)


def update_running(stat, store, decay):  # store ← decay·store + (1-decay)·stat
    store.mul_(decay / (1 - decay)).add_(stat).mul_(1 - decay)


class KFAC(optim.Optimizer):
    def __init__(self, model, lr=1e-3, momentum=0.9, stat_decay=0.95,
                 damping=1e-3, kl_clip=1e-3, weight_decay=0, t_cov=10, t_inv=100):
        super().__init__(model.parameters(),
                         dict(lr=lr, momentum=momentum, damping=damping,
                              weight_decay=weight_decay))
        self.stat_decay, self.kl_clip = stat_decay, kl_clip
        self.t_cov, self.t_inv, self.steps = t_cov, t_inv, 0
        self.A, self.G = {}, {}                 # running E[āāᵀ], E[ggᵀ]
        self.Qa, self.Qg, self.da, self.dg = {}, {}, {}, {}
        self.layers = []
        for m in model.modules():
            if m.__class__.__name__ in KNOWN:
                self.layers.append(m)
                m.register_forward_pre_hook(self._hook_fwd)
                m.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, m, inp):
        if torch.is_grad_enabled() and self.steps % self.t_cov == 0:
            a = cov_a(inp[0].data, m)
            if self.steps == 0:
                self.A[m] = torch.diag(a.new_ones(a.size(0)))
            update_running(a, self.A[m], self.stat_decay)

    def _hook_bwd(self, m, gin, gout):
        if self.steps % self.t_cov == 0:
            g = cov_g(gout[0].data, m, batch_averaged=True)
            if self.steps == 0:
                self.G[m] = torch.diag(g.new_ones(g.size(0)))
            update_running(g, self.G[m], self.stat_decay)

    def _eig(self, m):                          # A = Qa diag(da) Qaᵀ, G = Qg diag(dg) Qgᵀ
        self.da[m], self.Qa[m] = torch.linalg.eigh(self.A[m])
        self.dg[m], self.Qg[m] = torch.linalg.eigh(self.G[m])

    def _grad_mat(self, m):                     # gradient as an output×input matrix
        gm = m.weight.grad.data
        if m.__class__.__name__ == "Conv2d":
            gm = gm.view(gm.size(0), -1)
        if m.bias is not None:
            gm = torch.cat([gm, m.bias.grad.data.view(-1, 1)], dim=1)
        return gm

    def _natural_grad(self, m, gm, damping):
        # G⁻¹ (grad) Ā⁻¹ in eigenbasis; factored damping added to the eigenvalue products
        v1 = self.Qg[m].t() @ gm @ self.Qa[m]
        v2 = v1 / (self.dg[m].unsqueeze(1) * self.da[m].unsqueeze(0) + damping)
        v = self.Qg[m] @ v2 @ self.Qa[m].t()
        if m.bias is not None:
            return [v[:, :-1].view_as(m.weight.grad), v[:, -1:].view_as(m.bias.grad)]
        return [v.view_as(m.weight.grad)]

    def step(self, closure=None):
        grp = self.param_groups[0]
        lr, damping = grp["lr"], grp["damping"]
        updates = {}
        for m in self.layers:
            if self.steps % self.t_inv == 0:
                self._eig(m)
            updates[m] = self._natural_grad(m, self._grad_mat(m), damping)

        # rescale the whole proposal (cheap stand-in for the exact-F quadratic rescaling)
        vg = 0.0
        for m in self.layers:
            v = updates[m]
            vg += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / (vg + 1e-12)))

        wd, mom = grp["weight_decay"], grp["momentum"]
        for m in self.layers:
            v = updates[m]
            m.weight.grad.data.copy_(v[0]).mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1]).mul_(nu)
        for p in grp["params"]:
            if p.grad is None:
                continue
            d = p.grad.data
            if wd != 0:
                d = d.add(p.data, alpha=wd)
            if mom != 0:
                buf = self.state[p].setdefault("mom", torch.zeros_like(p.data))
                buf.mul_(mom).add_(d)
                d = buf
            p.data.add_(d, alpha=-lr)
        self.steps += 1
```
