# K-FAC: Kronecker-Factored Approximate Curvature

## Problem

Natural gradient descent updates parameters by `θ ← θ − α F⁻¹∇h`, where `F = E[∇θ ∇θᵀ]` is the
Fisher information matrix (equivalently, the Generalized Gauss-Newton matrix when the loss is the
negative log-likelihood of an exponential-family output with natural parameters). These updates are
far more powerful than plain or diagonally-scaled gradients because they respect the dense,
anisotropic, correlated curvature of a deep network's loss. But `F` is `(#params)²` — impossible to
store or invert directly. K-FAC builds an approximation of `F` that is non-diagonal and non-low-rank
(so its updates stay powerful), directly invertible (no inner conjugate-gradient solve), and
summarizable in a compact, data-amount-independent structure (so the curvature can be estimated by
an online average over many minibatches).

## Key idea

For a layer `s_i = W_i ā_{i-1}` (bias folded into `W` via a homogeneous coordinate `ā = [a;1]`), the
gradient is a single outer product `∇_{W_i}L = g_i ā_{i-1}ᵀ`, with `g_i` the backpropagated
pre-activation gradient. Using `vec(uvᵀ) = v⊗u`, the exact Fisher block is

    F_{i,j} = E[ vec(∇W_i) vec(∇W_j)ᵀ ] = E[ (ā_{i-1}ā_{j-1}ᵀ) ⊗ (g_i g_jᵀ) ].

**The Kronecker approximation:** replace the expectation of a Kronecker product by the Kronecker
product of expectations,

    F_{i,j} ≈ Ā_{i-1,j-1} ⊗ G_{i,j},    Ā_{i,j} = E[ā_iā_jᵀ],   G_{i,j} = E[g_ig_jᵀ].

This is exact when `(ā, g)` are jointly Gaussian; the error is a sum of their third- and
fourth-order cumulants. It relies on taking expectations under the *model's* output distribution
(sample targets `ŷ` from the network, not the training labels), which makes forward-pass quantities
uncorrelated with backward derivatives (`E[u·Dv] = 0`) — using the training labels would instead
give the empirical Fisher and break this.

**Cheap inverse and solve.** Each block is now a single Kronecker product, so
`(Ā⊗G)⁻¹ = Ā⁻¹⊗G⁻¹`, and applying the inverse to a layer gradient `V_i` uses `(A⊗B)vec(X)=vec(BXAᵀ)`:

    U_i = G_{i,i}⁻¹ V_i Ā_{i-1,i-1}⁻¹.

**Across-layer structure.** The full approximate Fisher is a Khatri-Rao product with no efficient
inverse, so approximate its *inverse* as block-diagonal (`F̆`) or block-tridiagonal (`F̂`). Justified
because (i) precision matrices are sparse wherever a variable is not a useful linear predictor of
another, so most mass sits on the diagonal blocks; (ii) information only flows between adjacent
layers, so the tridiagonal extension is a principled mild relaxation. The block-tridiagonal version
treats `∇θ` as a linear-Gaussian chain and uses a block-Cholesky factorization
`F̂⁻¹ = Ξᵀ Λ Ξ`, with `Ψ_{i,i+1} = (Ā_{i-1,i}Ā_{i,i}⁻¹)⊗(G_{i,i+1}G_{i+1,i+1}⁻¹)` and conditional
covariances `Σ_{i|i+1}` (differences of Kronecker products, inverted via a Stein-equation solver).

## Making it a real optimizer

- **Factored Tikhonov damping.** Adding `(λ+η)I` to a block breaks the Kronecker structure, so damp
  each factor: `(Ā + π√(λ+η) I) ⊗ (G + (1/π)√(λ+η) I)`, with
  `π_i = sqrt( [tr(Ā)/(d_{i-1}+1)] / [tr(G)/d_i] )` balancing the damping between the input-side and
  output-side factors (ratio of average eigenvalues).
- **Exact-F rescaling.** The raw proposal `Δ = F̃⁻¹(−∇h)` is poorly scaled; set `δ = α*Δ` with
  `α* = −∇hᵀΔ / (ΔᵀFΔ + (λ+η)‖Δ‖²)` using one exact-Fisher matrix-vector product.
- **Adapt `λ`** by Levenberg-Marquardt from the reduction ratio `ρ = (h(θ+δ)−h(θ))/(M(δ)−M(0))`;
  keep a **separate adaptive `γ`** for the factored damping of `F̃` (greedy 3-point search on `M`).
- **Parameter-free momentum:** `δ = αΔ + μδ₀`, with `(α,μ)` solving a 2×2 minimization of the
  exact-`F` quadratic — equivalent to preconditioned CG in the deterministic-quadratic limit.
- **Online statistics:** keep exponentially-decayed running averages of `Ā, G`
  (`ε = min{1−1/k, 0.95}`); refresh the inverses only every `T₃` iterations.

**Invariance.** With damping negligible, K-FAC's path through distribution space is invariant to
affine input transforms and to sigmoid-vs-tanh, because `F̆† = J_ζᵀ F̆ J_ζ` under the layerwise
reparameterization. Choosing the whitening transform shows block-diagonal K-FAC equals plain
gradient descent on a network whose activations and backpropagated gradients are centered and
whitened with respect to the model's distribution.

## Algorithm (high level)

1. Forward/backward pass on a minibatch → gradient `∇h`.
2. Extra backward pass with targets sampled from the model → update running `Ā_{i,i}`, `G_{i,i}`
   (and `Ā_{i,i+1}`, `G_{i,i+1}` for the tridiagonal version).
3. Every `T₃` iters: form the (factor-damped) approximate inverse from `Ā, G`.
4. Proposal `Δ_i = G_{i,i}⁻¹ (∇_{W_i}h) Ā_{i-1,i-1}⁻¹` (block-diagonal) or via `Ξᵀ Λ Ξ`
   (block-tridiagonal).
5. Final step `δ = αΔ + μδ₀` from the exact-`F` quadratic; update `θ`.
6. Periodically adapt `λ` (Levenberg-Marquardt) and `γ` (greedy search).

## Working code

Per-layer implementation (Linear/Conv2d): forward
and backward hooks accumulate the running factors `A = E[āāᵀ]` and `G = E[ggᵀ]`; periodically each
is eigendecomposed; the natural gradient per layer is `G⁻¹ (grad) Ā⁻¹` evaluated in the eigenbasis
with damping added to the eigenvalue products; the step is rescaled and applied with momentum.

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

Usage: build the model and `opt = KFAC(model)`; each step, run the forward/backward to populate
`.grad`, then call `opt.step()`. For the exact natural gradient one would also run the extra
model-sampled backward pass to populate `G` and apply the exact-`F` rescaling and momentum; the
running averages of `A` and `G` keep the curvature estimate compact and history-spanning, which is
what lets K-FAC be both powerful and cheap.
