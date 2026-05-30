# The slow SDE for adaptive gradient methods, and Adam's unique sharpness reduction

## What it is

A continuous-time characterization of what Adam — and a whole family of adaptive
gradient methods (AGMs) — does *after* it has reached a manifold of minimizers and
the training loss is already tiny. In that regime the gradient has essentially
vanished, so the optimizer no longer "descends"; instead it wanders along the
manifold of equally-good solutions, and a slow, noise-driven drift selects *which*
minimizer it ends up at. The result is a **slow SDE** that tracks this wandering for
a full `O(η⁻²)` steps and reveals that Adam performs **adaptive semi-gradient descent
on a sharpness measure**, biasing toward flatter regions in a way that differs from
SGD. Under label noise the slow SDE collapses to an ODE whose fixed points minimize

```
SGD :    tr(H)                 (≈ ℓ1 on the spectrum / ground truth)
Adam:    tr(Diag(H)^{1/2})     (≈ ℓ0.5 — sparser)
AdamE-λ: tr(Diag(H)^{1-λ})     (interpolates: λ=0 ⇒ SGD, λ=1/2 ⇒ Adam)
```

where `H = ∇²L`. The `^{1/2}` (instead of SGD's linear `tr H`) makes Adam's bias
align with sparsity: in sparse linear regression with a diagonal network it predicts
sample-efficient sparse recovery; in deep matrix factorization, where the useful bias
is toward low rank (roughly nuclear norm, tracked by `tr H`), the same
`tr(Diag(H)^{1/2})` target predicts worse recovery than SGD.

## The general AGM framework

```
m_{k+1} = β1·m_k + (1-β1)·g_k
v_{k+1} = β2·v_k + (1-β2)·V(g_k g_kᵀ)
θ_{k+1} = θ_k - η·S(v_{k+1})·m_{k+1}
```

with `V : ℝ^{d×d} → ℝ^D` linear and mapping outer products to nonnegative vectors,
and `S : ℝ^D_{≥0} → 𝕊^d_{++}` smooth with `S(v) ⪰ I/R0`. Instances:

| optimizer | V(M) | S(v) |
|---|---|---|
| SGD | 1 | I |
| Adam / RMSProp | diag(M) | Diag(1/(√v+ε)) |
| AdamE-λ | diag(M) | Diag(1/(v^λ+ε)) |
| Adam-mini / Adalayer | block/layer-averaged diag | Diag(1/(√v+ε)) |
| Shampoo | (V_L, V_R) Kronecker factors | ((V_R+εI)ᵀ⊗(V_L+εI))^{-1/2} |

## The slow SDE

Let `Φ_S(x)` be the limit of the **preconditioned** gradient flow `ẋ = -S∇L(x)`;
at `ζ∈Γ`, `∂Φ_S(ζ)` is identity on tangent directions and kills `S` times normal
directions. With
`c = (1-β2)/η²`, `S_t = S(v(t))`:

```
dζ = P_{ζ,S_t}( Σ_∥^{1/2}(ζ;S_t) dW_t  −  ½ S_t ∇³L(ζ)[Σ_◇(ζ;S_t)] dt )
dv = c ( V(Σ(ζ)) − v ) dt

Σ_◇(ζ;S) = S Σ(ζ) S − Σ_∥(ζ;S),     Σ_∥(ζ;S) = ∂Φ_S(ζ) S Σ(ζ) S ∂Φ_S(ζ)
```

The drift is the negative **semi-gradient** of `μ(ζ,v)=⟨∇²L(ζ), Σ_◇(ζ;S)⟩`
(gradient w.r.t. the first argument only), preconditioned by `S_t`. The second line
is an OU-like equation: the preconditioner state `v` relaxes toward `V(Σ(ζ))` on the
**same** `O(η⁻²)` timescale, which is why `1-β2` must scale as `Θ(η²)` (the
"2-scheme") — fast enough to inject adaptiveness, slow enough to be trackable.

**Approximation theorem.** Under `C⁵` smoothness of `L` and `Σ^{1/2}`, `C⁴` of `S`,
a compact `C∞` minimizer manifold, bounded noisy gradients, `β1 ≤ 0.9`, and
`1-β2 = Θ(η²)`: after `K0 = O((1/η)log(1/η))` convergence steps, for every `C³` test
function `g` and `K = ⌊T η⁻²⌋`,
`max_{k≤K} |E[g(X̄_k)] − E[g(X(kη²))]| = Õ(η^{0.25})`, where `X̄_k` is the AGM state
projected onto `Γ`. (Established via a high-probability convergence bound to a tube
around `Γ`, a giant-step moment calculation, and a weak-approximation argument.)

## Working code

The code below is grounded
in the standard PyTorch Adam update generalized to the `(V,S)` AGM form, and in the
diagonal-network sparse-regression-with-label-noise setup exactly as specified.

### The AGM optimizer (Adam / RMSProp / AdamE-λ)

```python
import torch

class AGM(torch.optim.Optimizer):
    """General Adaptive Gradient Method:
        m <- b1 m + (1-b1) g
        v <- b2 v + (1-b2) V(g gᵀ)
        θ <- θ - η S(v) m
    Diagonal V/S instances (Adam, RMSProp, AdamE-λ) need no matrix algebra:
    V(g gᵀ) = g⊙²  and  S(v) m = m / (v^λ + ε)   (λ=1/2 is Adam).
    """
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, lam=0.5):
        super().__init__(params, dict(lr=lr, b1=b1, b2=b2, eps=eps, lam=lam))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2, eps, lam, lr = grp["b1"], grp["b2"], grp["eps"], grp["lam"], grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)          # first moment
                v.mul_(b2).addcmul_(g, g, value=1 - b2)   # V(g gᵀ)=g⊙²  (second moment)
                S_diag = v.pow(lam).add_(eps)             # S(v)=Diag(1/(v^λ+ε))
                p.addcdiv_(m, S_diag, value=-lr)          # θ -= η S(v) m
```

Bias correction and the `2-scheme` are matters of hyperparameter choice; the analysis
omits bias correction (negligible for large `k`) and takes `1-β2 = Θ(η²)`.

### Diagonal-net sparse regression with label noise (prediction target)

```python
import torch

def run_diagnet(opt_name, n_train, d=10000, kappa=50, delta=0.1, steps=20000,
                lr=1e-2, lam=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    # κ-sparse ground truth
    w_star = torch.zeros(d); idx = torch.randperm(d, generator=g)[:kappa]
    w_star[idx] = torch.randn(kappa, generator=g)
    # data z ∈ {±1}^d, clean labels y = <z, w*>
    Z = (torch.randint(0, 2, (n_train, d), generator=g) * 2 - 1).float()
    y = Z @ w_star
    Ztest = (torch.randint(0, 2, (2000, d), generator=g) * 2 - 1).float()
    ytest = Ztest @ w_star

    # diagonal net parameterization  ŵ = u⊙² - v⊙²
    u = torch.full((d,), 0.1, requires_grad=True)
    v = torch.full((d,), 0.1, requires_grad=True)
    if opt_name == "sgd":
        opt = torch.optim.SGD([u, v], lr=lr)
    else:  # "adam" (lam=0.5) or "adame" (general lam)
        opt = AGM([u, v], lr=lr, lam=lam)

    for _ in range(steps):
        i = torch.randint(0, n_train, (1,), generator=g).item()
        noisy = y[i] + delta * (2 * torch.randint(0, 2, (1,), generator=g).item() - 1)
        pred = (Z[i] * (u**2 - v**2)).sum()             # <z_i, ŵ>
        loss = 0.5 * (pred - noisy) ** 2                # fresh label noise each step
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        w_hat = u**2 - v**2
        test_loss = 0.5 * ((Ztest @ w_hat - ytest) ** 2).mean()
    return test_loss.item()                              # "recovered" if < 1
```

The narrative: SGD's slow drift minimizes `tr H ∝ Σ(u_i²+v_i²)`, i.e. `‖ŵ‖₁`;
Adam's minimizes `tr(Diag(H)^{1/2}) ∝ Σ(|u_i|+|v_i|)`, i.e. the `ℓ_{0.5}`
quasi-norm up to a monotone power. Because
`u_i² - v_i²` is what the loss sees, the optimum forces `u_i=0 ∨ v_i=0`, so these
become genuine `ℓ₁` vs `ℓ_{0.5}` penalties on the recovered vector. The sparser
`ℓ_{0.5}` target predicts that Adam, and AdamE with `λ>0`, should need fewer samples
than SGD in this diagnostic setting.
