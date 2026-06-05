# Adam-Slow-SDE

## What it is

A continuous-time characterization of what Adam — and a whole family of adaptive
gradient methods (AGMs) — does *after* it has reached a manifold of minimizers and
the training loss is already tiny. In that regime the gradient has essentially
vanished, so the optimizer no longer "descends"; instead it wanders along the
manifold of equally-good solutions, and a slow, noise-driven drift selects *which*
minimizer it ends up at. The result is a **slow SDE** that tracks this wandering for
a full `O(η⁻²)` steps and reveals that Adam performs **adaptive semi-gradient descent
on a sharpness measure**, biasing toward flatter regions in a way that differs from
SGD. Under label noise the slow SDE collapses to an ODE whose fixed points are
stationary for

```
SGD :    tr(H)                 (diagonal net: ℓ1)
Adam:    tr(Diag(H)^{1/2})     (diagonal net: ℓ0.5)
AdamE-λ: tr(Diag(H)^{1-λ})     (λ=0 ⇒ SGD, λ=1/2 ⇒ Adam)
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
`c = (1-β2)/η²`, `S_t = S(v(t))`, the moment calculation gives the expanded
form:

```
dζ = S_t ∂Φ_{S_t}(ζ) S_t Σ^{1/2}(ζ) dW_t
     + ½ S_t ∂²Φ_{S_t}(ζ)[S_t Σ(ζ) S_t] dt
dv = c ( V(Σ(ζ)) − v ) dt

Σ_∥(ζ;S) = ∂Φ_S(ζ) S Σ(ζ) S ∂Φ_S(ζ),     Σ_◇(ζ;S) = S Σ(ζ) S − Σ_∥(ζ;S)
```

Using the projection identity, the drift is equivalently the projected negative
**semi-gradient** of `μ(ζ,v)=⟨∇²L(ζ), Σ_◇(ζ;S)⟩` (gradient w.r.t. the first
argument only), preconditioned by `S_t`. The second line
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

The update below is the analyzed one: no bias correction, no weight decay, and
`1-β2` chosen on the `η²` scale when using the slow-SDE regime.

### Optimizer and diagnostic harness

```python
import torch

class CoordinateRescaledOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, exponent=0.5):
        if not 0 <= exponent < 1:
            raise ValueError("exponent must lie in [0, 1)")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        eps=eps, exponent=exponent)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            exponent = group["exponent"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = v.pow(exponent).add(eps)
                p.addcdiv_(m, denom, value=-lr)

        return loss

def make_optimizer(name, params, lr, exponent=0.5):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "rmsprop":
        return CoordinateRescaledOptimizer(params, lr=lr, beta1=0.0, exponent=0.5)
    if name == "adam":
        return CoordinateRescaledOptimizer(params, lr=lr, exponent=0.5)
    if name == "adame":
        return CoordinateRescaledOptimizer(params, lr=lr, exponent=exponent)
    raise ValueError(f"unknown optimizer: {name}")

def make_diagonal_net(d, kappa, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_star = torch.zeros(d)
    idx = torch.randperm(d, generator=g)[:kappa]
    w_star[idx] = torch.randn(kappa, generator=g)
    u = torch.full((d,), 0.1, requires_grad=True)
    v = torch.full((d,), 0.1, requires_grad=True)

    def predict(z):
        return (z * (u.square() - v.square())).sum()

    return [u, v], predict, w_star

def label_noise_step(predict, z, y_clean, delta, opt, gen):
    noisy = y_clean + delta * (2 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
    loss = 0.5 * (predict(z) - noisy) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def run_diagnet(opt_name, n_train, d=10000, kappa=50, delta=0.1,
                steps=20000, lr=1e-2, exponent=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed + 1)
    params, predict, w_star = make_diagonal_net(d, kappa, seed)
    Z = (torch.randint(0, 2, (n_train, d), generator=gen) * 2 - 1).float()
    y = Z @ w_star
    Ztest = (torch.randint(0, 2, (2000, d), generator=gen) * 2 - 1).float()
    ytest = Ztest @ w_star
    opt = make_optimizer(opt_name, params, lr, exponent)

    for _ in range(steps):
        i = torch.randint(0, n_train, (1,), generator=gen).item()
        label_noise_step(predict, Z[i], y[i], delta, opt, gen)

    with torch.no_grad():
        u, v = params
        w_hat = u.square() - v.square()
        test_loss = 0.5 * ((Ztest @ w_hat - ytest) ** 2).mean()
    return test_loss.item()
```

The narrative: SGD's slow drift has stationary points for `tr H ∝ Σ(u_i²+v_i²)`,
i.e. `‖ŵ‖₁`; Adam's has stationary points for
`tr(Diag(H)^{1/2}) ∝ Σ(|u_i|+|v_i|)`, i.e. the `ℓ_{0.5}` quasi-norm up to a
monotone power. Because
`u_i² - v_i²` is what the loss sees, the optimum forces `u_i=0 ∨ v_i=0`, so these
become genuine `ℓ₁` vs `ℓ_{0.5}` penalties on the recovered vector. The sparser
`ℓ_{0.5}` target predicts that Adam, and AdamE with `λ>0`, should need fewer samples
than SGD in this diagnostic setting.
