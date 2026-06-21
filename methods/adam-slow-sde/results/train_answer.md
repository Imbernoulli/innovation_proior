Over-parameterized neural networks do not converge to a single optimal point; they land on a connected manifold of global minimizers that all give near-zero training loss but generalize differently. The optimizer's implicit bias — the slow force that continues to act after training loss has vanished — therefore determines which minimizer is selected. For plain SGD, this force is now fairly well understood: gradient noise drives a slow drift along the manifold that reduces a sharpness measure built from the Hessian and the noise covariance, and under label noise that measure collapses to the trace of the Hessian. But modern deep learning almost never uses plain SGD; it uses Adam, RMSProp, and other adaptive gradient methods that rescale each coordinate by a running estimate of the squared gradient. Existing analyses of implicit bias fail here because they rely on rotational symmetry, on the noise entering the update unmodified, or on short transients rather than the full O(η⁻²) manifold horizon. Prior convergence guarantees for adaptive methods are also the wrong shape: they bound averaged gradients, hold only in expectation, or do not vanish with the learning rate, none of which is enough to justify a long-horizon projected dynamics.

The new method is called Adam-Slow-SDE. It is a continuous-time characterization of what adaptive gradient methods do after they reach a manifold of minimizers, covering Adam, RMSProp, block- or layer-scaled variants, and even Kronecker-factored preconditioners through a single template. The update maintains a momentum vector and a second-moment statistic, then takes a preconditioned step: m_{k+1} = β1 m_k + (1−β1) g_k, v_{k+1} = β2 v_k + (1−β2) V(g_k g_k^T), and θ_{k+1} = θ_k − η S(v_{k+1}) m_{k+1}, where V is a linear map from gradient outer products to nonnegative second-moment states and S turns those states into a positive-definite preconditioner. For Adam, V takes the diagonal and S is Diag(1/(√v+ε)). The analysis requires two conceptual changes from the SGD story. First, the projection that divides out fast convergence must follow the preconditioned flow ẋ = −S∇L(x), not the gradient flow, because that is the direction in which the optimizer actually relaxes. Second, the second-moment decay rate must scale as 1−β2 = Θ(η²), dubbed the 2-scheme, so that v moves on the same slow clock as the manifold drift: slow enough to be approximately constant within a single giant step, but able to accumulate a Θ(1) change over the full O(η⁻²) horizon.

With these choices, the projected state (Φ_S(θ_k), v_k) converges to a slow SDE on the manifold. The manifold coordinate ζ follows dζ = S_t ∂Φ_{S_t}(ζ) S_t Σ^{1/2}(ζ) dW_t + ½ S_t ∂²Φ_{S_t}(ζ)[S_t Σ(ζ) S_t] dt, while the preconditioner state follows dv = c(V(Σ(ζ)) − v) dt with c = (1−β2)/η². The drift is equivalently the projected negative semi-gradient of μ(ζ,v) = ⟨∇²L(ζ), S_t Σ(ζ) S_t − Σ_∥(ζ;S_t)⟩, where Σ_∥ is the tangent-projected covariance, so Adam performs adaptive semi-gradient descent on a sharpness measure that differs from SGD's. A side result is that momentum does not change the implicit bias; β1 drops out of the slow SDE. Under label noise, where the gradient-noise covariance is proportional to the Hessian, the diffusion term vanishes because the preconditioned projection kills normal directions, and the dynamics collapses to a deterministic flow. For a tunable second-moment exponent λ in S(v) = Diag(1/(v^λ+ε)), the fixed points satisfy ∇_Γ tr(Diag(H)^{1−λ}) = 0. Thus SGD (λ=0) implicitly regularizes tr(H), while Adam (λ=1/2) implicitly regularizes tr(Diag(H)^{1/2}). On a diagonal linear network where diag(H) = 4θ², this corresponds to minimizing the ℓ_1 norm of the recovered signal for SGD and the ℓ_{0.5} quasi-norm for Adam, predicting sparser recovery with fewer samples. In deep matrix factorization, where the useful bias is toward low rank and is captured by tr(H), Adam's different sharpness target is expected to hurt, illustrating that the bias is not universally better but is genuinely distinct.

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
