Let me start from the thing that nags me about the optimizer I reach for by default. It keeps a decaying mean of the gradient and a decaying mean of the squared gradient, steps along the first rescaled by the square root of the second, and that adaptive rescaling is the part everyone credits for its robustness — one learning rate that works across coordinates of wildly different gradient scale. I have no complaint about the rescaling. My complaint is with the *other* half: the momentum. The first moment mₜ = μ mₜ₋₁ + (1−μ) gₜ is a decaying average of past gradients, and when I step I move in the direction of that average. That is *classical* momentum dressed up as a mean. And I have good reason to think classical momentum is the weaker of the two kinds of momentum I know.

So let me lay out the two kinds carefully, because the whole improvement lives in the difference between them. Classical momentum, the Polyak form, accumulates a velocity mₜ = μ mₜ₋₁ + gₜ and steps θₜ = θₜ₋₁ − α mₜ. Write the update out: θₜ = θₜ₋₁ − α(μ mₜ₋₁ + gₜ). It is a step of size αμ in the previous velocity direction plus a step of size α along the current gradient. The picture that justifies it: along a low-curvature direction the gradient is small but points the same way every step, so the velocity builds up and I make fast progress; along a high-curvature direction the gradient keeps flipping sign, the contributions cancel in the velocity, and the oscillation is damped. Good. But stare at the term μ mₜ₋₁. It is a step I am *going* to take, and it does not depend on gₜ at all. I commit to moving αμ in the old velocity direction before I have looked at where the current gradient points.

That is exactly the inefficiency Nesterov's accelerated gradient removes. The momentum step μ mₜ₋₁ is going to happen regardless, so why not compute the gradient *after* taking it — evaluate at the look-ahead point θₜ₋₁ − μ mₜ₋₁ instead of at θₜ₋₁? Then gₜ = ∇fₜ(θₜ₋₁ − μ mₜ₋₁); mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. The gradient now corrects the momentum step partway through, instead of being a blind addition to it. If the momentum is about to overshoot, the look-ahead gradient already points back and tempers it. This is not just a heuristic improvement; on a smooth convex objective the accelerated method attains an O(1/k²) error rate where plain gradient descent gets O(1/k), and on hard non-convex deep-learning objectives the accelerated form has been shown (Sutskever et al. 2013) to beat classical momentum, plain SGD, and Hessian-free methods. So the better momentum exists, it is well understood, and it is just sitting in non-adaptive SGD. I want to move it inside the adaptive method.

The obstacle is the look-ahead point. The clean NAG iteration wants me to evaluate the gradient at θₜ₋₁ − μ mₜ₋₁, but my training loop computes the gradient at the *current* parameters θₜ₋₁ — that is where the forward and backward pass happen. I do not want to maintain a shadow set of look-ahead parameters and shuffle between them every step; that is awkward and error-prone in a real loop. So I need the accelerated behavior while only ever differentiating at θₜ₋₁.

Sutskever's reformulation gives me exactly that, and it is worth re-deriving so I see why it is equivalent. The trick is to apply the *next* timestep's momentum step once, during the current update, rather than stepping to a look-ahead point and back. Take the gradient at the plain point, gₜ = ∇fₜ(θₜ₋₁); update the velocity, mₜ = μₜ mₜ₋₁ + gₜ; and now form the parameter step as θₜ = θₜ₋₁ − α(μₜ₊₁ mₜ + gₜ). Compare this to the classical update θₜ = θₜ₋₁ − α(μ mₜ₋₁ + gₜ): the only change is that the momentum part of the step uses μₜ₊₁ mₜ — the freshly updated velocity, which already contains gₜ — rather than μ mₜ₋₁. So the momentum step has been advanced by one: it now incorporates the current gradient, which is precisely what the look-ahead evaluation was buying me, except I never had to move off θₜ₋₁ to get it. The look-ahead is folded into the *update* instead of the *gradient point*.

Now carry this substitution into the adaptive method, term by term, because that is where the care is needed. The adaptive first moment is mₜ = μ mₜ₋₁ + (1−μ) gₜ — a mean, with the (1−μ) factor — and the bias-corrected step is α m̂ₜ/(√n̂ₜ + ε) with m̂ₜ = mₜ/(1−μᵗ). I want to expand the *update* in terms of mₜ₋₁ and gₜ, then advance the momentum step the way Sutskever did. Substitute mₜ into the step:

  m̂ₜ = mₜ/(1−μᵗ) = [μ mₜ₋₁ + (1−μ) gₜ] / (1−μᵗ) = μ mₜ₋₁/(1−μᵗ) + (1−μ) gₜ/(1−μᵗ).

There are the two pieces I expected: a momentum piece riding on mₜ₋₁ and a fresh-gradient piece riding on gₜ, each with the same bias-correction denominator. Now apply the substitution: replace the *current* momentum piece μ mₜ₋₁/(1−μᵗ) with the *next* one, μ mₜ/(1−μᵗ⁺¹) — advance the momentum term by one step so it uses the freshly updated mₜ (which already contains gₜ), exactly as the simplified NAG advanced μ mₜ₋₁ to μₜ₊₁ mₜ. That gives a Nesterov-corrected step

  m̂ₜ = μ mₜ/(1−μᵗ⁺¹) + (1−μ) gₜ/(1−μᵗ).

The fresh-gradient piece keeps the denominator (1−μᵗ) it was born with; the momentum piece, now advanced, takes the denominator (1−μᵗ⁺¹) appropriate to the advanced moment. The second moment is untouched — n̂ₜ = nₜ/(1−νᵗ), the adaptive rescaling stays exactly as it was — and the parameter step is

  θₜ = θₜ₋₁ − α m̂ₜ/(√n̂ₜ + ε).

The adaptive step I want is now a bias-corrected blend of two pieces: the current gradient, still visible as its own term, and the first moment advanced by one momentum step. The step reacts to the current gradient *twice* — once directly and once through its presence in the advanced moment — which is the accelerated correction, while everything that made the optimizer robust, the per-coordinate √nₜ rescaling, stays unchanged.

There is one subtlety I glossed: I let myself write μ as a single constant, but the Nesterov blend can use a scheduled momentum cache. The stored first-moment buffer remains the ordinary adaptive-method EMA, mₜ = β₁mₜ₋₁ + (1−β₁)gₜ; the scheduled values μₜ and μₜ₊₁ are the coefficients that weight the direct-gradient and advanced-momentum pieces of the step. Once those coefficients vary with t, the normalizers attached to the blend cannot be the naive μᵗ term from the constant-coefficient derivation. I keep a scalar product mu_productₜ = Π_{i=1}^{t} μᵢ after multiplying by the current μₜ. The fresh-gradient piece is normalized by 1 − mu_productₜ, while the advanced-momentum piece looks one scheduled step ahead and is normalized by 1 − mu_productₜ μₜ₊₁ = 1 − Π_{i=1}^{t+1} μᵢ. Why schedule μ at all? Early in training the moment estimates are built from very few gradients and are noisy; leaning hard on the accelerated momentum term then commits to a poorly estimated direction. So start the cache near half strength and let it rise toward β₁ as the estimates stabilize. A schedule that does this smoothly is μₜ = β₁·(1 − ½·0.96^{t·ψ}) with a small decay rate ψ: near the first update 0.96^{t·ψ} is close to 1, so μₜ is close to β₁/2, and as t grows 0.96^{t·ψ} → 0 so μₜ → β₁. The running product belongs to this scheduled Nesterov blend; the stored first-moment EMA itself still uses β₁.

Now the regularizer, because for a large model I will want weight decay and it has to compose correctly with the adaptive rescaling. The wrong way is to fold an L2 penalty into the gradient, gₜ ← gₜ + λθₜ: that term then flows through both moments and gets divided by √n̂ₜ, so the effective shrinkage on each weight is scaled by its own 1/(√n̂+ε) — coordinates with large second moment get *less* decay, which is not what "decay the weights" should mean. The fix is to decouple it: shrink the weights directly before the adaptive step, θ ← (1 − αλ)θ, then apply the adaptive Nesterov step. Now every weight is pulled toward zero by the same factor (1 − αλ) regardless of its gradient history, and the decay no longer interacts with the moment buffers. I will use this decoupled form — it is the regularizer that actually behaves like weight decay under an adaptive method, and it slots in cleanly ahead of the Nesterov-corrected step.

Let me reconcile the math I derived with how it is cleanest to *implement*, because the buffer I store does not have to be the bias-corrected quantity. I keep the raw first moment mₜ = β₁ mₜ₋₁ + (1−β₁) gₜ and the raw second moment nₜ = β₂ nₜ₋₁ + (1−β₂) gₜ² as the two state buffers, exactly as the non-accelerated optimizer would, and I apply the scheduled Nesterov blend and its normalizers at *step time*. So the step decomposes into two additive applications: one along the current gradient, weighted (1 − μₜ) and divided by (1 − mu_product); one along the stored first moment, weighted μₜ₊₁ and divided by (1 − mu_product·μₜ₊₁); both then divided by the shared adaptive denominator √(nₜ/(1−β₂ᵗ)) + ε; the whole thing scaled by −α. Writing it as two `addcdiv_` operations onto the parameter, after the weight-decay branch and the two moment EMAs, is the faithful and allocation-light form.

So end to end, for each parameter at step t:

  bump the step counter t;
  compute μₜ = β₁(1 − ½·0.96^{tψ}) and μₜ₊₁ = β₁(1 − ½·0.96^{(t+1)ψ}); accumulate mu_product ·= μₜ;
  decoupled weight decay: θ ← θ·(1 − αλ);
  moment EMAs: m ← β₁ m + (1−β₁) g;  n ← β₂ n + (1−β₂) g²;
  adaptive denominator: denom = √(n/(1−β₂ᵗ)) + ε;
  the fresh-gradient step: θ ← θ − α·(1−μₜ)·g / (1 − mu_product) / denom;
  the advanced-momentum step: θ ← θ − α·μₜ₊₁·m / (1 − mu_product·μₜ₊₁) / denom.

```python
import torch
from torch.optim.optimizer import Optimizer


class NesterovAdaptiveOptimizer(Optimizer):
    """Adaptive moment estimation with the momentum upgraded from classical to
    simplified-Nesterov (the look-ahead absorbed into the update), with decoupled
    weight decay. The second-moment adaptive rescaling is unchanged."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum_decay=momentum_decay,
                        decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            psi = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["mu_product"] = torch.tensor(1.0)
                    state["exp_avg"] = torch.zeros_like(p)       # first moment m
                    state["exp_avg_sq"] = torch.zeros_like(p)    # second moment n
                state["step"].add_(1)
                t = int(state["step"].item())
                m, n = state["exp_avg"], state["exp_avg_sq"]

                # momentum schedule: warm up mu from ~beta1/2 toward beta1
                mu_t  = beta1 * (1.0 - 0.5 * (0.96 ** (t * psi)))
                mu_t1 = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * psi)))
                state["mu_product"].mul_(mu_t)
                mu_product = float(state["mu_product"].item())

                # weight decay matches the optimizer branch; decoupled shrink is AdamW-style
                if wd != 0:
                    if group["decoupled_weight_decay"]:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g.add(p, alpha=wd)

                # moment EMAs (raw; bias correction applied at step time)
                m.lerp_(g, 1 - beta1)
                n.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # shared adaptive denominator (second-moment rescaling, unchanged)
                denom = (n / (1 - beta2 ** t)).sqrt_().add_(eps)

                # simplified-Nesterov step, split into the two additive pieces:
                #  fresh gradient, weight (1 - mu_t) / (1 - mu_product)
                p.addcdiv_(g, denom, value=-lr * (1 - mu_t) / (1 - mu_product))
                #  advanced first moment, weight mu_{t+1} / (1 - mu_product * mu_{t+1})
                p.addcdiv_(m, denom, value=-lr * mu_t1 / (1 - mu_product * mu_t1))
        return loss
```

I end with the adaptive optimizer's robustness intact because the second-moment per-coordinate rescaling stays untouched, but the momentum is no longer the classical kind that commits to the old velocity direction before consulting the current gradient. Accelerated momentum corrects that step with a look-ahead gradient; I cannot evaluate at the look-ahead point inside an ordinary training loop, so I use the simplified-NAG reformulation that advances the momentum step by one and folds the look-ahead into the *update* while differentiating only at the current parameters. Substituting that into the adaptive step gives an advanced-momentum piece and a fresh-gradient piece, whose scheduled-momentum normalizers differ because one looks through μₜ₊₁ and the other stops at μₜ, while the stored first moment remains the usual β₁ EMA. Decoupled weight decay then shrinks the weights uniformly before those two adaptive `addcdiv_` pieces, instead of letting the regularizer be distorted by the per-coordinate rescaling.
