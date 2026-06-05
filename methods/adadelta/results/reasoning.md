Let me start from the thing that wastes the most of my time: picking the learning rate. The update is x_{t+1} = x_t − η·g_t, and η is a single global number I have to guess. Guess too big and the loss diverges; too small and it crawls. And it's worse than a one-time guess — the right η differs between layers (the gradient magnitudes in a deep net span orders of magnitude), so one global number is never right everywhere at once. I want to get rid of the hand-tuned global η entirely: a learning rate that's computed automatically, separately per dimension, from first-order information only, with minimal overhead over SGD.

What's already on the table to lean on? Newton's method, Δx = −H⁻¹g, divides the gradient by the curvature and gives the perfect descent step on a convex quadratic — but H is hopeless to form for a big model, so that's aspirational, not usable. Momentum keeps a decaying accumulation of past updates and does something genuinely nice per-dimension — it accelerates directions whose gradients keep the same sign and damps directions whose gradients flip — but it still rides on a global η. The one that's closest to what I want is AdaGrad: Δx_t = −η/√(Σ_{τ≤t} g_τ²)·g_t. The per-dimension denominator is the running ℓ₂ norm of that coordinate's gradients, so big-gradient dimensions get small rates and small-gradient dimensions get big rates. That evens out progress across dimensions — exactly the per-layer-scale problem I have — and the growing denominator anneals the rate over time for free, all from first-order quantities.

So why not just use AdaGrad? Look at what its denominator does over a long run. It sums g_τ² from τ=1 to the current t, every term positive, so the sum only grows. The effective rate η/√(sum) therefore shrinks monotonically and eventually becomes infinitesimal — training grinds to a halt before it's done. That's flaw one: the rate decays to zero. Flaw two follows from the same all-time accumulation: the sum is dominated by whatever happened early, so if the initial gradients are large the rate is permanently depressed, and the only lever to fix that is... cranking the global η back up. Which means AdaGrad is sensitive to initialization *and* still saddles me with tuning η. Both flaws trace to one design choice: accumulating from the very beginning of training.

The fix for the decay is to stop accumulating forever. Instead of summing over all t, accumulate over a fixed *window* of the most recent w gradients. Then the denominator can't run off to infinity — it's a local estimate of the recent gradient scale, and learning keeps making progress no matter how many steps have passed. But literally storing w past squared gradients per dimension is wasteful. The standard trick to get a windowed average without a buffer is an exponentially decaying running average:

E[g²]_t = ρ·E[g²]_{t-1} + (1−ρ)·g_t²,

one number per dimension, ρ playing the role of the window (the same decay idea momentum uses). I need the square root of this for a denominator that scales like a gradient magnitude, and I'll add a small ε before the root to keep it from blowing up when gradients vanish — the same conditioning role μ plays in the diagonal-Hessian methods. Define RMS[g]_t = √(E[g²]_t + ε). The windowed update is then

Δx_t = −η/RMS[g]_t · g_t.

That kills flaw one (no more decay to zero) and softens flaw two (recent gradients, not ancient ones, set the scale). But η is still sitting right there in the numerator. I haven't actually removed the global learning rate — I've just made it a bit more forgiving. I need a principled reason to replace η with something automatic, not another fudge factor.

I should check units before I decide what can replace η. A parameter x has some hypothetical units, and a change Δx applied to it had better be in those same units, or the update is dimensionally nonsense. For SGD and momentum, Δx ∝ g = ∂f/∂x, and if f is unitless then ∂f/∂x has units of 1/x — so Δx is in units of 1/x, the wrong units. For my windowed rule and for AdaGrad, the update is a ratio of gradient quantities times a gradient, so the gradient units cancel and the update comes out unitless, also wrong. Newton's descent step has a minus sign, Δx = −H⁻¹g, but the sign only chooses the descent direction; the scale H⁻¹g has units (∂f/∂x)/(∂²f/∂x²), which is x. So the second-order methods aren't just more accurate; they are dimensionally consistent because the second-derivative term supplies the missing units. My windowed rule is missing whatever supplies those units.

So let me ask the second-order method what quantity I'm missing. Take the one-dimensional Newton descent step, Δx = −(∂f/∂x)/(∂²f/∂x²), and rearrange for the inverse curvature:

1/(∂²f/∂x²) = −Δx / (∂f/∂x).

The minus sign is already handled by choosing the update direction −g_t, so the scale I need is the magnitude of Δx divided by a gradient. The denominator of my rule, RMS[g], already supplies the ∂f/∂x part because it has units of a gradient. What's missing — the thing that would give the whole update units of x — is a quantity in the numerator with units of Δx. So I should multiply by a measure of Δx and let it replace η. The numerator η/RMS[g] becomes (something with units of Δx)/RMS[g], and now the scale behaves like Δx/g, i.e. an inverse-curvature scale, with no free global learning rate left.

What do I put in the numerator? It should be the size of the update at the current step, Δx_t. But that's circular — Δx_t is exactly what I'm computing. So assume the curvature is locally smooth and use the recent *past* updates as a stand-in: take the same exponentially-decaying RMS, now of the Δx values, and use the value from the previous step:

E[Δx²]_t = ρ·E[Δx²]_{t-1} + (1−ρ)·Δx_t², RMS[Δx]_t = √(E[Δx²]_t + ε),

and the update is

Δx_t = − RMS[Δx]_{t-1} / RMS[g]_t · g_t.

There is no η. The per-dimension effective rate is the ratio of two RMS quantities — the recent update scale over the recent gradient scale — which, by the rearrangement above, is a first-order estimate of a positive inverse diagonal-curvature scale. Because both RMS terms are nonnegative, the ratio is always positive, so the update always points along −g_t: I'm always descending, just as Becker & LeCun guaranteed with their absolute value on the diagonal Hessian.

The ε in the numerator RMS isn't just conditioning — it bootstraps the very first step, where Δx_0 = 0 would otherwise make the numerator zero and freeze the parameter, and it keeps the step from dying out later if the recent updates happen to get tiny. RMS[Δx]_{t-1} uses updates only up to the previous step because it can't include Δx_t, which doesn't exist yet, so the numerator lags the denominator by one step. That lag is useful: if a large, sudden gradient arrives, it immediately spikes RMS[g]_t in the denominator and shrinks the effective rate this step, while the numerator, based on past smaller updates, hasn't reacted yet. So a gradient spike is automatically dampened, which is exactly the robustness I want in a noisy distributed setting.

This rule inherits something useful from each ancestor. It always follows −g_t, like SGD. Its numerator accumulates past update information over a window, an acceleration term in the spirit of momentum. Its denominator uses per-dimension squared-gradient information to even out progress across dimensions, like AdaGrad — but windowed, so it does not force the effective rate to vanish. And the whole ratio approximates an inverse diagonal Hessian, like the second-order methods — but it costs only the one gradient evaluation per step that I was already doing, with no explicit Hessian and no extra backward pass.

I should sanity-check the dynamics qualitatively. If backprop makes lower-layer gradients smaller, the denominator in those layers becomes smaller too, so the effective rate there becomes larger and can help those layers keep up. Late in training, as both the gradients and the updates shrink toward zero, the ε terms start to dominate both RMS quantities, so the ratio drifts toward 1; a rate of 1 would normally be wildly too large, but it only happens once g and Δx are already tiny, and the resulting updates smoothly tend to zero — an implicit annealing with no schedule imposed by hand. Two hyperparameters remain, ρ and ε, but they only shape the window and the conditioning; the sharp, must-tune η is gone from the derived update.

The rule now has a compact form. Per dimension keep two running accumulators, E[g²] and E[Δx²], both initialized to zero:

  given decay ρ, constant ε; E[g²]_0 = 0, E[Δx²]_0 = 0
  for t = 1, 2, …:
    g_t = ∇f(x_t)
    E[g²]_t      = ρ·E[g²]_{t-1}   + (1−ρ)·g_t²
    Δx_t         = − RMS[Δx]_{t-1} / RMS[g]_t · g_t          # RMS[·] = √(·+ε)
    E[Δx²]_t     = ρ·E[Δx²]_{t-1}  + (1−ρ)·Δx_t²
    x_{t+1}      = x_t + Δx_t

In code, two per-parameter buffers — call them `square_avg` for E[g²] and `acc_delta` for E[Δx²]. Update `square_avg`, take its RMS as `std`, divide the RMS of `acc_delta` (the previous step's, since we update `acc_delta` *after* applying) by it, multiply by the gradient to get `delta`, apply it, then fold `delta` into `acc_delta`:

```python
import torch
from torch.optim.optimizer import Optimizer


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # lr is an optional outer scale; lr=1.0 is the pure unit-corrected method (no global learning rate)
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rho, eps = group['rho'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")

                state = self.state[p]

                # per-dimension accumulators: E[g^2] and E[dx^2], both start at 0
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)   # E[g^2]
                    state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)    # E[dx^2]

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # E[g^2]_t = rho*E[g^2]_{t-1} + (1-rho)*g^2
                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()                    # RMS[g]_t = sqrt(E[g^2]+eps)
                # delta is the positive preconditioned gradient; the applied update is -lr*delta
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.add_(delta, alpha=-group['lr'])                    # lr=1.0 gives the derived update
                # E[dx^2]_t = rho*E[dx^2]_{t-1} + (1-rho)*delta^2
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        return loss
```

Recapping the chain: I want to drop the hand-tuned global learning rate, and AdaGrad gets me a per-dimension rate from first-order info but its all-time squared-gradient denominator decays the rate to zero and is hostage to the initial gradients; replacing that sum with an exponentially-decaying windowed average, E[g²]_t = ρ·E[g²]_{t-1} + (1−ρ)·g_t², fixes the decay but still leaves a global η in the numerator; a units argument then shows every first-order rule has dimensionally wrong updates while Newton's −H⁻¹g has the correct scale, and rearranging the one-dimensional Newton step reveals the missing numerator is a quantity with units of Δx, which I supply (by local-smoothness) as the RMS of past updates — yielding Δx_t = −RMS[Δx]_{t-1}/RMS[g]_t · g_t, a learning-rate-free, always-descending, per-dimension inverse-diagonal-Hessian approximation that inherits SGD's descent direction, momentum's accumulation, AdaGrad's per-dimension evening-out (now windowed), and the second-order methods' correct units and curvature scaling, all for one gradient evaluation per step.
