Let me start from the thing that wastes the most of my time: picking the learning rate. The update is x_{t+1} = x_t − η·g_t, and η is a single global number I have to guess. Guess too big and the loss diverges; too small and it crawls. And it's worse than a one-time guess — the right η differs between layers (the gradient magnitudes in a deep net span orders of magnitude), so one global number is never right everywhere at once. I want to get rid of the hand-tuned global η entirely: a learning rate that's computed automatically, separately per dimension, from first-order information only, with minimal overhead over SGD.

What's already on the table to lean on? Newton's method, Δx = −H⁻¹g, divides the gradient by the curvature and gives the perfect descent step on a convex quadratic — but H is hopeless to form for a big model, so that's aspirational, not usable. Momentum keeps a decaying accumulation of past updates and does something genuinely nice per-dimension — it accelerates directions whose gradients keep the same sign and damps directions whose gradients flip — but it still rides on a global η. The one that's closest to what I want is AdaGrad: Δx_t = −η/√(Σ_{τ≤t} g_τ²)·g_t. The per-dimension denominator is the running ℓ₂ norm of that coordinate's gradients, so big-gradient dimensions get small rates and small-gradient dimensions get big rates. That evens out progress across dimensions, which is the per-layer-scale issue I started with, and the growing denominator anneals the rate over time for free, all from first-order quantities.

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

There is no η. The per-dimension effective rate is the ratio of two RMS quantities — the recent update scale over the recent gradient scale — which, by the rearrangement above, should behave like an inverse-curvature scale Δx/g. Because both RMS terms are nonnegative, the ratio is always positive, so the update always points along −g_t: I'm always descending, the same safety the diagonal-Hessian methods buy with an absolute value on the curvature.

The ε in the numerator RMS isn't just conditioning — it bootstraps the very first step, where Δx_0 = 0 would otherwise make the numerator zero and freeze the parameter, and it keeps the step from dying out later if the recent updates happen to get tiny. RMS[Δx]_{t-1} uses updates only up to the previous step because it can't include Δx_t, which doesn't exist yet, so the numerator lags the denominator by one step. That lag might be useful: if a large, sudden gradient arrives, it immediately spikes RMS[g]_t in the denominator and shrinks the effective rate this step, while the numerator, based on past smaller updates, hasn't reacted yet. So a gradient spike could be damped on the very step it lands — a property I'd want in a noisy distributed setting, and one I'll put a number to below rather than leave as a hope.

This rule inherits something from each ancestor. It always follows −g_t, like SGD. Its numerator accumulates past update information over a window, an acceleration term in the spirit of momentum. Its denominator uses per-dimension squared-gradient information to even out progress across dimensions, like AdaGrad — but windowed, so it does not force the effective rate to vanish. And the whole ratio is meant to track an inverse diagonal Hessian, like the second-order methods — but it costs only the one gradient evaluation per step that I was already doing, with no explicit Hessian and no extra backward pass. Several of those claims I've only argued by units and intuition, though. Before I trust the rule I want to run the actual recurrence on a case where I know the right answer, and watch the numbers.

The cleanest test is a 1-D quadratic f(x) = ½·a·x², where g = a·x and the curvature is a constant H = a. Newton's step here is Δx = −g/H = −x, which lands on the minimum at 0 in one move; the inverse-curvature scale I'm supposedly estimating is exactly 1/a. So I'll take a = 4 (true scale 0.25), start at x = 1 with ρ = 0.9, ε = 1e-6, and print the per-step ratio RMS[Δx]/RMS[g]. The first handful of steps:

  step   x        g       RMS[g]   RMS[Δx]_{t-1}   ratio       Δx
   1   1.00000   4.0000   1.2649     0.001000     0.000791   -0.003162
   2   0.99684   3.9874   1.7407     0.001414     0.000812   -0.003240
   5   0.98698   3.9479   2.5415     0.002158     0.000849   -0.003353
  10   0.97003   3.8801   3.1713     0.002807     0.000885   -0.003434

This is sobering. The ratio isn't anywhere near 0.25 — it's ~0.0008, three hundred times too small, and x has barely moved. The reason is staring at me in the table: RMS[Δx]_0 = √(0 + ε) = √(1e-6) = 0.001, because E[Δx²] starts at zero, so the entire numerator is pinned at the ε floor until the accumulator of past updates fills up. The rule has to *bootstrap* its own step size, and at first it crawls. That's a genuine cost I hadn't seen from the algebra — the unit correction buys me a learning-rate-free rule, but it pays for it with a slow warm-up. Let me make sure it does eventually move, by running the same quadratic much longer:

  t=   1   x=1.00e+00   ratio=0.000791
  t= 100   x=6.55e-01   ratio=0.001314
  t= 500   x=-1.97e-03  ratio=0.506565
  t=1000   x=-1.00e-02  ratio=0.500236
  t=5000   x=-2.92e-02  ratio=0.500028

So it does get there — by t≈500 it has essentially reached the minimum, and the ratio settles to a steady value. But that steady value is 0.5, not 1/a = 0.25. So the ratio is *not* literally the inverse curvature; it's a quantity of the same character — positive, with units of Δx/g, scaling inversely with gradient size — but off by a constant that depends on the regime. That's the honest statement: this is an *estimate* of an inverse-diagonal-curvature scale, not an equality, and I should describe it that way rather than claim it reproduces Newton. The reassuring part is that it gets the per-dimension *scale* right enough to converge without any η at all.

Now the two dynamical claims I waved at. First, spike robustness. I warm the rule up on steady gradients of magnitude 1 (so RMS[g] ≈ 1 and |Δx| ≈ 0.0056), then inject a single gradient of magnitude 100 and look at the update it produces. The spike pushes E[g²] up so RMS[g] jumps to 31.6, while RMS[Δx] in the numerator still reflects the pre-spike updates. The result: |Δx| on the spike step is 0.0178 — only 3.17× the steady update, even though the gradient grew 100×. A fixed-rate rule Δx = −r·g would have moved 100× as far. The factor 3.17 ≈ 100/31.6 is exactly the denominator absorbing the spike, which is what the one-step numerator lag predicted. Good — the robustness is real, not just plausible.

Second, the late-training behavior. I feed the rule a gradient that decays geometrically toward 0 and watch the ratio:

  g_t        RMS[g]      RMS[Δx]    ratio
  1.0e+00    3.16e-01    1.00e-03   0.00316
  5.7e-03    7.55e-02    1.06e-03   0.01404
  2.9e-05    5.52e-03    1.00e-03   0.18111
  7.8e-10    1.00e-03    1.00e-03   0.99961
  2.3e-14    1.00e-03    1.00e-03   1.00000

As g and Δx both shrink, the ε floor (√ε = 1e-3) comes to dominate *both* RMS terms, so they become equal and the ratio climbs to exactly 1. A rate of 1 sounds dangerous, but it only arrives once g is ~1e-14, so the update Δx = ratio·g is ~1e-14 too — vanishing. So the ε terms give an implicit annealing: the step size doesn't blow up at the end, it dies smoothly, with no schedule imposed by hand. Both checks land where the intuition said they would, but now I've seen the numbers.

Two hyperparameters remain, ρ and ε, but they only shape the window and the conditioning; the sharp, must-tune η is gone from the derived update. The one caveat the numbers added is the warm-up: with E[Δx²] starting at zero the first many steps are throttled to the ε floor, so this rule trades a fast, brittle, hand-tuned start for a slow, robust, automatic one.

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
