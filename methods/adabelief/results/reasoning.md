Let me start from the split that keeps frustrating me. When I train a CNN, Adam gets the loss down fast in the first few epochs, but the model it lands on generalizes worse than if I'd ground it out with plain SGD plus momentum. When I train a GAN, the situation flips — SGD is fragile and people reach for Adam precisely because it's more stable. So I'm always trading: fast-but-generalizes-worse against slow-but-generalizes-better, and neither is reliably stable. I want one optimizer that's fast like Adam, generalizes like SGD, and is stable enough for the hard cases. And I'd like it to be a tiny change to Adam, because Adam's machinery — the moving averages, the bias correction, the per-coordinate denominator — clearly works; I don't want to throw it out, I want to find the one piece that's wrong.

So let me look hard at what Adam's denominator actually does and ask where it misleads. Adam keeps mₜ = β₁mₜ₋₁ + (1−β₁)gₜ, an EMA of the gradient, and vₜ = β₂vₜ₋₁ + (1−β₂)gₜ², an EMA of the squared gradient, and steps by m̂ₜ/√v̂ₜ. The denominator √vₜ tracks the recent *magnitude* of the gradient in each coordinate: big gradients → small steps, small gradients → big steps. The usual justification is that this is a cheap stand-in for curvature — you want small steps in steep directions. But is gradient magnitude really a proxy for curvature? Let me pressure-test that with the simplest pictures I can draw of a 1D loss.

Region one: a flat stretch. Gradient small, and the gradient is *barely changing* as I move, so curvature is small too. The right thing is a big step — I'm in a gentle region, get across it. SGD takes a tiny step because its step is proportional to the (small) gradient. Adam takes a big step because √vₜ is small. So far Adam looks like the better curvature proxy.

Region two: a steep, narrow valley where I'm oscillating across the bottom. Gradient is large in magnitude *and* it's flipping/changing fast, so curvature is large. The right thing is a small step, to stop overshooting. SGD takes a big step (∝ large gradient) and overshoots. Adam takes a small step because √vₜ is large. Again Adam wins.

Two for two — so where's the problem? Region three, and this is the one that exposes the flaw: a region where the gradient is *large in magnitude* but *almost constant* — it's hardly changing from step to step. Large gradient, but tiny curvature. (This happens, for instance, when the learning rate is small so I'm crawling down a long steady slope.) The right thing here is a *large* step: the surface is locally gentle, the gradient is pointing steadily downhill, so I should move boldly. What does Adam do? √vₜ is *large*, because vₜ only sees the gradient's magnitude and the magnitude is large — so Adam takes a *small* step. Wrong. And what does SGD do? Step ∝ large gradient → large step. *Right.* So in exactly the regime where Adam underperforms, SGD does the sensible thing — and this smells like the source of Adam's generalization gap: it's being timid on long steady slopes where it should be making progress.

Stare at the three cases together. In all three, the *ideal* step depends on the curvature — how fast the gradient is *changing* — not on the gradient's bare magnitude. Cases one and two happen to agree with magnitude only because there magnitude and curvature move together. Case three is where they decouple, and magnitude gives the wrong answer. So the denominator I actually want should depend on the *change* in the gradient, not its size.

How do I measure "the gradient is changing"? The naive thing is |gₜ − gₜ₋₁|. But I already have a smoother, less noisy estimate of where the gradient *should* be: mₜ, the EMA of past gradients, is precisely a running prediction of the current gradient. So the natural measure of "is the gradient changing / surprising me" is the deviation of the observation from that prediction, gₜ − mₜ. If the gradient is steady, mₜ tracks it and gₜ − mₜ ≈ 0 (small curvature, want big step); if the gradient is jumping around, gₜ − mₜ is large (large curvature, want small step). And gₜ − mₜ is essentially gₜ − gₜ₋₁ smoothed, so it really is the change-in-gradient signal.

So replace Adam's vₜ = EMA(gₜ²) with sₜ = EMA((gₜ − mₜ)²), and divide by √sₜ instead of √vₜ. Everything else — the mₜ EMA, the bias correction, the structure of the step — stays exactly as in Adam. One line changes. Let me re-run the three cases with √sₜ in the denominator. Case one (flat): gradient steady and small, gₜ − mₜ ≈ 0, sₜ small → big step. Right. Case two (steep oscillating valley): gradient large *and* changing fast, gₜ − mₜ large, sₜ large → small step. Right. Case three (large gradient, not changing): gₜ ≈ mₜ, so gₜ − mₜ ≈ 0, sₜ small → *big* step. Right — and this is exactly where Adam was wrong. So the centered second moment fixes the one case the magnitude-based one botched, while keeping the two cases Adam already got right. That's the move.

Let me give this an interpretation I can trust, because "centered second moment" is mechanical and I want to know what it *means*. mₜ is my prediction of the gradient. gₜ is the observation. gₜ − mₜ is the prediction error. So 1/√sₜ = 1/√(EMA of squared prediction error) is a measure of how much I should *believe* the current gradient. If the observation lands close to what I predicted, the error is small, sₜ is small, 1/√sₜ is large — I trust this gradient and take a confident, large step. If the observation wildly disagrees with my prediction, the error is large, sₜ is large, 1/√sₜ is small — I distrust it and take a cautious, small step. The stepsize is adapted by my *belief* in the observed gradient. That's a clean Bayesian-filtering flavor: prediction, measurement, trust proportional to agreement.

And there's a statistical reading that pins down what sₜ is. mₜ ≈ E[gₜ] once the EMA bias has decayed, so sₜ = EMA[(gₜ − mₜ)²] ≈ E[(gₜ − E[gₜ])²] = Var(gₜ). So this is an *adaptive variance* method: Adam adapts to the second *moment* E[gₜ²], I'm adapting to the *variance* Var(gₜ). Where the gradient is consistent (low variance), step big; where it's noisy/conflicting (high variance), step small. That's a more sensible notion of "how reliable is this direction" than raw magnitude.

Now let me check the 2D example that made me suspicious in the first place, because it's the cleanest test of the sign issue. f(x,y) = |x| + |y|, each gradient component ±1. Start near the x-axis with x ≪ 0: the optimizer marches steadily in x (gₓ always +1) and oscillates in y (g_y flips −1, +1, −1, …). I want a big step in x and a small step in y. After enough steps the EMA bias is gone, so mₓ ≈ E[gₓ] = 1 and m_y ≈ E[g_y] = 0. Adam's vₓ ≈ E[gₓ²] = 1 and v_y ≈ E[g_y²] = 1 — *equal*, because squaring throws away the sign, so Adam takes the same-size step in both directions. That's the failure: it can't tell the consistent direction from the oscillating one. Now sₜ: sₓ ≈ Var(gₓ) = E[(1 − 1)²] = 0 (the x-gradient never deviates from its mean), and s_y ≈ Var(g_y) = E[(±1 − 0)²] = 1. So 1/√sₓ ≫ 1/√s_y: a large step in x, a small step in y — exactly right. The reason it works is that gₜ − mₜ keeps the *sign* information that gₜ² discards: in x the deviation is zero, in y it's large. Adam sees only |g|; this sees the deviation from the predicted mean.

There's one edge I have to handle before this is safe. If the gradient becomes perfectly predictable, sₜ → 0, and m̂ₜ/√ŝₜ blows up. Adam has the same risk and handles it with an ε in the denominator. I'll add the usual ε in the denominator, and I'll also add a small ε *inside* the sₜ update itself, sₜ = β₂sₜ₋₁ + (1−β₂)(gₜ − mₜ)² + ε, so that sₜ is bounded below by roughly ε at all times — it keeps the "perfect belief" case from producing an unbounded step and keeps the analysis honest (sₜ has a positive lower bound). Bias-correct mₜ and sₜ the same way Adam corrects mₜ and vₜ, since both are EMAs started at zero. And critically, no new hyperparameters: I keep Adam's α = 1e-3, (β₁, β₂) = (0.9, 0.999), ε = 1e-8. It's a drop-in.

Does this deliver the three goals I wanted? Fast convergence: the denominator is still adaptive and per-coordinate, so cases one and two behave like Adam — fast early. Generalization: in the large-gradient/low-curvature regions, where Adam is timid and SGD is bold, this now behaves like SGD (small sₜ → large step), which is the behavior that was buying SGD its better generalization — so I'd expect the generalization gap to close. Stability: the denominator reacts to the *variance/surprise* of the gradient, shrinking the step exactly when the direction is unreliable, which is the kind of damping that should help in volatile settings like GANs. There's also a curvature link I can make explicit: a finite-difference Hessian diagonal is Hᵢᵢ ≈ [gᵢ(θ + δ) − gᵢ(θ)]/δ; identifying the predicted gradient with mₜ and the observed with gₜ, √sₜ ≈ |gₜ − mₜ| tracks the magnitude of the gradient's change, i.e. the curvature — so dividing by √sₜ really is a cheap curvature scaling, which dividing by √vₜ (a magnitude) is not.

A couple of orthogonal robustness knobs that cost nothing to expose, since the structure is Adam's. I can keep a running maximum of sₜ in the denominator (the AMSGrad-style fix that makes the effective rate non-increasing). I can apply the RAdam-style rectification that tames the adaptive denominator's high variance in the first handful of steps (a warmup-like factor that, when the estimated degrees of freedom are too low, falls back to an SGD-like step). And I can decouple the weight decay (AdamW-style) by shrinking the weights directly rather than folding the penalty into the gradient. None of these change the core idea — they're the same well-understood patches that apply to any Adam-family rule.

Let me write the core step, mirroring Adam's structure. Two per-parameter buffers: `exp_avg` for mₜ and `exp_avg_var` for sₜ (named "var" because sₜ ≈ Var(gₜ)). Update mₜ, form the residual gₜ − mₜ, accumulate its square into sₜ, add ε, bias-correct, and divide.

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, weight_decouple=True, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight_decouple = weight_decouple
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)       # m_t : prediction of the gradient
                    state['exp_avg_var'] = torch.zeros_like(p.data)   # s_t : EMA of (g_t - m_t)^2  (~ Var g_t)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                # decoupled weight decay: shrink weights directly (AdamW-style)
                if self.weight_decouple and group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # m_t : EMA of the gradient (same as Adam) -- the running prediction
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # the one change vs Adam: track the squared PREDICTION ERROR, not the squared gradient
                grad_residual = grad - exp_avg                        # g_t - m_t  (keeps sign info v_t throws away)
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)  # s_t

                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # +eps inside s_t lower-bounds the denominator (safe when belief is near-perfect, s_t -> 0)
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # theta <- theta - lr * mhat / (sqrt(shat) + eps)
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
```

Recapping the chain: Adam's per-coordinate denominator is sold as curvature scaling but really tracks gradient *magnitude*, which is right in flat regions and steep oscillating valleys but wrong on long steady slopes (large gradient, low curvature), where it takes a timid step exactly where SGD's bold step generalizes better; the signal that actually distinguishes these is how much the gradient *deviates from its own running prediction* mₜ, so I replace EMA(gₜ²) with sₜ = EMA((gₜ − mₜ)²), which equals the gradient's variance, retains the sign information squaring discards, and reads as "step proportional to my belief in the observed gradient" — large step when the observation agrees with the prediction, small when it surprises me; bounded below by an ε for safety, bias-corrected like Adam, with zero new hyperparameters, so it converges fast like an adaptive method, takes SGD-like bold steps where that generalizes better, and damps itself when the direction is unreliable.
