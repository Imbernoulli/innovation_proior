OK, let me think about this from the actual constraint I'm under, which is wall-clock time. Training a big convolutional net on ImageNet is slow on one device, and the only knob that turns more hardware into less time without touching the model is the mini-batch. Data-parallel synchronous SGD: split the global batch of B examples across N workers, each worker computes the gradient on its B/N chunk, sum the partials, do one update. If I want each worker busy enough to be efficient, its chunk has to be sizeable, so the way I add workers to go faster is by pushing the global batch B up. More nodes, bigger batch, fewer-but-heavier steps. That's the whole appeal.

So the question is just: how big can B get before something breaks, and what breaks?

The obvious tension is that I'm fixing the number of epochs — I want the same amount of data seen, just faster. If I multiply B by k, then with the same epochs I do k times fewer weight updates. The optimizer is being asked to reach the same place in k-fold fewer steps. So each step has to cover more ground, which means a bigger learning rate. How much bigger?

Let me actually work out the natural scaling instead of guessing. Take two consecutive steps at batch B with rate λ:

w_{t+2} = w_t − λ·(1/B)( Σ_{i=1}^B ∇L(x_i, w_t) + Σ_{j=1}^B ∇L(x_j, w_{t+1}) ).

Now compare to a single step at batch 2B with some rate λ₂:

w_{t+1} = w_t − λ₂·(1/(2B)) Σ_{i=1}^{2B} ∇L(x_i, w_t).

These line up if λ₂ = 2λ — but only if I'm allowed to say ∇L(x_j, w_{t+1}) ≈ ∇L(x_j, w_t), i.e. the weights barely moved during the first sub-step so the second sub-step's gradient is evaluated at essentially the same point. So the linear rule — grow B by k, grow λ by k, leave momentum and weight decay alone — comes straight out of that approximation. Notice what the approximation says, though: it is exactly "the step is small." Once λ is large the step isn't small, w_{t+1} is far from w_t, and ∇L(x_j, w_{t+1}) is nothing like ∇L(x_j, w_t). The rule licenses a big rate and then the big rate invalidates the rule. So I should expect it to hold for a while and then break, and the break should arrive precisely when the per-step movement stops being negligible.

There's a more cautious alternative I should weigh before committing. The mini-batch gradient is an average of B samples, so its standard error shrinks like 1/√B; from a pure noise-reduction view I should only raise λ like √B, not B. That's gentler and it won't outrun its own justification as fast. Which should I prefer? The two disagree more and more as B grows: at B = 4096 vs a 256 baseline, k = 16, so linear says ×16 and square-root says ×4 — a factor of four apart, and the gap widens with B. Square-root is the safe bet for not diverging, but it is also deliberately leaving optimization speed on the table at exactly the batch sizes I care about, and the reported experience is that the linear rule is the one that actually keeps accuracy up to a moderate ceiling while square-root is too timid below it. So I'll take linear scaling as the aggressive baseline — it is the one whose justification I can see, and the one that fails for a reason I can name — rather than the conservative rule whose only argument is "be safe." I expect linear scaling to be the thing that breaks, and I want to understand the break.

There's also a practical softener that the same algebra suggests. The initial phase is where the big rate hurts most — the very first steps, before anything has settled, are where w moves most violently relative to its own scale, so that's where the "step is small" approximation is most badly violated. So ramp λ from a small safe value up to the target over the first few epochs, then hand off to the regular decay. That warmup, plus linear scaling, is the recipe to try, and it's the one reported to carry a deep residual network to B = 8K at baseline accuracy.

I can't run an ImageNet training in my head, so let me be honest about what I know rather than narrate an experiment. What's reported is that this recipe holds up on networks with Batch Normalization and on residual architectures, but for an AlexNet-style network the same recipe stops matching the small-batch baseline somewhere around B = 2K and degrades from there — best accuracy in the low-50s at B = 4K, worse by B = 8K, against a ~58% baseline at B = 256. I'll take that ceiling as the phenomenon to explain. The thing I can do here, and the thing that matters, is figure out the mechanism — because if I understand why it caps out, I'll know whether warmup-plus-linear can be pushed or whether it's the wrong tool.

Two structural reasons the cap could exist, and they call for different fixes. One: it's the generalization gap people talk about — large batches supposedly drift toward sharp minimizers that test worse. If that's the cause, the *training* loss at B = 8K would look fine while the *test* loss lags, so the train–test loss gap should widen at large batch. Two: the network is simply not being optimized far enough — under-training — in which case the training loss itself is worse and the train–test gap is unchanged. These are distinguishable by a single measurement, the train–test loss gap at small vs. large batch, and the reported result is that the gap does *not* widen at B = 8K. So whatever the residual is, it is on the optimization side, not the generalization side. That's the useful branch: under-optimization is something I can attack by changing the optimizer; a sharp-minima generalization gap would not be.

So the real question becomes: why can't I just optimize harder by cranking the rate? Because the rate is exactly what diverges. Let me stare at the update and ask what "diverging" means mechanically. Standard SGD uses one rate λ for the whole network: w_{t+1} = w_t − λ·∇L(w_t). For a given layer, the size of the step relative to the size of that layer's weights is ‖λ·∇L(w)‖ / ‖w‖. If that's around or above 1, the layer's parameters move as much as their own magnitude in one step — you've thrown the weights across the landscape, and that's the blow-up. So the safe condition, per layer, is that λ·‖∇L(w)‖ stays well below ‖w‖, i.e. λ stays well below ‖w‖ / ‖∇L(w)‖.

Now the question that decides everything: is that bound ‖w‖/‖∇L(w)‖ roughly the same across layers, or not? If it's uniform, one global λ can satisfy it everywhere and the whole "single rate" framing is fine — I'd be barking up the wrong tree and the cap is something else. So this has to be measured: take the network after a single iteration and tabulate, layer by layer, ‖w‖, ‖∇L(w)‖, and their ratio. The reported measurement is that the ratios are nowhere near uniform — an early convolutional layer sits at a single-digit ratio, a late fully-connected layer sits in the thousands, and weights and biases within the same layer differ too. Orders of magnitude apart.

I want to make sure I actually believe the consequence of that spread, not just repeat it, so let me push two layers with realistically different ratios through both the global-rate update and the per-layer idea and see what each does to the relative step. Take layer 1 with ‖w₁‖ = 0.5, ‖g₁‖ = 10 (ratio 0.05, a small-ratio early layer) and layer 2 with ‖w₂‖ = 50, ‖g₂‖ = 0.01 (ratio 5000, a large-ratio late layer). Pick the global λ so layer 1 takes a safe relative step of 0.1 of its own norm: λ·‖g₁‖ = 0.1·‖w₁‖ gives λ = 0.1·0.5/10 = 0.005. Now apply that same λ to both layers and look at ‖λ·g‖/‖w‖:

- layer 1: 0.005·10 / 0.5 = 0.100  (the target)
- layer 2: 0.005·0.01 / 50 = 1e-6

Layer 2 moves a millionth of its own norm. For layer 2 to take the same 0.1 relative step it would need λ = 0.1·50/0.01 = 500 — a factor of 100,000 larger than what layer 1 can survive, which is exactly the ratio of the two ratios. There is no single λ that is simultaneously ≈ safe for layer 1 and ≈ useful for layer 2: pick λ for the small-ratio layer and the big-ratio layer barely budges; pick λ for the big-ratio layer and the small-ratio layer is thrown across the landscape. So the global rate is pinned by the worst (smallest-ratio) layer, and every layer with a larger ratio is starved. That's a concrete, computed picture of the residual under-training — not a generalization gap, just the worst layer capping the rate for everyone.

And now warmup makes mechanical sense — it's a crutch for exactly this. The earliest steps are the most sensitive: the ratios are changing quickly, the network has not settled, and the target large-batch rate is exactly the thing that can make a small-ratio layer jump farther than its own scale. A small starting rate is the one value that's safe for *all* layers at once, and the ramp delays the target rate until those layer-wise ratios are less volatile. Warmup is sneaking up on a moving, per-layer constraint with a single scalar. It's the right instinct applied through the wrong variable — it tames *when* the global rate arrives, but it never fixes that one number can't fit every layer.

So drop the assumption that there must be one rate. Give each layer its own. Let layer ℓ have its own local rate λ^ℓ, and let a single global γ still ride on top so I keep one knob for the overall schedule (warmup, decay). The per-layer update direction is

Δw^ℓ = γ · λ^ℓ · ∇L(w^ℓ).

Now what should λ^ℓ be? The thing I want is the same safety condition holding *for every layer at once*: the step should be a fixed, safe fraction of that layer's own weight norm, no matter how big or small that layer's gradient happens to be. The quantity that varied across layers was precisely ‖w^ℓ‖/‖∇L(w^ℓ)‖, so the obvious thing to try is to set the local rate to it:

λ^ℓ = η · ‖w^ℓ‖ / ‖∇L(w^ℓ)‖,

with a small coefficient η < 1. Let me check what that does to the update magnitude:

‖Δw^ℓ‖ = γ · η · ‖w^ℓ‖ / ‖∇L(w^ℓ)‖ · ‖∇L(w^ℓ)‖ = γ · η · ‖w^ℓ‖.

The gradient norm cancels. The size of the step is γ·η times the weight norm — a fixed fraction of how big the layer already is — and crucially it's the *same* fraction for every layer regardless of whether that layer's gradient is huge or vanishing. Let me confirm that on the same two-layer example, with η = 0.1 and γ = 1 so it's apples-to-apples with the 0.1 target above:

- layer 1: trust ratio η·‖w₁‖/‖g₁‖ = 0.1·0.5/10 = 0.005; relative step ‖γ·r·g₁‖/‖w₁‖ = 0.100
- layer 2: trust ratio 0.1·50/0.01 = 500; relative step ‖γ·r·g₂‖/‖w₂‖ = 0.100

Both layers land on exactly 0.1 of their own norm, where the global rate gave 0.1 and 1e-6. The worst layer no longer pins the others, and a layer with a tiny gradient still gets a step proportional to its weights while a layer with a huge gradient doesn't overshoot. η is just "how much do I trust this layer to change in one update" — the fraction of its own norm it's allowed to move. Keep it well below 1 so each step stays inside the safe region; in practice something tiny like 1e-3, because γ is large in the large-batch regime and the product γ·η is what actually sets the relative step.

Let me sanity-check this against the per-coordinate adaptive optimizers, since they also "scale the rate by gradient statistics" and I should know why I'm not just reinventing them. Adam and RMSProp divide each individual weight's step by a running norm of that weight's own gradient history. Two things are different. The granularity: theirs is per *weight*, mine is per *layer* — a single norm over the whole layer is a far more stable aggregate than per-coordinate statistics, and the divergence I'm fighting is a layer-level phenomenon (a whole layer's update outrunning a whole layer's weights), so layer granularity is the right unit and it's less noisy. The control variable: their step is governed by the gradient's magnitude alone, so they don't control the quantity that actually causes the blow-up, which is the step *as a fraction of the weights* — note the cancellation above only happens because ‖w^ℓ‖ is in the numerator, which is exactly what gradient-only methods lack. So it's a genuinely different control variable, not Adam with a coarser grain. (It is, if I want a frame for it, a block-diagonal rescaling of the gradient with one block per layer — the simplest possible version of that, where each block is a scalar.)

Now fold in the pieces a real optimizer needs: momentum and weight decay. Weight decay adds β·w to the gradient, so the thing I'm actually stepping along is d = g + β·w, not g. If I left only ‖g‖ in the denominator, the trust ratio would be computed for the wrong direction, and a large decay term could make the real step bigger than the trust calculation says it is. So the denominator should reflect the decayed direction. The exact thing is ‖g + β·w‖, but I'm wary of that — when g and β·w nearly cancel, ‖g + β·w‖ collapses toward zero and the ratio η·‖w‖/‖g+βw‖ spikes. Let me see how bad that actually is before I reject it. Take ‖w‖ = 10, g = (−4.9, 0.2), β = 0.5 so β·w = (5, 0) and d = g + β·w = (0.1, 0.2), a near-cancellation; η = 0.1, γ = 1.

- exact denominator ‖d‖ = 0.224 → trust ratio = 0.1·10/0.224 = 4.47
- triangle-bound denominator ‖g‖ + β‖w‖ = 4.90 + 5.0 = 9.90 → trust ratio = 0.1·10/9.90 = 0.101

The exact-norm denominator gives a trust ratio of 4.47 where the bounded one gives 0.101 — a factor of ~44 swing, triggered by a tiny perturbation of g that happens to align against β·w. That instability is exactly what I don't want in a thing whose whole job is to keep step sizes controlled. The triangle inequality ‖g + β·w‖ ≤ ‖g‖ + β·‖w‖ buys stability:

λ^ℓ = η · ‖w^ℓ‖ / ( ‖∇L(w^ℓ)‖ + β·‖w^ℓ‖ ).

Let me check that this denominator actually bounds the decayed step rather than just being smoother. With it, the decayed relative step is ‖γ·λ^ℓ·(g+βw)‖/‖w‖ = γ·η·‖g+βw‖/(‖g‖+β‖w‖) ≤ γ·η, since the numerator is bounded by the denominator. On the cancellation example: ‖γ·r·d‖/‖w‖ = 1·0.101·0.224/10 = 0.0023, comfortably under γ·η = 0.1. So the decayed step is controlled by a guarantee, not by wishful cancellation, and as a bonus the extra β·‖w‖ in the denominator gently shrinks the local rate for layers with large weights, which is sensible. And momentum just accumulates the already-scaled, already-decayed direction into a velocity, the way heavy-ball always does.

Putting the whole step together for layer ℓ, with a global polynomial-decay schedule γ_t = γ_0·(1 − t/T)^2 sitting on top, and using the PyTorch-style convention where momentum stores the scaled direction and the global rate is applied in the final parameter update:

g_t^ℓ = ∇L(w_t^ℓ)
γ_t   = γ_0·(1 − t/T)^2
r_t^ℓ = η · ‖w_t^ℓ‖ / ( ‖g_t^ℓ‖ + β·‖w_t^ℓ‖ + ε )      — the trust ratio
u_t^ℓ = r_t^ℓ · ( g_t^ℓ + β·w_t^ℓ )                     — scaled, decayed direction
v_{t+1}^ℓ = m·v_t^ℓ + u_t^ℓ                              — momentum buffer, with dampening 0
w_{t+1}^ℓ = w_t^ℓ − γ_t·v_{t+1}^ℓ                        — global schedule applied once

One thing to be careful about at the very start: a tensor with zero weight norm has no meaningful "fraction of its own norm," and a tensor with zero raw gradient norm would make the no-decay ratio blow up. Let me trace the guard on both to be sure it does the harmless thing: for ‖w‖ = 5, ‖g‖ = 2, η = 1e-3, β = 1e-4 the ratio is 1e-3·5/(2 + 5e-4) ≈ 0.0025, a normal small multiplier; for ‖g‖ = 0 the formula would give a finite-but-meaningless number once ε is added, and for ‖w‖ = 0 it would give 0, so in both degenerate cases I override the multiplier to 1.0 — i.e. fall back to plain SGD for that tensor this step rather than adapt on a norm that carries no information. And do I still want warmup? Yes, but as a complement, not a crutch. The per-layer rate fixes the cross-layer mismatch — the structural reason the global rate diverged. Warmup now only has to smooth the first few steps where the ratios are still settling, so I can ask less of a single scalar ramp. The two are doing different jobs.

Now to real code, mirroring how this drops into a standard momentum-SGD optimizer. Per parameter tensor (each tensor is a "layer" for the purpose of the norm). Compute the two norms, form the trust ratio η·‖w‖/(‖g‖+β‖w‖+ε), clamp the degenerate cases to 1, fold weight decay into the gradient and multiply by the ratio — then it's ordinary momentum SGD with the *global* rate. In this implementation, groups with no weight decay stay on plain SGD unless `always_adapt=True`; that keeps the code faithful to the common parameter-group split while still allowing pure no-decay adaptation when I ask for it.

```python
import torch
from torch.optim.optimizer import Optimizer

class Lars(Optimizer):
    """Layer-wise Adaptive Rate Scaling on top of momentum SGD.

    Per parameter tensor: scale the step by a trust ratio
    eta * ||w|| / (||g|| + wd*||w||) so each layer moves a fixed fraction of
    its own weight norm, then apply plain momentum SGD with the GLOBAL lr.
    """
    def __init__(self, params, lr=1.0, momentum=0, dampening=0,
                 weight_decay=0.0, nesterov=False,
                 trust_coeff=0.001, eps=1e-8, trust_clip=False, always_adapt=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        trust_coeff=trust_coeff, eps=eps,
                        trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            wd        = group['weight_decay']     # beta
            momentum  = group['momentum']         # m
            dampening = group['dampening']
            nesterov  = group['nesterov']
            tc        = group['trust_coeff']      # eta
            eps       = group['eps']
            lr        = group['lr']               # global gamma_t (set by scheduler)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # --- LARS: per-layer trust ratio ---------------------------
                if wd != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    # eta * ||w|| / (||g|| + beta*||w||)  -> step is ~ a fixed
                    # fraction of ||w||, independent of the gradient magnitude
                    trust_ratio = tc * w_norm / (g_norm + wd * w_norm + eps)
                    # degenerate layers (zero weight or zero grad): don't adapt
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group['trust_clip']:          # optional: local lr never exceeds global lr
                        trust_ratio = torch.clamp(trust_ratio / lr, max=1.0)
                    grad.add_(p, alpha=wd)           # fold weight decay: g + beta*w
                    grad.mul_(trust_ratio)           # scale by the trust ratio

                # --- plain momentum SGD with the GLOBAL learning rate -------
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    grad = grad.add(buf, alpha=momentum) if nesterov else buf

                p.add_(grad, alpha=-lr)
        return loss
```

The causal chain, start to end: I scale up the batch to use more hardware, which forces fewer-but-bigger steps, so I raise the learning rate (linearly, with warmup) — and past a point training diverges. The divergence isn't a generalization gap (the train–test gap doesn't widen); it's under-optimization, because a single global rate is pinned by the one layer whose ‖w‖/‖∇L‖ is smallest while every other layer is starved — and I checked that on two layers whose ratios differ by 1e5, the global rate gives relative steps of 0.1 and 1e-6. So I give each layer its own rate set to η·‖w‖/‖∇L‖, which I verified makes both of those layers' no-decay step the same fraction γ·η·‖w‖ of their own weight norm; add the conservative weight-decay denominator (which I checked bounds the decayed step at γ·η and avoids the ~44× swing the exact-norm denominator suffers under cancellation), wrap the scaled direction in momentum and a global decay schedule, and the worst layer no longer caps the whole network. That gives me a concrete optimizer to try at the large batches that broke the global-rate recipe.
