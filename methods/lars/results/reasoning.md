OK, let me think about this from the actual constraint I'm under, which is wall-clock time. Training a big convolutional net on ImageNet is slow on one device, and the only knob that turns more hardware into less time without touching the model is the mini-batch. Data-parallel synchronous SGD: split the global batch of B examples across N workers, each worker computes the gradient on its B/N chunk, sum the partials, do one update. If I want each worker busy enough to be efficient, its chunk has to be sizeable, so the way I add workers to go faster is by pushing the global batch B up. More nodes, bigger batch, fewer-but-heavier steps. That's the whole appeal.

So the question is just: how big can B get before something breaks, and what breaks?

The obvious tension is that I'm fixing the number of epochs — I want the same amount of data seen, just faster. If I multiply B by k, then with the same epochs I do k times fewer weight updates. The optimizer is being asked to reach the same place in k-fold fewer steps. So each step has to cover more ground, which means a bigger learning rate. How much bigger?

Let me actually work out the natural scaling instead of guessing. Take two consecutive steps at batch B with rate λ:

w_{t+2} = w_t − λ·(1/B)( Σ_{i=1}^B ∇L(x_i, w_t) + Σ_{j=1}^B ∇L(x_j, w_{t+1}) ).

Now compare to a single step at batch 2B with some rate λ₂:

w_{t+1} = w_t − λ₂·(1/(2B)) Σ_{i=1}^{2B} ∇L(x_i, w_t).

These line up if λ₂ = 2λ — but only if I'm allowed to say ∇L(x_j, w_{t+1}) ≈ ∇L(x_j, w_t), i.e. the weights barely moved during the first sub-step so the second sub-step's gradient is evaluated at essentially the same point. So the linear rule — grow B by k, grow λ by k, leave momentum and weight decay alone — comes straight out of that approximation. And I can already see where it's going to fail: the approximation is exactly "the step is small." Once λ is large the step isn't small, w_{t+1} is far from w_t, and ∇L(x_j, w_{t+1}) is nothing like ∇L(x_j, w_t). The rule licenses a big rate and then the big rate invalidates the rule.

There's a more cautious alternative. The mini-batch gradient is an average of B samples, so its standard error shrinks like 1/√B; from a pure noise-reduction view I should only raise λ like √B, not B. That's gentler and it won't outrun its own justification as fast. But it's also leaving speed on the table early, and in practice the linear rule has been the one people get away with up to some moderate batch. So I'll start from linear scaling as the aggressive baseline and watch for the wall.

There's also a practical softener: the initial phase is where the big rate hurts most — the very first steps, before anything has settled, blow up. So ramp λ from a small safe value up to the target over the first few epochs, then hand off to the regular decay. That warmup, plus linear scaling, is the recipe to try.

Let me run it in my head on AlexNet/ImageNet. Baseline B=256, around 58%. Push to B=4K with a proportionally larger rate and a warmup... and it diverges once the rate goes past a threshold, even with the warmup. The best I can squeeze at B=4K is low-50s; at B=8K, mid-40s. So scaling effectively dies around B=2K. The recipe everyone quotes just doesn't carry this network to where I need it.

What's special about the networks where linear scaling reportedly does carry further — the ones that hit B=5K, 6K? They have Batch Normalization. So let me put BN into this network: replace the Local Response Normalization with BN. And it helps a lot — now I can use large rates even without warmup, the band of rates that train well gets much wider, and the B=8K accuracy gap collapses from enormous down to a couple of percent. BN is normalizing each layer's pre-activations over the batch, which keeps the scales sane and pushes the divergence ceiling up. Good. But there's still a residual gap at B=8K — a stubborn couple of percent — and I want to know what it is, because that's the thing standing between me and "no accuracy loss."

Two hypotheses. One: it's the generalization gap people talk about — large batches supposedly drift toward sharp minima that test worse. If that's it, I'd see the training loss look fine while the test loss lags, a widening train–test gap at large batch. So I measure the gap between training and testing loss at small batch versus B=8K. They look the same. No extra train–test divergence at large batch. So this residual is not a generalization problem — the model isn't overfitting a sharp basin, it's just not being optimized far enough. It's under-training. Which is actually good news: under-training is an optimization problem, and optimization I can attack.

So why can't I just crank the rate to optimize harder? Because the rate is exactly what's diverging. Let me stare at the update and ask what "diverging" means mechanically. Standard SGD uses one rate λ for the whole network: w_{t+1} = w_t − λ·∇L(w_t). For a given layer, the size of the step relative to the size of that layer's weights is ‖λ·∇L(w)‖ / ‖w‖. If that's around or above 1, the layer's parameters move as much as their own magnitude in one step — you've thrown the weights across the landscape, and that's the blow-up. So the safe condition, per layer, is that λ·‖∇L(w)‖ stays well below ‖w‖, i.e. λ stays well below ‖w‖ / ‖∇L(w)‖.

Now the obvious question: is that bound ‖w‖/‖∇L(w)‖ roughly the same across layers, or not? If it's uniform, one global λ is fine and I'm barking up the wrong tree. So measure it. Take the BN network after a single iteration and tabulate, layer by layer, ‖w‖ and ‖∇L(w)‖ and their ratio.

And the ratios are all over the place. An early convolutional layer's weights sit around a single-digit ratio — a few. A later fully-connected layer's weights sit in the thousands. Weights and biases within the same layer differ too. So the safe ceiling on λ for the early layer is a few, while the later layer would happily take a rate hundreds of times larger. Orders of magnitude apart.

That's the whole problem in one table. A single global λ has to satisfy the *tightest* constraint or it diverges — so it gets pinned by the worst layer (the small-ratio one). But that same λ, safe for the worst layer, is hundreds of times too small for the layers with big ratios, and those layers' weights then barely budge. One number cannot be simultaneously safe for the layer that needs a tiny rate and large enough for the layer that wants a huge one. So when I "raise the global rate to optimize harder," I'm either diverging the small-ratio layer or, if I back off to keep it safe, starving everyone else. That's the residual under-training: the global rate is throttled by the worst layer.

And now warmup makes mechanical sense — it's a crutch for exactly this. The earliest steps are the most sensitive: the ratios are changing quickly, the network has not settled, and the target large-batch rate is exactly the thing that can make a small-ratio layer jump farther than its own scale. A small starting rate is the one value that's safe for *all* layers at once, and the ramp delays the target rate until those layer-wise ratios are less volatile. Warmup is sneaking up on a moving, per-layer constraint with a single scalar. It's the right instinct applied through the wrong variable.

So drop the assumption that there must be one rate. Give each layer its own. Let layer ℓ have its own local rate λ^ℓ, and let a single global γ still ride on top so I keep one knob for the overall schedule (warmup, decay). The per-layer update direction is

Δw^ℓ = γ · λ^ℓ · ∇L(w^ℓ).

Now what should λ^ℓ be? I want the same safety condition to hold *for every layer at once*: the step should be a fixed, safe fraction of that layer's own weight norm, no matter how big or small that layer's gradient happens to be. The thing that varied across layers was precisely ‖w^ℓ‖/‖∇L(w^ℓ)‖, so set the local rate to it:

λ^ℓ = η · ‖w^ℓ‖ / ‖∇L(w^ℓ)‖,

with a small coefficient η < 1. Look at what the update magnitude becomes:

‖Δw^ℓ‖ = γ · η · ‖w^ℓ‖ / ‖∇L(w^ℓ)‖ · ‖∇L(w^ℓ)‖ = γ · η · ‖w^ℓ‖.

The gradient norm cancels. The size of the step is γ·η times the weight norm — a fixed fraction of how big the layer already is — and it's the *same* fraction for every layer regardless of whether that layer's gradient is huge or vanishing. So every layer moves the same relative amount, the worst layer no longer pins the others, and the update becomes much less sensitive to vanishing or exploding gradient scales: a layer with a tiny gradient still gets a step proportional to its weights, a layer with a huge gradient doesn't overshoot. η is just "how much do I trust this layer to change in one update" — the fraction of its own norm it's allowed to move. Keep it well below 1 so each step stays inside the safe region; in practice something tiny like 1e-3, because γ is large in the large-batch regime and the product γ·η is what actually sets the relative step.

Let me sanity-check this against the per-coordinate adaptive optimizers, since they also "scale the rate by gradient statistics" and I should know why I'm not just reinventing them. Adam and RMSProp divide each individual weight's step by a running norm of that weight's own gradient history. The granularity is different: theirs is per *weight*, mine is per *layer* — a single norm over the whole layer is a far more stable aggregate than per-coordinate statistics, and the divergence I'm fighting is a layer-level phenomenon (a whole layer's update outrunning a whole layer's weights), so layer granularity is the right unit and it's less noisy. The control variable is different too: their step is governed by the gradient's magnitude, not by the weight's magnitude, so they don't control the quantity that actually causes the blow-up, which is the step *as a fraction of the weights*. My ratio puts the weight norm in the numerator on purpose. So it's a genuinely different control variable, not Adam with a coarser grain. (It is, if I want a frame for it, a block-diagonal rescaling of the gradient with one block per layer — the simplest possible version of that, where each block is a scalar.)

Now fold in the pieces a real optimizer needs: momentum and weight decay. Weight decay adds β·w to the gradient, so the thing I'm actually stepping along is d = g + β·w, not g. If I used only ‖g‖ in the denominator, a large decay term could make the real step bigger than the trust calculation says it is. I could use ‖g + β·w‖ directly, but cancellation between g and β·w would make the scale jump around. The conservative, stable choice is the triangle-inequality bound ‖g + β·w‖ ≤ ‖g‖ + β·‖w‖:

λ^ℓ = η · ‖w^ℓ‖ / ( ‖∇L(w^ℓ)‖ + β·‖w^ℓ‖ ).

Then the decayed step is controlled by a bound, not by wishful cancellation:

‖γ · λ^ℓ · (g^ℓ + β·w^ℓ)‖ ≤ γ · η · ‖w^ℓ‖.

That keeps the trust-ratio interpretation intact once decay is present, and it also gently shrinks the local rate for layers with large weights, which is sensible. And momentum just accumulates the already-scaled, already-decayed direction into a velocity, the way heavy-ball always does.

Putting the whole step together for layer ℓ, with a global polynomial-decay schedule γ_t = γ_0·(1 − t/T)^2 sitting on top, and using the PyTorch-style convention where momentum stores the scaled direction and the global rate is applied in the final parameter update:

g_t^ℓ = ∇L(w_t^ℓ)
γ_t   = γ_0·(1 − t/T)^2
r_t^ℓ = η · ‖w_t^ℓ‖ / ( ‖g_t^ℓ‖ + β·‖w_t^ℓ‖ + ε )      — the trust ratio
u_t^ℓ = r_t^ℓ · ( g_t^ℓ + β·w_t^ℓ )                     — scaled, decayed direction
v_{t+1}^ℓ = m·v_t^ℓ + u_t^ℓ                              — momentum buffer, with dampening 0
w_{t+1}^ℓ = w_t^ℓ − γ_t·v_{t+1}^ℓ                        — global schedule applied once

One thing to be careful about at the very start: a tensor with zero weight norm has no meaningful "fraction of its own norm," and a tensor with zero raw gradient norm would make the no-decay ratio blow up. So guard both cases: if either norm is zero, don't adapt that tensor this step and treat the multiplier as 1. And do I still want warmup? Yes, but as a complement, not a crutch. The per-layer rate fixes the cross-layer mismatch — the structural reason the global rate diverged. Warmup now only has to smooth the first few steps where the ratios are still settling, so I can ask less of a single scalar ramp. The two are doing different jobs.

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

The causal chain, start to end: I scale up the batch to use more hardware, which forces fewer-but-bigger steps, so I raise the learning rate (linearly, with warmup) — and past a point training diverges. The divergence isn't a generalization gap (the train–test gap doesn't widen); it's under-optimization, because a single global rate is pinned by the one layer whose ‖w‖/‖∇L‖ is smallest while every other layer — whose ratios I measured to span orders of magnitude — is starved. So I give each layer its own rate set to η·‖w‖/‖∇L‖, which makes every layer's no-decay step a fixed fraction γ·η·‖w‖ of its own weight norm independent of the gradient scale; add the conservative weight-decay denominator and the decayed gradient, wrap the scaled direction in momentum and a global decay schedule, and the worst layer no longer caps the whole network. That gives me a concrete optimizer to try at the large batches that broke the global-rate recipe.
