Ten point eight minutes now, 4578 steps, val_loss 3.2762, about 142 ms a step. The untie experiment landed
the way I predicted on the load-bearing factor: the step count fell 5100 → 4578 (~10% fewer), which is the
specialization buying steps, and the val_loss stayed comfortably under the bar at 3.2762 (a touch less margin
than the modern-arch record's 3.2741, but well clear). The step_avg also dropped, 178.89 → 141.87 ms, more
than the untie itself could explain — untying adds no forward matmul — so I read that as intervening schedule
and systems tuning rather than the untie, and I won't over-credit the untie for it. The point that matters is
the diagnosis it confirmed: the embedding and head were the limiting factor, and the wins came from cleaning
up how information enters and leaves the residual stream. The architecture is good, the optimizers are good.
So the next place to look is the same place — the flow of information *through* the residual stream over the
twelve blocks, and the stability of the logits at the very end. None of what I'm about to reach for is a big
new mechanism; they're a handful of small architectural shortcuts, and I think they compound. And unlike the
modern-arch rung — where the risk was that stacking many changes muddied attribution — these can be bundled
*safely*, because I'll construct each one to be a pure pass-through at initialization, so it can only add
capacity the model chooses to use and never degrades the known-good starting point. Let me derive each from a
pain point I can actually name.

Start with depth. A residual transformer is supposed to let every layer read the running sum, but there's a
well-known failure mode: as I stack blocks, attention *concentrates* — later layers pour their attention
mass onto a few positions and effectively stop attending broadly, and in doing so they lose access to the
value information the early layers computed. Think about what that means for the value stream specifically.
The value vectors at block 1 carry a clean, lightly-processed view of each token; by block 8 the values are
heavily transformed and the early signal is hard to recover from the residual stream alone, because it's been
overwritten by eight rounds of additive updates and the concentrated attention can't gather it back. This is
exactly the diagnosis in value residual learning, Zhou et al. 2024 ("Value Residual Learning For Alleviating
Attention Concentration In Transformers"). Their fix is simple and cheap: give every later layer a direct
line back to the *first* block's value vector. The first block computes v1 and exposes it; every subsequent
attention layer mixes its own freshly-projected v with v1 before doing attention. In the original form the
mix is a fixed half-and-half, `v = 0.5*v + 0.5*v1`. So each head, no matter how deep, always retains half of
the pristine early-layer value, and the attention output can't drift entirely away from the input's value
content. It's the value-stream analogue of a residual connection, but across depth rather than within a
block.

Let me make "attention concentrates with depth" concrete, because it's the load-bearing premise. A head's
attention distribution over its window is a softmax; its entropy measures how broadly it reads. A maximally
broad head over a window of W keys has entropy ln W (uniform); a fully concentrated head that dumps all its
mass on one key has entropy 0. The empirical finding behind value residual is that entropy *falls* with depth
— early layers read broadly (entropy near ln W), late layers collapse toward a few keys (entropy toward 0).
The consequence for the value stream is direct: a head that attends to only one or two positions computes its
output as essentially the value of those one or two tokens, so whatever rich, broad value information the
early layers gathered is not being read by the deep layers at all — they've narrowed their aperture. Giving
the deep layer a direct additive line to v1 sidesteps the narrowed aperture entirely: even a fully
concentrated deep head still gets a (1−λ)·(its own value) + λ·v1 blend, so v1 arrives regardless of where the
softmax points. And why v1 — the *first* block's value — rather than some middle layer's? Because v1 is the
least-processed thing that is still *in value space*: it's the embedding passed through exactly one value
projection, so it's the cleanest per-token value available, one transformation deep, before concentration has
had any chance to act. A middle-layer value would already be partly narrowed; the embedding itself isn't a
value at all (wrong space, no c_v applied). v1 is the sweet spot — clean, but genuinely a value vector the
attention can use.

One check on the value-residual init, to be sure I begin from the known-good point. The first block *defines*
v1: in its forward `v1 is None`, so I set `v1 = v`, and then the mix is `(1−λ)·v + λ·v = v` for any λ — the
first block is unaffected by the mechanism at every step, not just at init, which is correct since there's no
earlier value to reach back to. For the deeper blocks at init λ = 0.5, so the mix is exactly the original
half-and-half `0.5·v + 0.5·v1`, the configuration Zhou et al. reported to help — so I'm not gambling on the
learnable version; I start at a value known to work and let the per-layer lambdas depart from it.

A fixed 0.5 bothers me a little, though, and it's worth pushing on. Why should every layer want exactly half
of v1? The first decoder layer is right next to v1 and probably wants very little of it — its own value is
already almost as clean; a deep layer that's badly concentrated probably wants a lot. A single constant can't
express that depth profile. The cheapest possible way to let the model decide is to make the mixing
coefficient a *learnable scalar*, one per attention layer, and let the optimizer find the right value per
depth. So instead of a constant I write `self.lamb = nn.Parameter(torch.tensor(0.5))`, initialized at 0.5 so
I start exactly at the original half-and-half behavior — the known-good configuration is nested inside the
new one at init — and the mix becomes `v = (1 - self.lamb)*v + self.lamb*v1`. It's a single extra scalar per
layer — negligible parameters — and these scalars are tiny 1-D tensors, so they can't go to Muon (which
orthogonalizes 2-D matrices and is meaningless on a scalar); I hand them to Adam, at a fairly aggressive
lr=0.02 since they're scalars that should move fast and have no risk of the anisotropy Muon exists to fix. The
model can now turn the early-value injection up where attention is concentrated and down where it isn't.

That value-residual idea is really an instance of a more general principle: deep blocks benefit from a
*direct path back to something early and clean*. v1 is one such early signal. But there's an even earlier
one — the input embedding itself, the post-lookup, post-norm activation x0 that seeds the whole residual
stream (the very quantity I just pinned to unit RMS in the last rung). By block 10 the residual stream is a
deep transformation of x0, and the only way a block can still "see" the raw input is through whatever
survived ten rounds of additive updates — the same dilution problem the value residual attacks, one level up.
Why not give every block a direct line to x0 as well, exactly as I'm now giving them a direct line to v1? The
same shape of fix: before each block does its attention and MLP, let it form a learnable blend of the current
residual x and the original embedding x0. I add two scalars per block, `self.lambdas =
nn.Parameter(torch.tensor([1., 0.]))`, and I want to check the init is a genuine pass-through: with lambdas =
(1, 0), the blend `x = self.lambdas[0]*x + self.lambdas[1]*x0` is `1·x + 0·x0 = x`, exactly the current
residual, so at step zero this block is untouched and the model begins from precisely the 4578-step
configuration I already have. Training then dials in how much raw input each block wants to re-inject, from a
starting point that costs nothing. This is the embed shortcut: input → every block, learnable, pass-through
at init. Why two independent scalars (1, 0) rather than a single convex mixing weight, or an init of (1, 1)?
A single convex weight would force x and x0 to trade off — more x0 means proportionally less x — but I don't
want a trade-off, I want the block to be able to *add* raw input on top of the full residual if that helps,
so two free scalars let it scale each independently. And (1, 1) would inject a full copy of x0 into every
block at step zero, changing the known-good starting point and doubling the input's contribution before any
learning — exactly the kind of unrequested perturbation I'm trying to avoid. (1, 0) is the unique init that
is both a pure pass-through and leaves both scalars free to move.

Both value-path shortcuts share one plumbing change: the block forward now has to receive x0 and v1 from the
top-level module, so its signature becomes `forward(self, x, x0, v1)` and the model threads the post-embed x0
and the first block's v1 down through every block. That's a small structural cost — the blocks are no longer
purely sequential in a single tensor — but it's the same wiring for both shortcuts, so I pay it once. The
first block produces v1 (as shown, `v1 = v` there) and every later block consumes it; x0 is computed once
after the post-embed norm and broadcast to all. Nothing about this changes the matmul structure, so it's the
handful of pointwise mixes I flagged as the only per-step-time cost.

Two of these are about *getting clean early signal deep into the net*. Now two more about *not destabilizing
the run*. First, the optimizer side. Muon's momentum is set to 0.95, which is great once the network has
found a sensible region — momentum averages the orthogonalized updates and smooths the trajectory. But at
the very start, when the weights are near their zero-init and the loss is dropping fastest, a heavy momentum
buffer is averaging over updates that are themselves changing direction rapidly; the buffer lags the true
gradient and the early steps are noisier than they need to be. Let me quantify "heavy": an exponential
moving average with momentum m has an effective averaging window of about 1/(1−m) updates, so momentum 0.95
averages over ~1/0.05 = 20 updates, while 0.85 averages over ~1/0.15 ≈ 6.7. Early in training the "right"
gradient direction is turning over far faster than 20 steps, so a 20-step memory is stale — it's pointing
where the loss surface *was*, not where it is. Starting at 0.85 (a ~7-step memory) tracks the fresh gradient
roughly three times more responsively, then ramping to 0.95 restores the smoothing once the trajectory
settles. Concretely, the buffer is buf ← m·buf + g, so a gradient from k steps ago carries weight ~mᵏ in the
current buffer: at m = 0.95 a gradient 10 steps old still weighs 0.95¹⁰ ≈ 0.60, and 20 steps old 0.95²⁰ ≈
0.36 — a long tail of stale directions. At m = 0.85 the same 10-step-old gradient weighs 0.85¹⁰ ≈ 0.20 and
the 20-step-old one 0.85²⁰ ≈ 0.039 — the tail is cut off fast, so the buffer is dominated by recent gradients.
That's the quantitative content of "gentler early": in the first 500 steps the update leans on the last ~7
gradients, which in the fast-turning early landscape actually point where the loss is now. So a momentum *warmup*: start Muon at 0.85 and ramp linearly up to 0.95 over the first 500 steps. In
the loop: `frac = min(step/500, 1)` and `optimizer3.param_groups[0]['momentum'] = (1 - frac)*0.85 +
frac*0.95`. It costs nothing — it's a schedule on a hyperparameter I already have, and 500 steps out of ~4500
is the first ~11% of the run, exactly the fast-changing early phase.

Second, the logits. I zero-init the head, so the run *starts* from uniform max-entropy logits, which is
clean — but nothing stops a single logit from blowing up *later* in training. As the head sharpens, one
coordinate of the 50304-way output can run away, the softmax saturates, the cross-entropy gradient through
that coordinate goes tiny or spikes, and I get a noisy, slightly unstable loss curve right where I want it
smooth. Gemma 2 (Team et al. 2024) handles exactly this with a tanh logit softcap: pass the logits through a
smooth saturating nonlinearity so they're bounded but the function stays differentiable and monotone. I'll
cap at 30: `logits = 30 * torch.tanh(logits / 30)`. Let me verify this is near-inert for normal logits and
only bites on runaways, because a cap that distorts ordinary training would raise the loss. Expand for small
argument: tanh(z) ≈ z − z³/3, so 30·tanh(logit/30) ≈ logit − logit³/(3·30²) = logit − logit³/2700. At a
typical logit of 5, the correction is 125/2700 ≈ 0.046 — under 1%, invisible. At logit 10 it's 1000/2700 ≈
0.37, a few percent. Only as the logit climbs toward 30 does it bend hard: 30·tanh(1) = 30·0.7616 ≈ 22.8, so
a raw logit of 30 is squashed to 22.8, and the function asymptotes to ±30 no matter how large the input.
So it's the identity where training lives and a wall only at the runaway edge — a free stabilizer with a
negligible footprint (tanh costs the same regardless of the constant).

Before I bundle, let me consider whether these are the *right* tools versus the obvious alternatives. For
the logit runaway I could use plain gradient clipping or weight decay on the head instead of a softcap — but
clipping needs a per-run threshold I'd have to tune and it acts on the whole gradient rather than the specific
pathology, and weight decay pulls the head toward zero indiscriminately, fighting the very sharpening I want.
The tanh cap is targeted (it acts only on large logits), smooth (no gradient discontinuity to destabilize the
optimizer), and self-tuning in the sense that the constant 30 is a soft bound, not a hard clip point. For the
attention-concentration problem I could add more heads or a concentration penalty, but those cost compute or
a tuned regularizer weight, whereas the value residual is a pointwise mix that's nearly free and addresses the
mechanism directly (restore access to clean values) rather than penalizing the symptom. So the four I'm
bundling are each the cheap, targeted, pass-through-at-init choice.

So I'm bundling four shortcuts, each derived from the same instinct — give the network direct paths to clean
signal and don't let any one quantity run away. Value residual with a learnable per-layer lambda (mix in the
first block's v1, decide how much per depth). Embed shortcut (re-inject the original x0 into every block via
two learnable scalars). Momentum warmup for Muon (0.85→0.95 over 500 steps, gentler early updates). And the
tanh logit softcap at 30 (bound the output so the softmax can't saturate). Each is a couple of lines; each
starts from a pass-through or known-good initialization so I'm only ever *adding* capacity the model can
choose to ignore — which is exactly why bundling them is low-risk here, unlike the modern-arch rung.

If the mechanism is right, the falsifiable signature: the two value-path changes should buy real *steps* —
better information flow to deep layers is the kind of thing that cuts the iteration count meaningfully — so I
expect the step count down from 4578 toward ~3200, with val_loss holding under 3.28. The per-step time should
be roughly flat: the mixes and the x0 threading add a few pointwise operations (a slight bump, not a matmul),
and the warmup and softcap are free. If instead the step count barely moves, that would say the deep layers
weren't actually starved of early signal and the value paths are inert — but the fixed-0.5 value residual is
already reported to help, and the learnable lambda only relaxes it, so I'd be surprised. The learned lambdas
themselves are a bonus diagnostic: if they settle well away from 0.5, that's direct evidence the layers
wanted a depth-varying amount of v1 that the constant couldn't give.

```python
# value residual + learnable lambda, inside the attention block
def __init__(self, ...):
    ...
    self.lamb = nn.Parameter(torch.tensor(0.5))   # value residual lambda, optimized by Adam(lr=0.02)

def forward(self, x, v1=None):
    ...
    v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
    if v1 is None:
        v1 = v                                            # first block defines the shared value
    v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)   # mix with first-block value

# embed shortcut, in the Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    def forward(self, x, x0, v1):
        x = self.lambdas[0] * x + self.lambdas[1] * x0   # embed shortcut
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# momentum warmup for Muon (optimizer3), in the training loop
frac = min(step/500, 1)
optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

# tanh logit softcap, before computing the loss
logits = 30 * torch.tanh(logits / 30)
```

The chain: as depth grows attention concentrates and late layers lose the early value signal, so I mix each
layer's value with the first block's v1 (value residual, Zhou et al. 2024) and let a learnable per-layer
lambda (init 0.5 = the original half-and-half, so pass-through) decide how much per depth; generalizing "give
deep blocks a clean early signal" I also re-inject the original embedding x0 into every block through two
learnable scalars (init (1,0), verified to be exactly `x` at step zero); I ramp Muon's momentum from 0.85 to
0.95 over 500 steps so the early buffer averages ~7 updates instead of ~20 and tracks the fast-changing
gradient; and I cap the logits with `30*tanh(logits/30)` (Gemma 2's softcap), which the small-argument
expansion shows is within ~1% of the identity for normal logits and a wall only near ±30. Four small
shortcuts, each a pass-through at init so bundling is safe, compounding into fewer and cleaner steps to 3.28
at roughly flat per-step time.
