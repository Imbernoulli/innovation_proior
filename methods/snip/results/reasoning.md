OK, let me think this through from scratch. The thing I actually want is small networks. These models have millions of weights and most of them, the folklore says, are dead weight — I should be able to throw out the vast majority and still match accuracy. So the object is clear: minimize the loss subject to a hard budget on the number of non-zero weights, ‖w‖_0 ≤ κ. The L0 constraint is combinatorial, no way I'm solving that directly for millions of weights, so the whole game is going to come down to a *criterion*: a per-connection score s_j that tells me which connections to keep, and a *time* in the pipeline at which I compute it.

What does everyone do today? Two scores dominate. Magnitude: s_j = |w_j|, throw away small weights. And curvature, Optimal-Brain-Damage style: expand the loss around a trained minimum, δL = (∂L/∂w)ᵀδw + ½ δwᵀ H δw + …, kill the first-order term because at a minimum the gradient is zero, and the saliency falls out as s_j = w_j² H_jj / 2 (or w_j²/(2[H⁻¹]_jj) if you keep the full Hessian). The Hessian one is more principled but H is huge, dense, not positive-definite, intractable. Forget it for big nets.

But here's the thing that's been bugging me, and it's true of *both* of those: they depend on the **scale of the weights**. s_j = |w_j| is literally the weight. The OBD score is w_j² times curvature — also weight-scaled. And that's not a cosmetic detail. It means the score is only meaningful once the weights have *organized* — once the network is trained. On a fresh random initialization, |w_j| is just noise from my init distribution; it says nothing about whether connection j matters for the task. That's *why* pruning has always been a post-processing step glued onto a trained model, and why it drags this expensive prune–retrain cycle behind it: train fully, score, cut, retrain to heal the damage, repeat. The retraining loop is the cost, and the schedule for it is all heuristics. Worse, because the scores ride on weight scale, they're sensitive to learning rate and to architecture — a batchnorm or a different normalization rescales weights and silently shifts who gets pruned.

So let me ask the greedy question directly. Could I prune *once*, at initialization, before any training, and then just train the survivors? That's the dream — no prune/retrain cycles at all, and if the survivors form a sparse mask I can even use sparse kernels. But to do that I cannot use any weight-scale-dependent criterion, because at init the weights carry no task information. I need a criterion that asks a different question: not "is this weight big?" but "does this *connection* matter to the loss for this task?" — and asks it in a way that's robust to the (arbitrary) scale of the random weights.

Let me try to make "does the connection matter" precise. The trouble with the weights w is that they conflate two things: the *strength* of a connection and *whether the connection is even there*. I want to isolate the second. So let me literally introduce a second variable per connection — an indicator c_j ∈ {0,1} that says connection j is present (1) or removed (0) — and write the network's weights as the Hadamard product c ⊙ w. Now the problem becomes

```
min_{c,w}  L(c ⊙ w; D),   c ∈ {0,1}^m,  ‖c‖_0 ≤ κ.
```

I've doubled the variables, and this nested combinatorial thing over c is *harder* to solve, not easier — so I'm not going to solve it. But that's not why I introduced c. I introduced it because now I have a clean handle: c_j is the dial that turns connection j on and off, *decoupled* from its weight w_j. The importance of connection j is just how much the loss reacts when I flip its dial.

So measure that. The exact effect of removing connection j is

```
ΔL_j = L(1 ⊙ w; D) − L((1 − e_j) ⊙ w; D),
```

the loss with everything on, minus the loss with only connection j knocked out (e_j the j-th unit vector). This is exactly what I want — it's intrinsically about the *connection*, and I can evaluate it at any w, including random init. But it's hopeless to compute: m+1 forward passes over the data, m in the millions. And because c is binary, L isn't even differentiable in c, so I can't just call a gradient.

The escape: relax c off the integers. If I let c_j vary continuously, then for an infinitesimal nudge, ΔL_j is well approximated by the derivative of L with respect to c_j, evaluated at the all-ones point:

```
ΔL_j ≈ g_j(w; D) = ∂L(c ⊙ w; D)/∂c_j  |_{c=1}
     = lim_{δ→0} [ L(c ⊙ w; D) − L((c − δ e_j) ⊙ w; D) ] / δ  |_{c=1}.
```

This is just the rate of change of L as c_j slides from 1 toward 1−δ. And the beautiful part: I do *not* need m+1 forward passes. ∂L/∂c_j for *all* j at once is one forward–backward pass through autodiff — c enters as a multiplicative gate, so the backward pass hands me every g_j simultaneously. m+1 forward passes collapse to one.

Now stare at what g_j actually is, because this is the crux of why it works at init where magnitude fails. By the chain rule, with the gated pre-activation u_j = c_j w_j,

```
∂L/∂c_j = (∂L/∂u_j)(∂u_j/∂c_j) = (∂L/∂u_j) · w_j.
```

So the connection-sensitivity gradient is the ordinary weight-gradient direction *multiplied by the current weight*. Compare it to the bare gradient w.r.t. the weight, ∂L/∂w_j, which is what older sensitivity criteria (Mozer–Smolensky, Karnin) effectively used. The bare ∂L/∂w_j measures the change from an *additive* perturbation δ applied uniformly, in isolation from how big w_j already is — and at init the weights are not uniform in scale, so that's a biased ruler. But ∂L/∂c_j perturbs c_j multiplicatively, which means the effective perturbation to the weight is δ·w_j — it's *scaled in proportion to the current weight*. That's the trick. The sensitivity automatically normalizes by the local weight scale, so it doesn't require the weights to be optimal or pretrained. It can be read off a randomly initialized net. That's the whole point: it severs the dependence on weight scale that chained magnitude and Hessian criteria to a trained model.

One more thing to be careful about: sign. If g_j is large *negative*, connection j still has a large effect on the loss — pushing it would change things a lot, so it's important; I must not discard it just because the sign is negative. What I care about is the *magnitude* of the effect, regardless of direction. So take |g_j|. And to make the criterion comparable across the network — and not blow up with the absolute scale of the loss or the batch — normalize by the total:

```
s_j = |g_j(w; D)| / Σ_k |g_k(w; D)|.
```

That's connection sensitivity. Then, given the budget κ, keep the top-κ:

```
c_j = 1[ s_j − s̃_κ ≥ 0 ],
```

where s̃_κ is the κ-th largest entry of s (ties broken arbitrarily to land exactly κ). Sort descending, threshold, done.

Now — wait. I claimed this works "at any init," but let me not wave that away, because the score *does* still touch w through that ∂L/∂c_j = (∂L/∂u_j)·w_j factor and through the forward pass. So the choice of initial w isn't irrelevant; it just needs to be *sensible*. Two failure modes. First, if the initial weights are too large, the post-nonlinearity activations (sigmoid, say) saturate, the local gradients go flat, and ∂L/∂u_j is uninformative — garbage in. So the weights must sit in a reasonable range; this is exactly what standard initialization schemes are *for*, so I can prune at any of them. Second, and this is the subtle one for robustness across architectures: if I init with a *fixed* variance everywhere, the variance of the signal — and hence of the gradients — drifts from layer to layer (LeCun's old observation). Then my saliency picks up a spurious dependence on depth and width: deep layers get systematically smaller or larger g_j just because of variance drift, not because they matter less. The fix is to use a *variance-scaling* init (Glorot-style) so the signal variance is held roughly constant through the layers. Then g_j reflects task importance, not architectural accident, and the same criterion transfers unmodified across convolutional, residual, and recurrent nets. So variance-scaling init isn't decoration — it's what makes connection sensitivity architecture-robust.

And the data: s_j depends on D and on L, i.e. on the *task*. That's a feature — it means the retained connections are the ones important *for this task*. In practice I don't even need the whole training set; one mini-batch of a reasonable size gives an effective ranking. If memory is tight or I want a steadier estimate I can accumulate |g| over several batches or keep an exponential moving average — but a single batch is enough for single-shot pruning. (As a sanity check on whether the score is really task-driven: prune using batches from different classes/datasets and look at which input pixels' connections survive — they should track where the task's signal lives.)

So the whole method assembles itself. Variance-scaling init. One mini-batch. One forward–backward pass with a multiplicative gate c=1 to get all g_j. Normalize the magnitudes into s_j. Threshold at the top-κ to set the binary mask c. Then throw the gates away as variables, fix the mask, and train the sparse network the standard way — w* = argmin L(c ⊙ w). No pretraining, no prune–retrain cycle, no extra hyperparameters beyond κ.

Let me write it. The clean way to compute g_j in a framework is exactly the gate trick: attach a multiplicative mask parameter initialized to ones in front of each weight, freeze the weights (`requires_grad=False`), run one batch, backprop, and read off the *gradient of the mask* — that's ∂L/∂c_j by construction.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# A Conv2d / Linear that multiplies its weight by a learnable connectivity gate c.
class MaskedConv2d(nn.Conv2d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))  # c = 1
        self.weight.requires_grad = False                              # freeze w; we want ∂L/∂c
    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class MaskedLinear(nn.Linear):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))
        self.weight.requires_grad = False
    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def variance_scaling_init(net):
    # Glorot/Xavier so signal variance is preserved across layers -> architecture-robust saliency.
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

def snip(net, batch, keep_fraction):
    x, y = batch
    net.zero_grad()
    F.cross_entropy(net(x), y).backward()        # one forward-backward; gates are at c = 1

    # g_j = dL/dc_j is exactly the gradient on each mask; saliency is its magnitude.
    masked = [m for m in net.modules() if isinstance(m, (MaskedConv2d, MaskedLinear))]
    g_abs = [torch.abs(m.weight_mask.grad) for m in masked]
    all_scores = torch.cat([g.flatten() for g in g_abs])
    all_scores = all_scores / all_scores.sum()    # s_j = |g_j| / Σ_k |g_k|

    k = int(keep_fraction * all_scores.numel())
    threshold = torch.topk(all_scores, k, sorted=True).values[-1]   # s̃_κ
    masks = [((torch.abs(m.weight_mask.grad) / all_scores.sum().detach()) >= threshold).float()
             for m in masked]
    return masked, masks

def apply_and_train(masked, masks, net, loader, epochs, lr):
    for m, msk in zip(masked, masks):
        m.weight_mask.data = msk            # fix the binary mask c
        m.weight_mask.requires_grad = False
        m.weight.requires_grad = True       # now train the surviving weights
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()                      # pruned connections stay at zero: w * mask, mask fixed 0
```

So the causal chain, start to finish: I want a hard sparsity budget, which forces me to pick connections by a *criterion*; the standard criteria are weight-scale-dependent, which chains them to a trained net and the expensive prune–retrain loop; to break that chain I decouple "is the connection present" (c) from "how strong is it" (w), and measure importance as the loss's sensitivity to the gate, ∂L/∂c_j = (∂L/∂u_j)·w_j — a multiplicatively-scaled perturbation that self-normalizes by weight scale and so reads correctly at random init; I take its magnitude (sign-agnostic), normalize across the net, keep the top-κ, and — using a variance-scaling init so the score doesn't track architecture — prune once, before training, in a single forward–backward pass, then train the sparse survivors normally.
