OK, let me think this through from scratch. The thing I actually want is small networks. These models have millions of weights and most of them, the folklore says, are dead weight — I should be able to throw out the vast majority and still match accuracy. So the object is clear: minimize the loss subject to a hard budget on the number of non-zero weights, ‖w‖_0 ≤ κ. The L0 constraint is combinatorial, no way I'm solving that directly for millions of weights, so the whole game is going to come down to a *criterion*: a per-connection score s_j that tells me which connections to keep, and a *time* in the pipeline at which I compute it.

What does everyone do today? Two scores dominate. Magnitude: s_j = |w_j|, throw away small weights. And curvature, Optimal-Brain-Damage style: expand the loss around a trained minimum, δL = (∂L/∂w)ᵀδw + ½ δwᵀ H δw + …, kill the first-order term because at a minimum the gradient is zero, and the saliency falls out as s_j = w_j² H_jj / 2 (or w_j²/(2[H⁻¹]_jj) if you keep the full Hessian). The Hessian one is more principled but H is huge, dense, not positive-definite, intractable. Forget it for big nets.

But here's the thing that's been bugging me, and it's true of *both* of those: they depend on the **scale of the weights**. s_j = |w_j| is literally the weight. The OBD score is w_j² times curvature — also weight-scaled. And that's not a cosmetic detail. It means the score is only meaningful once the weights have *organized* — once the network is trained. On a fresh random initialization, |w_j| is just noise from my init distribution; it says nothing about whether connection j matters for the task. That's *why* pruning has always been a post-processing step glued onto a trained model, and why it drags this expensive prune–retrain cycle behind it: train fully, score, cut, retrain to heal the damage, repeat. The retraining loop is the cost, and the schedule for it is all heuristics. Worse, because the scores ride on weight scale, they're sensitive to learning rate and to architecture — a batchnorm or a different normalization rescales weights and silently shifts who gets pruned.

So let me ask the greedy question directly. Could I prune *once*, at initialization, before any training, and then just train the survivors? That would dodge the whole prune/retrain cycle, and if the survivors form a sparse mask I can even use sparse kernels. But to do that I cannot use any weight-scale-dependent criterion, because at init the weights carry no task information. I need a criterion that asks a different question: not "is this weight big?" but "does this *connection* matter to the loss for this task?" — and asks it in a way that's robust to the (arbitrary) scale of the random weights.

Let me try to make "does the connection matter" precise. The trouble with the weights w is that they conflate two things: the *strength* of a connection and *whether the connection is even there*. I want to isolate the second. So let me literally introduce a second variable per connection — an indicator c_j ∈ {0,1} that says connection j is present (1) or removed (0) — and write the network's weights as the Hadamard product c ⊙ w. Now the problem becomes

```
min_{c,w}  L(c ⊙ w; D),   c ∈ {0,1}^m,  ‖c‖_0 ≤ κ.
```

I've doubled the variables, and this nested combinatorial thing over c is *harder* to solve, not easier — so I'm not going to solve it. But that's not why I introduced c. I introduced it because now I have a clean handle: c_j is the dial that turns connection j on and off, *decoupled* from its weight w_j. The importance of connection j is just how much the loss reacts when I flip its dial.

So measure that. The exact effect of removing connection j is

```
ΔL_j = L(1 ⊙ w; D) − L((1 − e_j) ⊙ w; D),
```

the loss with everything on, minus the loss with only connection j knocked out (e_j the j-th unit vector). This is intrinsically about the *connection*, and I can evaluate it at any w, including random init. But it's hopeless to compute: m+1 forward passes over the data, m in the millions. And because c is binary, L isn't even differentiable in c, so I can't just call a gradient.

The escape: relax c off the integers. If I let c_j vary continuously, then for an infinitesimal nudge from the all-on network, ΔL_j is approximated by the derivative of L with respect to c_j, evaluated at the all-ones point:

```
ΔL_j ≈ g_j(w; D) = ∂L(c ⊙ w; D)/∂c_j  |_{c=1}
     = lim_{δ→0} [ L(c ⊙ w; D) − L((c − δ e_j) ⊙ w; D) ] / δ  |_{c=1}.
```

This is the rate of change of L as c_j slides from 1 toward 1−δ. And here's the payoff I was hoping for: I do not need m+1 forward passes. ∂L/∂c_j for *all* j at once is one forward–backward pass through autodiff — c enters as a multiplicative gate, so the backward pass hands me every g_j simultaneously. m+1 forward passes collapse to one.

Before I trust this, let me stare at what g_j actually is and make sure I have the algebra right, because the whole method rests on it. With the effective gated weight u_j = c_j w_j, the chain rule gives

```
∂L/∂c_j = (∂L/∂u_j)(∂u_j/∂c_j) = (∂L/∂u_j) · w_j.
```

That's a clean prediction, and it's the kind of thing I tend to get a sign or a factor wrong on, so let me actually check it numerically rather than nod at it. Take a tiny net: two inputs, two hidden units with a ReLU, two outputs, one labelled example. Put gates c = 1 on the first weight matrix W1, run autodiff to get ∂L/∂c directly. Separately, perturb the effective weight additively (effective weight = W1 + δ, gates fixed at 1) and get ∂L/∂u that way. The claim is ∂L/∂c = (∂L/∂u) · W1 elementwise. Running it on x = (1, −2), W1 = [[0.5, −0.3],[0.2, 0.7]], W2 = [[0.1,0.4],[−0.6,0.2]], label 1:

```
∂L/∂c (autodiff) = [[0.2392, 0.2871], [0, 0]]
∂L/∂u (autodiff) = [[0.4785, −0.9569], [0, 0]]
(∂L/∂u) ⊙ W1     = [[0.4785·0.5, −0.9569·(−0.3)], …] = [[0.2392, 0.2871], [0, 0]]
```

The two match to the last digit — max absolute difference 0.0. Good, the identity is real and the factor is exactly w_j, not w_j² or anything else. So the connection-sensitivity gradient is the ordinary effective-weight gradient *multiplied by the current weight*.

Now contrast that with the bare gradient w.r.t. the weight, ∂L/∂w_j, which is what older sensitivity criteria (Mozer–Smolensky, Karnin) effectively used. At c = 1, ∂L/∂u_j *is* ∂L/∂w_j, so in my example the bare-gradient criterion would rank the two live connections by |0.4785| vs |0.9569| — and the second connection wins decisively. But my gate score ranks them by |0.2392| vs |0.2871| — much closer, and the gap has shrunk because the bigger raw gradient sat on a smaller weight (−0.3 vs 0.5). So this is not a relabelling of the same ranking; multiplying by w_j genuinely reorders things relative to |∂L/∂w|. That reordering is exactly what I want: the bare ∂L/∂w_j measures the change from an *additive* perturbation δ applied uniformly, in isolation from how big w_j already is, whereas ∂L/∂c_j is the derivative with respect to a multiplicative gate — a small change in c_j changes the effective weight by δ·w_j. That is closer to the pruning operation I care about, because removing a connection is shrinking that connection's own contribution toward zero, not adding the same small number to every weight. This does not make the initialized weights irrelevant. It still uses w through the forward pass and through the factor above, which is why initialization has to be chosen carefully. What it removes is the need for w to be pretrained: the score is the loss response to a present connection, not a trained magnitude used as a proxy.

I also notice something in that same numerical run that I should not gloss over: the second hidden unit's gates came out with gradient exactly 0. Let me see why, because a zero score means "prune this for free" and I want to know whether that's signal or an artifact. The pre-activations are x @ W1ᵀ = (1·0.5 + (−2)·(−0.3), 1·0.2 + (−2)·0.7) = (1.1, −1.2). The second unit is at −1.2, so the ReLU zeroes it and its local gradient is dead — every connection feeding that unit gets score 0, regardless of how important it might become after training. That's a concrete, computed instance of a real failure mode: if the activations are saturated or dead at the scoring moment, ∂L/∂u_j is uninformative and the score collapses to zero for the wrong reason. So the criterion is only as good as the state of the network when I evaluate it, which tells me the initialization is load-bearing, not a detail I can wave away.

One more thing to be careful about: sign. If g_j is large *negative*, connection j still has a large effect on the loss — moving its gate changes the objective a lot, so it is important; I must not discard it just because the sign is negative. (In the example above ∂L/∂u had a −0.9569 entry, which is large in magnitude; throwing it away on account of its sign would be a mistake.) What I care about is the *magnitude* of the effect, regardless of direction. Since the computational object is the gate vector, the local saliency is |g ⊙ c|. At the scoring moment c = 1, so this is exactly |∂L/∂c|. To make the criterion comparable across the network — and not tied to the absolute scale of the loss or the batch — normalize by the total:

```
a_j = |g_j(w; D)c_j|,   c = 1,
s_j = a_j / Σ_k a_k = |g_j(w; D)| / Σ_k |g_k(w; D)|.
```

That gives a connection-sensitivity score. The normalization is a positive rescaling shared by every entry, so it does not change the ordering; given the budget κ I keep the top-κ entries by |∂L/∂c|:

```
c_j = 1[ s_j − s̃_κ ≥ 0 ],
```

where s̃_κ is the κ-th largest entry of s (ties broken arbitrarily to land exactly κ). Sort descending, threshold, done.

Now wait. I claimed this works at initialization, but the score still touches w through that ∂L/∂c_j = (∂L/∂u_j)·w_j factor and through the forward pass — the dead-unit example already showed me the score reacts to the network's state. So the choice of initial w is not irrelevant; it needs to be sensible. The saturation mode I just saw numerically is the first concern: if the initial weights are too large, the post-nonlinearity activations saturate or die, the local gradients go flat, and ∂L/∂u_j is uninformative. The weights have to sit in a reasonable range; this is exactly what standard initialization schemes are for, so the pruning criterion can be evaluated at such an initialization. And for robustness across architectures, fixed-variance random weights are not enough. With a fixed variance everywhere, the variance of the signal — and hence of the gradients — can drift from layer to layer, as the initialization literature warns. Then my saliency would pick up a spurious dependence on depth and width: layers get systematically smaller or larger g_j because of variance drift, not because their connections matter less or more. Since s_j is literally proportional to ∂L/∂u_j, anything that scales the gradient by layer scales my score by layer. The fix is to use a variance-scaling init, Glorot- or He-style, so the signal variance is held roughly constant through the layers; then g_j is much less likely to be a proxy for architectural scale. So variance scaling isn't decoration here — it is what keeps connection sensitivity from tracking architecture-scale drift.

And the data: s_j depends on D and on L, i.e. on the *task*. That is a feature — it means the retained connections are the ones important *for this task*. I can compute the gradient on the whole training set, but that is unnecessary and expensive; one mini-batch of a reasonable size gives the single-shot ranking. If memory is tight or I want a steadier estimate I can accumulate |g| over several batches or keep an exponential moving average.

So the whole method assembles itself. Variance-scaling init. One mini-batch. One forward–backward pass with a multiplicative gate c=1 to get every ∂L/∂c_j. Form |g ⊙ c|, which here is |g|, and keep the top-κ entries. Then throw the gates away as variables, fix the binary mask c, and train the sparse network the standard way — w* = argmin L(c ⊙ w). No pretraining, no prune–retrain cycle, no extra hyperparameters beyond κ.

Let me write it. The clean way to compute g_j in an autodiff framework is exactly the gate trick I used in the numerical check: attach a multiplicative mask parameter initialized to ones in front of each weight, freeze the weights while scoring, run one batch, backprop, and read off the *gradient of the mask* — that's ∂L/∂c_j by construction, which is what I confirmed equals (∂L/∂u_j)·w_j above. Ranking |mask.grad * mask| is the same as ranking |∂L/∂c| at c=1, and after pruning the fixed mask keeps removed connections at zero during normal training.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# A Conv2d / Linear that multiplies its weight by a learnable connectivity gate c.
class GatedConv2d(nn.Conv2d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_mask = nn.Parameter(torch.ones_like(self.weight))  # c = 1
        self.weight.requires_grad = False                              # freeze w; we want ∂L/∂c
    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class GatedLinear(nn.Linear):
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

def compute_keep_mask(net, batch, keep_fraction):
    x, y = batch
    net.zero_grad(set_to_none=True)
    F.cross_entropy(net(x), y).backward()        # one forward-backward; gates are at c = 1

    # g_j = dL/dc_j is the gradient on each gate; score |g_j * c_j|, with c_j = 1 here.
    gated = [m for m in net.modules() if isinstance(m, (GatedConv2d, GatedLinear))]
    scores = [torch.abs(m.weight_mask.grad * m.weight_mask).detach() for m in gated]
    flat_scores = torch.cat([s.flatten() for s in scores])
    flat_scores = flat_scores / flat_scores.sum().clamp_min(torch.finfo(flat_scores.dtype).tiny)

    k = int(keep_fraction * flat_scores.numel())
    k = max(1, min(k, flat_scores.numel()))
    keep = torch.zeros_like(flat_scores)
    keep[torch.topk(flat_scores, k, sorted=False).indices] = 1.0

    masks, offset = [], 0
    for score in scores:
        n = score.numel()
        masks.append(keep[offset:offset + n].view_as(score))
        offset += n
    return gated, masks

def apply_and_train(gated, masks, net, loader, epochs, lr):
    for m, msk in zip(gated, masks):
        with torch.no_grad():
            m.weight_mask.copy_(msk)        # fix the binary mask c
        m.weight_mask.requires_grad = False
        m.weight.requires_grad = True       # now train the surviving weights
    net.zero_grad(set_to_none=True)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()                      # pruned connections stay at zero: w * mask, mask fixed 0
```

So the causal chain, start to finish: I want a hard sparsity budget, which forces me to pick connections by a *criterion*; the standard criteria are weight-scale-dependent, which chains them to a trained net and the expensive prune–retrain loop; to break that chain I decouple "is the connection present" (c) from "how strong is it" (w), and measure importance as the loss's sensitivity to the gate, ∂L/∂c_j = (∂L/∂u_j)·w_j — the derivative of a multiplicative removal dial, which I checked numerically reduces exactly to the effective-weight gradient times the weight, evaluated at c=1; I take |g ⊙ c|, which is |∂L/∂c| at scoring time, keep the top-κ entries, and — using a variance-scaling init so the score does not track architecture-scale drift or die on saturated units — prune once, before training, in a single forward–backward pass, then train the sparse survivors normally.
