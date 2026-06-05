Let me start from the thing that's bugging me. Magnitude pruning is the standard move: keep the weights with the biggest absolute value, throw the rest away, and it works beautifully for models trained from scratch. But everyone's workflow now is transfer learning — take a pretrained BERT, fine-tune it on a small end task — and in that regime magnitude pruning gets noticeably worse, especially when I push to high sparsity. I want to understand *why*, because the fix should fall out of the diagnosis.

So why would magnitude work from-scratch but not when fine-tuning? When I train from scratch, the final value of a weight is determined entirely by the end-task data — the optimization carved it. So "this weight has large magnitude" really does mean "the end task made it important." The criterion and the objective are aligned. But when I fine-tune a pretrained model, the weights start at their pretrained values and barely move — fine-tuning nudges them, it doesn't relearn them. There's an empirical observation that says exactly this: fine-tuned weights stay close in absolute value to where pretraining left them. So which weights are large is decided mostly by *pretraining*, not by my end task. If I prune by magnitude, I'm pruning according to a generic pretraining signal. I can practically predict, before I even start fine-tuning, which weights magnitude pruning will keep — the ones that were big at pretraining will stay big and survive; the ones that were small will stay small and get cut. The pruning decision is blind to the end task. That's the failure: in transfer learning, magnitude pruning can't *learn what to prune* from the fine-tuning step.

So I need a criterion that the fine-tuning process actually shapes. Magnitude is the *value* of the weight — a zeroth-order quantity. What's the fine-tuning process producing that magnitude ignores? The *change*. Fine-tuning moves weights, even if only a little, and the *direction* of that motion is end-task information. So instead of asking "is this weight far from zero?" I should ask "is this weight *moving away* from zero during fine-tuning?" A weight drifting away from zero is one the end task is actively reinforcing; a weight drifting toward zero is one the task is shutting off — and that should be pruned, *even if it's currently large*. Symmetrically, a currently-small weight that's heading away from zero should be kept. That's the conceptual jump: move the criterion from the value of the weight (0th order) to the motion of the weight (1st order). Both large and small weights become prunable, depending on which way they're moving.

How do I turn "moving away from zero" into something I can compute and learn? Let me set up the score-based framing that's standard now. Give each weight matrix $\mathbf{W}\in\mathbb{R}^{n\times n}$ a parallel matrix of importance scores $\mathbf{S}$, build a binary mask $\mathbf{M}$ from $\mathbf{S}$, and run inference as $\mathbf{a}=(\mathbf{W}\odot\mathbf{M})\mathbf{x}$. Magnitude pruning is the case $S_{i,j}=|W_{i,j}|$ with $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$, the top-$v\%$ kept. The new idea is to *learn* $\mathbf{S}$ during fine-tuning rather than read it off the weights, and to design its learning so that it accumulates "movement away from zero."

The natural thing is to learn both $\mathbf{W}$ and $\mathbf{S}$ together by gradient descent, with $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$. Forward pass for output $i$: $a_i=\sum_k W_{i,k}M_{i,k}x_k$. But there's an immediate wall. The mask comes from $\mathrm{Top}_v$, which is a hard selection — its gradient with respect to $\mathbf{S}$ is zero almost everywhere (and undefined at the ties). So $\partial \mathcal{L}/\partial \mathbf{S}$ through the mask is zero; the scores would never update. The mask blocks the gradient.

The straight-through estimator gets me around this: when a forward op has no usable gradient, pretend in the backward pass that it was the identity and let the gradient flow straight through it to the thing underneath. So in the forward pass I apply $\mathrm{Top}_v$ to get the hard mask, but in the backward pass I ignore $\mathrm{Top}_v$ and pass the gradient straight to $\mathbf{S}$. Concretely, treating $a_i=\sum_k W_{i,k}M_{i,k}x_k$ and pretending $M_{i,j}$ depends on $S_{i,j}$ as the identity for backprop,

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂a_i)(∂a_i/∂S_{i,j}) = (∂𝓛/∂a_i) · W_{i,j} x_j.

So the score gets a gradient even for weights that are currently *masked out* in the forward pass — the score keeps accumulating a signal whether or not the weight is active. That's exactly what I want: a masked weight isn't frozen out of the competition; its score keeps moving and it can re-enter.

Now let me check that this $\mathbf{S}$ actually means "movement away from zero," because that was the whole point. Look at the ordinary weight gradient in this masked layer: $\partial\mathcal{L}/\partial W_{i,j} = (\partial\mathcal{L}/\partial a_i)\,M_{i,j}\,x_j$. Compare to the score gradient I just derived. Dropping the binary $M_{i,j}$ factor for the comparison (it just gates), the score gradient is $(\partial\mathcal{L}/\partial a_i)\,W_{i,j}\,x_j$, which is exactly the weight gradient with $W_{i,j}$ in place of $M_{i,j}$. So

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂W_{i,j}) · W_{i,j}.

Gradient descent does $S_{i,j} \leftarrow S_{i,j} - \alpha_S\,\partial\mathcal{L}/\partial S_{i,j}$, so $S_{i,j}$ *increases* exactly when $\partial\mathcal{L}/\partial S_{i,j}<0$, i.e. when $(\partial\mathcal{L}/\partial W_{i,j})\,W_{i,j}<0$. Two cases:

  (a) $\partial\mathcal{L}/\partial W_{i,j}<0$ and $W_{i,j}>0$, or
  (b) $\partial\mathcal{L}/\partial W_{i,j}>0$ and $W_{i,j}<0$.

In case (a): the weight gradient is negative, so gradient descent ($W \leftarrow W - \alpha_W\,\partial\mathcal{L}/\partial W$) pushes $W_{i,j}$ *up*, and it's already positive — so it's growing more positive, moving away from 0. In case (b): the weight gradient is positive, so descent pushes $W_{i,j}$ *down*, and it's already negative — growing more negative, again moving away from 0. So in both cases where the score increases, the weight is moving away from zero. And conversely $S_{i,j}$ decreases exactly when the weight is shrinking toward zero. The score literally is a "movement away from zero" detector. It checks out.

Even better, let me unroll the score update over training. With $S^{(0)}=0$ and the straight-through gradient, after $T$ steps,

  S_{i,j}^{(T)} = -α_S Σ_{t<T} (∂𝓛/∂W_{i,j})^{(t)} · W_{i,j}^{(t)}.

So the score is a running *accumulator of movement*: summing $-(\partial\mathcal{L}/\partial W)\,W$ over training, which is positive on every step the weight moves away from 0 and negative when it heads back. The keep/prune decision is the accumulated end-task movement — not the static pretrained value. This is genuinely first-order: it uses the gradient (direction of motion), not just the value. And note I get the importance scores for free as a by-product of standard fine-tuning — no Hessian, no second derivatives like Optimal Brain Surgeon needs.

Let me predict what the pruned models should look like, to make sure the story is coherent. Magnitude pruning removes everything near zero, so the surviving weight distribution should be two clumps, a gap around 0 with nothing in it. Movement pruning selects on motion, not value, so a weight of *any* current magnitude can survive or be cut depending on how it moved — the surviving distribution should be smooth, spread across the whole range, only missing the values right at 0 (since a weight pinned at 0 hasn't moved). And there should be no clean relationship between a weight's final value and its score: both small and large weights can score high. The one structural thing I'd expect is that *high scores go with non-zero weights* (a high score means it moved away from 0, so it ended up away from 0) — a "v-shape" when I plot score against weight. Good, that's a falsifiable picture.

Now the mask. I've been using hard $\mathrm{Top}_v$: keep the top $v\%$ of scores. That gives "hard movement pruning." But $\mathrm{Top}_v$ enforces a *fixed* sparsity and selects locally per matrix (or globally over all matrices if I rank scores across the whole net). I can also relax it: instead of a top-$k$ cut, threshold the scores, $\mathbf{M}=(\mathbf{S}>\tau)$ for a single global $\tau$. The straight-through trick works the same way for thresholding. But a bare threshold has no mechanism to *reach* a target sparsity — nothing pushes the scores down, so the mask density is whatever it is. So I add a regularizer that gently pushes scores down over time:

  R(𝐒) = λ_mvp Σ_{i,j} σ(S_{i,j}),

a sum of sigmoids of the scores, with coefficient $\lambda_{\text{mvp}}$ controlling the penalty strength and therefore the eventual sparsity. The full objective for "soft movement pruning" is $\mathcal{L} + \lambda_{\text{mvp}} R(\mathbf{S})$. The sigmoid keeps the penalty bounded and smooth; I tried a plain $\sum|S_{i,j}|$ instead, but it was harder to tune for similar results. Soft movement with a global threshold has the nice property that sparsity emerges non-uniformly across the network on its own — the regularizer lets different layers settle at different densities — rather than being forced equal per matrix.

There's a connection worth noticing to $L_0$ regularization, because it's also a first-order learned-mask method and I want to know whether I'm just reinventing it. $L_0$ makes the mask stochastic via a hard-concrete distribution: sample $u\sim\mathcal{U}(0,1)$, set $\overline{S}_{i,j}=\sigma((\log u-\log(1-u)+S_{i,j})/b)$, stretch $Z_{i,j}=(r-l)\overline{S}_{i,j}+l$, clamp $M_{i,j}=\min(1,\mathrm{ReLU}(Z_{i,j}))$, so an expected-$L_0$ penalty $\sum_{i,j}\sigma(\log S_{i,j}-b\log(-l/r))$ is differentiable. Its score gradient works out to

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂a_i) W_{i,j} x_j · f(\overline{S}_{i,j}),   f = ((r-l)/b)\overline{S}(1-\overline{S})\,𝟙_{0≤Z≤1}.

That's *exactly my straight-through score gradient times an extra factor* $f(\overline{S}_{i,j})$ — the same $(\partial\mathcal{L}/\partial a_i)W_{i,j}x_j$ core, decorated by the hard-concrete's local density. So movement pruning is the same first-order signal as $L_0$, but stripped of the stochastic reparameterization — deterministic, just straight-through. Simpler, and I don't have to tune the concrete temperature and stretch.

Now I should prove that learning the scores this way doesn't break training — that when a swap happens (a previously-pruned connection becomes more important than a kept one and replaces it), the loss actually goes *down*, not up. Take the cleanest case: $\mathrm{TopK}$ with $k=1$, so exactly one connection is active. At step $t$ the sole survivor is $(i,j)$, meaning $S^{(t)}_{u,v}\le S^{(t)}_{i,j}$ for all $u,v$. Suppose at $t+1$ the swap puts $(k,l)$ on top: $S^{(t+1)}_{u,v}\le S^{(t+1)}_{k,l}$ for all $u,v$. Apply both inequalities to the *same* pair: from the first, $S^{(t)}_{k,l}\le S^{(t)}_{i,j}$; from the second, $S^{(t+1)}_{i,j}\le S^{(t+1)}_{k,l}$. Subtract:

  S^{(t+1)}_{k,l} - S^{(t)}_{k,l} ≥ S^{(t+1)}_{i,j} - S^{(t)}_{i,j}.

Substitute the score update $S^{(t+1)} - S^{(t)} = -\alpha_S(\partial\mathcal{L}/\partial a)\,W^{(t)}x$ on both sides:

  -α_S (∂𝓛/∂a_k) W^{(t)}_{k,l} x_l ≥ -α_S (∂𝓛/∂a_i) W^{(t)}_{i,j} x_j.   (★)

Now with only one connection live, the activation is just that connection: $a^{(t)}_i = W^{(t)}_{i,j}x_j$ at $t$, and $a^{(t+1)}_k = W^{(t+1)}_{k,l}x_l$ at $t+1$. Assume $\alpha_W$ is small, so the activations move only a little, and $\mathcal{L}$ is smooth — take its first-order Taylor expansion around $(a^{(t)}_i, a^{(t)}_k)$:

  𝓛(a^{(t+1)}_i, a^{(t+1)}_k) - 𝓛(a^{(t)}_i, a^{(t)}_k)
    ≈ (∂𝓛/∂a_k)(a^{(t+1)}_k - a^{(t)}_k) + (∂𝓛/∂a_i)(a^{(t+1)}_i - a^{(t)}_i).

Plug in $a^{(t+1)}_k = W^{(t+1)}_{k,l}x_l$ and $a^{(t)}_i = W^{(t)}_{i,j}x_j$. The terms $a^{(t)}_k$ and $a^{(t+1)}_i$ are the activations of the *inactive* connection at each step — but at $t$ connection $(k,l)$ is masked off so $a^{(t)}_k$ contributes 0, and at $t+1$ connection $(i,j)$ is masked off so $a^{(t+1)}_i$ contributes 0. So the difference becomes

  = (∂𝓛/∂a_k) W^{(t+1)}_{k,l} x_l - (∂𝓛/∂a_i) W^{(t)}_{i,j} x_j.

Add and subtract $(\partial\mathcal{L}/\partial a_k)W^{(t)}_{k,l}x_l$ to split it:

  = (∂𝓛/∂a_k)(W^{(t+1)}_{k,l} - W^{(t)}_{k,l}) x_l + [ (∂𝓛/∂a_k) W^{(t)}_{k,l} x_l - (∂𝓛/∂a_i) W^{(t)}_{i,j} x_j ].

The first term: $W^{(t+1)}_{k,l} - W^{(t)}_{k,l}$ is the weight update of the $(k,l)$ connection. But $(k,l)$ was masked at step $t$, so its weight gradient $\partial\mathcal{L}/\partial W_{k,l} = (\partial\mathcal{L}/\partial a_k)M^{(t)}_{k,l}x_l$ had $M^{(t)}_{k,l}=0$ — the masked weight didn't move from the loss gradient. So that update is null, and the first term vanishes. The second bracketed term is precisely the right-minus-left of inequality (★) rearranged: (★) says $-\alpha_S(\partial\mathcal{L}/\partial a_k)W^{(t)}_{k,l}x_l \ge -\alpha_S(\partial\mathcal{L}/\partial a_i)W^{(t)}_{i,j}x_j$, and dividing by $-\alpha_S<0$ flips it to $(\partial\mathcal{L}/\partial a_k)W^{(t)}_{k,l}x_l \le (\partial\mathcal{L}/\partial a_i)W^{(t)}_{i,j}x_j$, i.e. the bracket is $\le 0$. So the whole difference is $\le 0$:

  𝓛(a^{(t+1)}_i, a^{(t+1)}_k) ≤ 𝓛(a^{(t)}_i, a^{(t)}_k).

When a connection becomes more important than the active one and the swap happens, the loss decreases. The argument generalizes to a set of simultaneously swapping connections, and it doesn't depend on $\mathrm{TopK}$ specifically — the same inequalities hold for the threshold mask $\mathbf{M}=(\mathbf{S}\ge\tau)$, so soft movement converges too.

And this is where I see *why the sign / direction matters* and I can't just take $|S|$ as importance. Suppose I'd used the *absolute* value of the score as the importance proxy (as some gradient-based methods do). Consider the degenerate version of that: a "negative threshold" mask $\mathbf{M}=(\mathbf{S}<\tau)$ with $\tau<0$ — keep the *most negative* scores. Redo the swap inequalities with this flipped selection: at $t$, $S^{(t)}_{i,j}\le\tau\le S^{(t)}_{u,v}$; at $t+1$, $S^{(t+1)}_{k,l}\le\tau\le S^{(t+1)}_{u,v}$. The same two-inequality manipulation now gives $S^{(t+1)}_{k,l}-S^{(t)}_{k,l}\le S^{(t+1)}_{i,j}-S^{(t)}_{i,j}$, the *reverse* inequality, so (★) flips and the bracket becomes $\ge 0$, and the Taylor difference comes out $\ge 0$: the loss *increases* on a swap. So selecting on $|S|$ destroys the guarantee. The direction of movement is load-bearing — keeping weights whose score is *most positive* (moving away from 0) is what makes the loss go down; keeping by magnitude of score is provably the wrong thing. This is the precise reason I preserve the sign rather than take absolute value.

A couple of practical wrappers, all pre-existing pieces. I won't do a hard one-shot cut; I'll use automated gradual pruning — raise the sparsity $v$ over training on a cubic schedule while masked weights keep updating, so the model recovers from early masking. The schedule is $v^{(t)} = v_f + (v_i - v_f)(1 - \frac{t-t_i}{N\Delta t})^3$, and adding a few cool-down steps at the very end (holding $v_f$) helps at high sparsity. For BERT I freeze the embeddings and prune the transformer layers and the task head. And since distillation is orthogonal, I can add a knowledge-distillation loss from a fine-tuned BERT-base teacher — a convex combination of task loss and distillation loss — to push performance further; it doesn't change the relative ordering of the pruning methods, it lifts them all.

Now the code. The heart is a masked linear layer where the mask comes from the *scores* through a straight-through top-$v$ (or threshold) function, so scores get gradients even when masked.

```python
import torch, torch.nn as nn

class TopVSTE(torch.autograd.Function):
    # forward: hard top-v% mask from scores; backward: straight-through to scores.
    @staticmethod
    def forward(ctx, scores, keep_ratio):
        mask = torch.zeros_like(scores)
        k = int(keep_ratio * scores.numel())
        idx = scores.flatten().argsort(descending=True)[:k]   # top-v% by SCORE
        mask.flatten()[idx] = 1.0
        return mask
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None        # straight-through: ignore Top_v, pass grad to S

class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, tau):
        return (scores > tau).float()
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None         # straight-through to S

class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, keep_ratio=1.0, soft=False, tau=0.0):
        super().__init__(in_f, out_f)
        self.score = nn.Parameter(torch.zeros_like(self.weight))  # learned, init 0
        self.keep_ratio, self.soft, self.tau = keep_ratio, soft, tau

    def forward(self, x):
        if self.soft:
            M = ThresholdSTE.apply(self.score, self.tau)          # soft movement
        else:
            M = TopVSTE.apply(self.score, self.keep_ratio)        # hard movement
        return nn.functional.linear(x, self.weight * M, self.bias)
        # backprop gives dL/dW = (dL/da) M x  and, via STE,
        #               dL/dS = (dL/da) W x  -> S accumulates -(dL/dW) W = movement
```

Training loop: cubic sparsity schedule, scores and weights learned jointly, optional soft-movement regularizer and distillation.

```python
def sparsity(t, v_i, v_f, t_i, T, t_f, dt):           # cubic w/ cool-down
    if t < t_i:            return v_i
    if t >= T - t_f:       return v_f
    return v_f + (v_i - v_f) * (1 - (t - t_i - t_f) / ((T - t_i - t_f) * dt)) ** 3

def fine_prune(model, loader, opt, T, sched, lam_mvp=0.0, teacher=None, soft=False):
    for t, (x, y) in zip(range(T), loader):
        if not soft:
            set_keep_ratio(model, 1 - sparsity(t, **sched))       # MvP follows schedule
        out = model(x)
        loss = task_loss(out, y)
        if teacher is not None:                                   # distillation (orthogonal)
            loss = 0.5 * loss + 0.5 * distill_loss(out, teacher(x))
        if soft:                                                  # soft MvP regularizer
            loss = loss + lam_mvp * sum(torch.sigmoid(m.score).sum()
                                        for m in masked_layers(model))
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```

So the chain: in transfer learning the weights are inherited from pretraining and barely move, so magnitude (a 0th-order, value-based criterion) prunes by the pretrained values and can't learn from the end task. Switching to a 1st-order criterion — keep the weights that *move away from zero* during fine-tuning — fixes this, realized by learning a score $\mathbf{S}$ jointly with $\mathbf{W}$ through a hard $\mathrm{Top}_v$ (or threshold) mask whose gradient is supplied by the straight-through estimator. That gradient works out to $\partial\mathcal{L}/\partial S_{i,j}=(\partial\mathcal{L}/\partial W_{i,j})W_{i,j}$, so $\mathbf{S}$ accumulates $-\sum_t(\partial\mathcal{L}/\partial W)^{(t)}W^{(t)}$, exactly the movement signal; a Taylor argument shows swaps decrease the loss, and that guarantee provably fails if one selects by $|S|$, which is why the sign is preserved. A sigmoid regularizer turns the hard top-$v$ into a soft global-threshold variant, and a cubic sparsity schedule plus optional distillation wrap it for fine-tuning BERT at high sparsity.
