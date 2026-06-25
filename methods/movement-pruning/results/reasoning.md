Let me start from the thing that's bugging me. Magnitude pruning is the standard move: keep the weights with the biggest absolute value, throw the rest away, and it works beautifully for models trained from scratch. But everyone's workflow now is transfer learning — take a pretrained BERT, fine-tune it on a small end task — and in that regime magnitude pruning gets noticeably worse, especially when I push to high sparsity. I want to understand *why*, because the fix should fall out of the diagnosis.

So why would magnitude work from-scratch but not when fine-tuning? When I train from scratch, the final value of a weight is determined entirely by the end-task data — the optimization carved it. So "this weight has large magnitude" really does mean "the end task made it important." The criterion and the objective are aligned. But when I fine-tune a pretrained model, the weights start at their pretrained values and barely move — fine-tuning nudges them, it doesn't relearn them. There's an empirical observation that says exactly this: fine-tuned weights stay close in absolute value to where pretraining left them. So which weights are large is decided mostly by *pretraining*, not by my end task. If I prune by magnitude, I'm pruning according to a generic pretraining signal. I can practically predict, before I even start fine-tuning, which weights magnitude pruning will keep — the ones that were big at pretraining will stay big and survive; the ones that were small will stay small and get cut. The pruning decision is blind to the end task. That's the failure: in transfer learning, magnitude pruning can't *learn what to prune* from the fine-tuning step.

So I need a criterion that the fine-tuning process actually shapes. Magnitude is the *value* of the weight — a zeroth-order quantity. What's the fine-tuning process producing that magnitude ignores? The *change*. Fine-tuning moves weights, even if only a little, and the *direction* of that motion is end-task information. So instead of asking "is this weight far from zero?" maybe I should ask "is this weight *moving away* from zero during fine-tuning?" A weight drifting away from zero is one the end task is actively reinforcing; a weight drifting toward zero is one the task is shutting off — and that should be prunable, *even if it's currently large*. Symmetrically, a currently-small weight that's heading away from zero should be keepable. That would move the criterion from the value of the weight (0th order) to the motion of the weight (1st order). Both large and small weights become prunable, depending on which way they're moving. That's the hypothesis; now I have to find out whether it can actually be computed and learned, and whether it does what I think.

How do I turn "moving away from zero" into something I can compute and learn? Let me set up the score-based framing that's standard now. Give each weight matrix $\mathbf{W}\in\mathbb{R}^{n\times n}$ a parallel matrix of importance scores $\mathbf{S}$, build a binary mask $\mathbf{M}$ from $\mathbf{S}$, and run inference as $\mathbf{a}=(\mathbf{W}\odot\mathbf{M})\mathbf{x}$. Magnitude pruning is the case $S_{i,j}=|W_{i,j}|$ with $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$, the top-$v\%$ kept. The idea I want to test is to *learn* $\mathbf{S}$ during fine-tuning rather than read it off the weights, and to design its learning so that it accumulates "movement away from zero."

The natural thing is to learn both $\mathbf{W}$ and $\mathbf{S}$ together by gradient descent, with $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$. Forward pass for output $i$: $a_i=\sum_k W_{i,k}M_{i,k}x_k$. But there's an immediate wall. The mask comes from $\mathrm{Top}_v$, which is a hard selection — its gradient with respect to $\mathbf{S}$ is zero almost everywhere (and undefined at the ties). So $\partial \mathcal{L}/\partial \mathbf{S}$ through the mask is zero; the scores would never update. The mask blocks the gradient.

The straight-through estimator gets me around this: when a forward op has no usable gradient, pretend in the backward pass that it was the identity and let the gradient flow straight through it to the thing underneath. So in the forward pass I apply $\mathrm{Top}_v$ to get the hard mask, but in the backward pass I ignore $\mathrm{Top}_v$ and pass the gradient straight to $\mathbf{S}$. Concretely, treating $a_i=\sum_k W_{i,k}M_{i,k}x_k$ and pretending $M_{i,j}$ depends on $S_{i,j}$ as the identity for backprop,

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂a_i)(∂a_i/∂S_{i,j}) = (∂𝓛/∂a_i) · W_{i,j} x_j.

The claim I most want to check is that this gradient reaches *masked-out* weights too — that a pruned weight isn't frozen out of the competition. That's not obvious to me from staring at the formula, so let me actually build the layer and read off the gradients. Take a two-input, one-output masked linear with $W=[2.0,\,-0.5]$, scores $S=[0.3,\,0.1]$, input $x=[1,1]$, keep the top-1 by score, and use the loss $\mathcal{L}=-a$ so that $\partial\mathcal{L}/\partial a=-1$ is something I can track by hand. Running it through the STE layer: the forward mask is $[1,0]$ — index 0 (score $0.3$) survives, index 1 (score $0.1$) is pruned. The weight gradient comes back $\partial\mathcal{L}/\partial W=[-1,\,0]$: the masked weight gets *no* weight gradient, as expected, because $\partial\mathcal{L}/\partial W_{i,j}=(\partial\mathcal{L}/\partial a_i)M_{i,j}x_j$ and $M=0$ there. But the score gradient comes back $\partial\mathcal{L}/\partial S=[-2.0,\,0.5]$, and the second entry is *non-zero* even though that weight was pruned. And $[-2.0,\,0.5]$ is exactly $(\partial\mathcal{L}/\partial a)\,W\,x=-1\cdot[2.0,-0.5]\cdot[1,1]$. So the formula is right and, crucially, the score of a masked weight keeps moving: $0.5$ of signal flowed into the pruned connection's score. A masked weight can re-enter, because its score never stopped accumulating.

Now I need to check the thing the whole idea rests on — that this $\mathbf{S}$ actually means "movement away from zero." Look at the ordinary weight gradient in this masked layer: $\partial\mathcal{L}/\partial W_{i,j} = (\partial\mathcal{L}/\partial a_i)\,M_{i,j}\,x_j$. Compare to the score gradient I just derived. The mask factor only says whether the weight itself is active; for the movement signal I want the same loss-direction factor even when the weight is currently masked. So, omitting the binary gate for this comparison, the score gradient should be the usual weight-gradient core multiplied by the current weight:

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂W_{i,j}) · W_{i,j},

where the displayed $\partial\mathcal{L}/\partial W_{i,j}$ is the ungated first-order factor $(\partial\mathcal{L}/\partial a_i)x_j$. I'd rather not trust that rearrangement blind, so I checked it numerically on a random three-weight layer: I computed $\partial\mathcal{L}/\partial S$ by autograd and, separately, $(\partial\mathcal{L}/\partial a)\,x$ (the ungated core) times $W$ entrywise, and the two vectors agreed to floating-point ($\approx 10^{-5}$). So the identity holds.

Now the sign argument. Gradient descent does $S_{i,j} \leftarrow S_{i,j} - \alpha_S\,\partial\mathcal{L}/\partial S_{i,j}$, so $S_{i,j}$ *increases* exactly when $\partial\mathcal{L}/\partial S_{i,j}<0$, i.e. when $(\partial\mathcal{L}/\partial W_{i,j})\,W_{i,j}<0$. Two cases:

  (a) $\partial\mathcal{L}/\partial W_{i,j}<0$ and $W_{i,j}>0$, or
  (b) $\partial\mathcal{L}/\partial W_{i,j}>0$ and $W_{i,j}<0$.

In case (a): the weight gradient is negative, so gradient descent ($W \leftarrow W - \alpha_W\,\partial\mathcal{L}/\partial W$) pushes $W_{i,j}$ *up*, and it's already positive — so it's growing more positive, moving away from 0. In case (b): the weight gradient is positive, so descent pushes $W_{i,j}$ *down*, and it's already negative — growing more negative, again moving away from 0. So in both cases where the score increases, the weight is moving away from zero. And conversely $S_{i,j}$ should decrease exactly when the weight is shrinking toward zero.

I wanted to watch this happen rather than just believe the case analysis, so I ran a four-weight masked layer for six joint SGD steps and, at each step, compared the sign of the score's movement against whether the weight's magnitude actually grew. For every *active* weight the two matched every step: $W=+1.5$ rising to $+1.7$ came with a rising score; $W=-1.0$ shrinking toward zero came with a falling score. But the trace also surfaced something I hadn't fully appreciated. At step 0, a weight that was *masked off* ($W=+0.4$) had a rising score yet its value didn't move at all ($+0.4\to+0.4$) — because, as the gradient read-off above showed, a masked weight gets no weight gradient, so it can't physically move that step. Its score rose anyway, on the strength of the ungated core, and by step 1 that score had climbed enough to bring the weight back into the active set, at which point it *did* start moving away from zero ($+0.4\to+0.54\to\dots$). So the honest statement is sharper than "S rises iff the weight is moving": the score tracks the *ungated* movement signal $-(\partial\mathcal{L}/\partial W_{\text{ungated}})W$, which is the rate at which the weight *would* move away from zero if it were live. For active weights that's the literal movement; for masked weights it's a prediction that drives re-entry. That's actually exactly the property I want — it's what lets a pruned-too-early weight come back.

This also tells me what $\mathbf{S}$ accumulates over training. With $S^{(0)}=0$ and the straight-through gradient, after $T$ steps,

  S_{i,j}^{(T)} = -α_S Σ_{t<T} (∂𝓛/∂W_{i,j})^{(t)} · W_{i,j}^{(t)},

a running *accumulator of movement*: summing $-(\partial\mathcal{L}/\partial W)\,W$ over training, positive on every step the weight moves away from 0 (in the ungated sense above) and negative when it heads back. The keep/prune decision is the accumulated end-task movement — not the static pretrained value. This is genuinely first-order: it uses the gradient (direction of motion), not just the value. And I get the importance scores essentially for free as a by-product of standard fine-tuning — no Hessian, no second derivatives like Optimal Brain Surgeon needs.

Before going further I want a falsifiable picture of what this should *do* to the weights, partly to make sure the story is coherent and partly so I have something to check later. Magnitude pruning removes everything near zero, so its surviving weight distribution should be two clumps with a gap around 0. Movement pruning selects on motion, not value, so a weight of *any* current magnitude can survive or be cut depending on how it moved — the surviving distribution should be smoother, spread across the range. And there shouldn't be a clean functional relationship between a weight's final value and its score. I ran a small synthetic stand-in for this — a 200-weight layer initialized to random "pretrained" values, then fine-tuned with the STE scores on a toy regression at 30% density. Among the kept weights, the smallest surviving $|W|$ was $0.02$ while some weights with $|W|>0.5$ were pruned — so yes, weights of any magnitude survived or died, unlike magnitude pruning. The correlation between score and $|W|$ among kept weights was positive but loose ($\approx 0.67$), consistent with a "v-shape" tendency (a high score means moved-away-from-0, hence usually ends up non-zero) rather than a clean function. This is a toy, not BERT, so I read it only as a sanity check that the predicted behavior is plausible; I'd want to confirm the actual score-vs-weight scatter on a real fine-tune.

Now the mask. I've been using hard $\mathrm{Top}_v$: keep the top $v\%$ of scores. That gives a hard variant. But $\mathrm{Top}_v$ enforces a *fixed* sparsity and selects locally per matrix (or globally over all matrices if I rank scores across the whole net). I can also relax it: instead of a top-$k$ cut, threshold the scores, $\mathbf{M}=(\mathbf{S}>\tau)$ for a single global $\tau$. The straight-through trick works the same way for thresholding. But a bare threshold has no mechanism to *reach* a target sparsity — nothing pushes the scores down, so the mask density is whatever it is. So I add a regularizer that gently pushes scores down over time:

  R(𝐒) = Σ_{i,j} σ(S_{i,j}),

a sum of sigmoids of the scores. The full objective for the soft variant is $\mathcal{L} + \lambda_{\text{mvp}} R(\mathbf{S})$, with $\lambda_{\text{mvp}}$ controlling the penalty strength and therefore the eventual sparsity. The sigmoid keeps the penalty bounded and smooth; a plain $\sum|S_{i,j}|$ penalty is the obvious alternative, but it is harder to tune for the same role. The soft global-threshold version has the nice property that sparsity can emerge non-uniformly across the network on its own — the regularizer lets different layers settle at different densities — rather than being forced equal per matrix.

There's a connection worth noticing to $L_0$ regularization, because it's also a first-order learned-mask method and I want to know whether I'm just reinventing it. $L_0$ makes the mask stochastic via a hard-concrete distribution: sample $u\sim\mathcal{U}(0,1)$, set $\overline{S}_{i,j}=\sigma((\log u-\log(1-u)+S_{i,j})/b)$, stretch $Z_{i,j}=(r-l)\overline{S}_{i,j}+l$, clamp $M_{i,j}=\min(1,\mathrm{ReLU}(Z_{i,j}))$, so an expected-$L_0$ penalty is differentiable. Its score gradient should work out to

  ∂𝓛/∂S_{i,j} = (∂𝓛/∂a_i) W_{i,j} x_j · f(\overline{S}_{i,j}),   f = ((r-l)/b)\overline{S}(1-\overline{S})\,𝟙_{0≤Z≤1}.

I derived $f$ by the chain rule through the clamp, stretch, and sigmoid, and then sanity-checked it numerically: for ten thousand random $(S,u)$ I compared the claimed $f$ against a finite-difference $\partial M/\partial S$, and they agreed to $\sim 10^{-10}$ (with $f=0$ exactly when $Z$ leaves $[0,1]$ and the clamp saturates, as expected). So $L_0$'s score gradient is *exactly my straight-through score gradient times an extra factor* $f(\overline{S}_{i,j})$ — the same $(\partial\mathcal{L}/\partial a_i)W_{i,j}x_j$ core, decorated by the hard-concrete's local density. The movement signal is the same first-order signal as $L_0$, but stripped of the stochastic reparameterization — deterministic, just straight-through. Simpler, and I don't have to tune the concrete temperature and stretch.

Now I should check that learning the scores this way doesn't break training — that when a swap happens (a previously-pruned connection becomes more important than a kept one and replaces it), the loss actually goes *down*, not up. Take the cleanest case: $\mathrm{TopK}$ with $k=1$, so exactly one connection is active. At step $t$ the sole survivor is $(i,j)$, meaning $S^{(t)}_{u,v}\le S^{(t)}_{i,j}$ for all $u,v$. Suppose at $t+1$ the swap puts $(k,l)$ on top: $S^{(t+1)}_{u,v}\le S^{(t+1)}_{k,l}$ for all $u,v$. Apply both inequalities to the *same* pair: from the first, $S^{(t)}_{k,l}\le S^{(t)}_{i,j}$; from the second, $S^{(t+1)}_{i,j}\le S^{(t+1)}_{k,l}$. Subtract:

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

The first term: $W^{(t+1)}_{k,l} - W^{(t)}_{k,l}$ is the weight update of the $(k,l)$ connection. But $(k,l)$ was masked at step $t$, so its weight gradient $\partial\mathcal{L}/\partial W_{k,l} = (\partial\mathcal{L}/\partial a_k)M^{(t)}_{k,l}x_l$ had $M^{(t)}_{k,l}=0$ — the masked weight didn't move from the loss gradient. So that update is null, and the first term vanishes. The second bracketed term is precisely the right-minus-left of inequality (★) rearranged: (★) says $-\alpha_S(\partial\mathcal{L}/\partial a_k)W^{(t)}_{k,l}x_l \ge -\alpha_S(\partial\mathcal{L}/\partial a_i)W^{(t)}_{i,j}x_j$, and dividing by $-\alpha_S<0$ flips it to $(\partial\mathcal{L}/\partial a_k)W^{(t)}_{k,l}x_l \le (\partial\mathcal{L}/\partial a_i)W^{(t)}_{i,j}x_j$, i.e. the bracket is $\le 0$. So the whole *first-order* difference is $\le 0$.

I wanted to be sure I hadn't fooled myself with the Taylor sleight-of-hand, so I checked it on random instances. First I confirmed the algebra: over thousands of random top-1 swaps, inequality (★) held at *every single* genuine swap (zero exceptions) — so the two-inequality manipulation is sound. But when I measured the *actual* loss before and right after the swap, I found a surprise: with a moderate score step ($\alpha_S=0.05$), the true loss decreased in about 1269 of 1397 swaps but *increased* in 128. That stopped me. Was the lemma wrong? I went back and computed the *first-order predicted* change — the very quantity the Taylor expansion bounds, $(\partial\mathcal{L}/\partial a_k)(a^{(t+1)}_k-a^{(t)}_k)+(\partial\mathcal{L}/\partial a_i)(a^{(t+1)}_i-a^{(t)}_i)$ — and *that* was $\le 0$ in 1397 of 1397, no exceptions. So the lemma is exactly right about what it actually proves: the *first-order* change is non-positive whenever a swap occurs. The 128 true-loss increases are the second-order curvature of the quadratic loss biting when the swap moves the activation by a non-infinitesimal amount — which is precisely why the derivation needed "$\alpha_W$ small" and a first-order expansion. So I'll state the guarantee honestly as a first-order / small-step result: locally, a swap does not increase the loss to first order. The argument generalizes to a set of simultaneously swapping connections, and the same inequalities hold for the threshold mask $\mathbf{M}=(\mathbf{S}\ge\tau)$, so the soft variant has the same first-order property.

And working through that swap argument shows me *why the sign / direction matters* and why I can't just take $|S|$ as importance. Suppose I'd used the *absolute* value of the score as the importance proxy (as some gradient-based methods do). Consider the degenerate version of that: a "negative threshold" mask $\mathbf{M}=(\mathbf{S}<\tau)$ with $\tau<0$ — keep the *most negative* scores. Redo the swap inequalities with this flipped selection: at $t$, $S^{(t)}_{i,j}\le\tau\le S^{(t)}_{u,v}$; at $t+1$, $S^{(t+1)}_{k,l}\le\tau\le S^{(t+1)}_{u,v}$. The same two-inequality manipulation now gives $S^{(t+1)}_{k,l}-S^{(t)}_{k,l}\le S^{(t+1)}_{i,j}-S^{(t)}_{i,j}$, the *reverse* inequality, so (★) flips and the bracket becomes $\ge 0$, and the first-order Taylor difference comes out $\ge 0$: a swap *increases* the loss to first order. So selecting on the most-negative score (the $|S|$-style choice) destroys the guarantee. I sanity-checked the direction by running both selection rules to convergence on a small toy: keeping the *most-positive* scores drove the loss reliably to ~0 across seeds, whereas keeping the *most-negative* scores was erratic — on one seed it stalled at loss $0.58$ after four thrashing swaps while positive-selection on the same seed reached $0$. The direction of movement is load-bearing — keeping weights whose score is *most positive* (moving away from 0) is what keeps swaps from hurting the loss; keeping by magnitude of score is the wrong thing. This is the precise reason I preserve the sign rather than take absolute value.

A couple of practical wrappers, all pre-existing pieces. I won't do a hard one-shot cut; I'll use automated gradual pruning — raise the sparsity $v$ over training on a cubic schedule while masked weights keep updating, so the model recovers from early masking. With warm-up and cool-down, the schedule is

  v^{(t)} =
    v_i,                                                    0 ≤ t < t_i
    v_f + (v_i - v_f)(1 - (t-t_i-t_f)/(NΔt))^3,             t_i ≤ t < T-t_f
    v_f,                                                    otherwise.

For BERT I freeze the embeddings and prune the transformer layers and the task head. And since distillation is orthogonal to the pruning rule, I can add a knowledge-distillation loss from a fine-tuned BERT-base teacher — a convex combination of task loss and distillation loss — without changing how the mask scores are defined.

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
        # dL/dS = (dL/da) W x, so S accumulates -(ungated dL/dW core) W.
```

Training loop: cubic sparsity schedule, scores and weights learned jointly, optional soft-movement regularizer and distillation.

```python
def sparsity(t, v_i, v_f, t_i, n_steps, delta_t, total_steps, t_f=0):
    if t < t_i:
        return v_i
    if t >= total_steps - t_f:
        return v_f
    progress = (t - t_i - t_f) / (n_steps * delta_t)
    return v_f + (v_i - v_f) * (1 - progress) ** 3

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

So the chain: in transfer learning the weights are inherited from pretraining and barely move, so magnitude (a 0th-order, value-based criterion) prunes by the pretrained values and can't learn from the end task. Switching to a 1st-order criterion — keep the weights that *move away from zero* during fine-tuning — addresses this, realized by learning a score $\mathbf{S}$ jointly with $\mathbf{W}$ through a hard $\mathrm{Top}_v$ (or threshold) mask whose gradient is supplied by the straight-through estimator. That gradient — which I verified on a small layer reaches masked weights and equals $(\partial\mathcal{L}/\partial a_i)W_{i,j}x_j$, the ungated weight-gradient core times $W_{i,j}$ — makes $\mathbf{S}$ accumulate $-\sum_t(\partial\mathcal{L}/\partial W_{\text{ungated}})^{(t)}W^{(t)}$, the away-from-zero signal (literal movement for live weights, a re-entry prediction for masked ones). A first-order Taylor argument shows swaps don't increase the loss to first order — a small-step guarantee I had to state carefully after seeing the true loss occasionally wiggle up by second-order curvature — and that guarantee reverses if one selects by $|S|$, which is why the sign is preserved. A sigmoid regularizer turns the hard top-$v$ into a soft global-threshold variant, and a cubic sparsity schedule plus optional distillation wrap it for fine-tuning BERT at high sparsity.
