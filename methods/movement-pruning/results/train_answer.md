We want to compress a fine-tuned BERT for deployment by unstructured weight pruning, removing as many individual weights as possible at a small accuracy cost, and the standard tool for this is magnitude pruning: keep the weights with the largest $|W|$, discard the rest, and gradually raise the sparsity during training. That recipe is excellent when a model is trained from scratch on the end task, because the optimization itself carved each final weight value out of the task data, so "large magnitude" genuinely means "important for this task" — the criterion and the objective are aligned. But everyone's workflow now is transfer learning: take a pretrained BERT-base and fine-tune it on a small annotated task, and there magnitude pruning degrades noticeably, especially when I push to high sparsity. The reason is structural. Fine-tuned weights stay close in absolute value to where pretraining left them; fine-tuning nudges weights, it does not relearn them. So which weights end up large is decided mostly by *pretraining*, a generic signal, not by my end task. I can practically predict, before fine-tuning even begins, which weights a magnitude criterion will keep — the ones that were big at pretraining survive, the ones that were small get cut — so the prune/keep decision is essentially fixed in advance and blind to the task. The learned-mask alternatives on the table either keep $\mathbf{W}$ frozen (Piggyback, hidden-in-a-random-network), which is the wrong setting for fine-tunable weights, or, like $L_0$ regularization, do adapt during fine-tuning but carry a stochastic hard-concrete reparameterization with a temperature and stretch parameters that is finicky to tune. The question is whether a simpler adaptive criterion that the fine-tuning step actually shapes will suffice.

I propose movement pruning. The conceptual move is to shift the importance criterion from zeroth order — the *value* of a weight — to first order — the *motion* of the weight during fine-tuning. Magnitude asks "is this weight far from zero?"; I instead ask "is this weight moving *away from* zero as I fine-tune?" A weight drifting away from zero is one the end task is actively reinforcing and should be kept even if it is currently small; a weight drifting toward zero is one the task is shutting off and should be pruned even if it is currently large. To compute and learn this, I use the standard score-based framing: give each weight matrix $\mathbf{W}\in\mathbb{R}^{n\times n}$ a parallel matrix of importance scores $\mathbf{S}$, build a binary mask $\mathbf{M}$ from $\mathbf{S}$, and run inference as $\mathbf{a}=(\mathbf{W}\odot\mathbf{M})\mathbf{x}$. Magnitude pruning is the degenerate case $S_{i,j}=|W_{i,j}|$ with $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$; the new idea is to *learn* $\mathbf{S}$ jointly with $\mathbf{W}$ by gradient descent rather than read it off the weights.

The immediate wall is that the mask $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$ is a hard selection whose gradient with respect to $\mathbf{S}$ is zero almost everywhere, so the scores would never update. The straight-through estimator resolves this: in the forward pass I apply $\mathrm{Top}_v$ to get the hard mask, but in the backward pass I pretend the mask was the identity and pass the gradient straight to $\mathbf{S}$. Treating the output as $a_i=\sum_k W_{i,k}M_{i,k}x_k$ and letting $M_{i,j}$ act as the identity in $S_{i,j}$ for backprop gives
$$\frac{\partial\mathcal{L}}{\partial S_{i,j}} = \frac{\partial\mathcal{L}}{\partial a_i}\,W_{i,j}\,x_j = \frac{\partial\mathcal{L}}{\partial W_{i,j}}\,W_{i,j},$$
where the last equality omits the binary gate that sits inside the ordinary weight gradient $\partial\mathcal{L}/\partial W_{i,j}=(\partial\mathcal{L}/\partial a_i)M_{i,j}x_j$. This is exactly what I need on two counts. First, because the straight-through gradient keeps the ungated first-order factor $(\partial\mathcal{L}/\partial a_i)x_j$, the score receives a signal even for weights that are currently *masked out* — a pruned connection is not frozen out of the competition; its score keeps moving and it can re-enter. Second, the score gradient is the ungated weight-gradient core multiplied by the current weight value, which is precisely a movement detector. Gradient descent does $S_{i,j}\leftarrow S_{i,j}-\alpha_S\,\partial\mathcal{L}/\partial S_{i,j}$, so $S_{i,j}$ increases exactly when $(\partial\mathcal{L}/\partial W_{i,j})W_{i,j}<0$. If $\partial\mathcal{L}/\partial W_{i,j}<0$ and $W_{i,j}>0$, descent pushes $W_{i,j}$ further up while it is already positive — moving away from zero; if $\partial\mathcal{L}/\partial W_{i,j}>0$ and $W_{i,j}<0$, descent pushes $W_{i,j}$ further down while it is already negative — again moving away from zero. So the score rises exactly when the weight moves away from zero and falls when it heads back. Unrolling from $S^{(0)}=0$,
$$S_{i,j}^{(T)} = -\alpha_S\sum_{t<T}\Big(\frac{\partial\mathcal{L}}{\partial W_{i,j}}\Big)^{(t)} W_{i,j}^{(t)},$$
the score is a running *accumulator of movement away from zero*. The keep/prune decision is the accumulated end-task movement, not the static pretrained value, and I get the scores for free as a by-product of ordinary fine-tuning — no Hessian, no second derivatives as Optimal Brain Surgeon would need. This makes a falsifiable prediction about the resulting weight distributions: magnitude pruning leaves two clumps with an empty gap around zero, whereas movement pruning, selecting on motion rather than value, should leave a smooth distribution spread across the whole range, missing only the values pinned right at zero, with high scores associated with non-zero weights — a v-shape when score is plotted against weight.

That is hard movement pruning, with a fixed per-matrix sparsity from $\mathrm{Top}_v$. I also relax it into soft movement pruning by replacing the top-$v$ cut with a single global threshold $\mathbf{M}=(\mathbf{S}>\tau)$. A bare threshold has nothing pushing the scores down, so it cannot reach a target sparsity on its own; I therefore add a sparsity-inducing regularizer
$$R(\mathbf{S}) = \sum_{i,j}\sigma(S_{i,j}),$$
a sum of sigmoids, and optimize $\mathcal{L}+\lambda_{\text{mvp}}R(\mathbf{S})$, where $\lambda_{\text{mvp}}$ sets the penalty strength and hence the eventual sparsity. The sigmoid keeps the penalty bounded and smooth, which is easier to tune than the obvious $\sum|S_{i,j}|$ alternative, and with a single global threshold the sparsity emerges non-uniformly across layers on its own rather than being forced equal per matrix.

Two things make the design load-bearing rather than arbitrary. First, a first-order Taylor argument shows the method does not break training: take the cleanest case, $\mathrm{TopK}$ with $k=1$, and suppose at step $t{+}1$ a previously-pruned connection $(k,l)$ swaps in for the active $(i,j)$. The top-score inequalities at the two steps, applied to the same pairs, give $S^{(t+1)}_{k,l}-S^{(t)}_{k,l}\ge S^{(t+1)}_{i,j}-S^{(t)}_{i,j}$; substituting the score update and expanding $\mathcal{L}$ to first order around the activations — using that the inactive connection contributes zero activation at each step and that a masked weight receives no loss-gradient update — collapses the loss difference to exactly the rearranged swap inequality, which is $\le 0$. So when a connection becomes more important than the active one and the swap happens, the loss decreases; the argument carries over to a set of simultaneous swaps and to the threshold mask, so soft movement converges too. Second, this same argument is *why I must preserve the sign of the score* and cannot use $|S|$ as importance the way some gradient-based methods do. If I keep the most-negative scores instead — a $\mathbf{M}=(\mathbf{S}<\tau)$ selection — the two-inequality manipulation flips, the bracket comes out $\ge 0$, and the loss *increases* on a swap. Keeping the weights whose score is most positive, i.e. those moving away from zero, is exactly what makes the loss go down; selecting by magnitude of score is provably the wrong thing. It is worth noting that this signal is the same one $L_0$ regularization uses — its score gradient is $\frac{\partial\mathcal{L}}{\partial a_i}W_{i,j}x_j\cdot f(\overline{S}_{i,j})$, my straight-through core times a hard-concrete density factor — so movement pruning is the same first-order signal stripped of the stochastic reparameterization: deterministic, and without a concrete temperature or stretch to tune.

For deployment I wrap this in pre-existing machinery. Rather than a one-shot cut I use automated gradual pruning, raising the sparsity $v$ over training on a cubic schedule (with warm-up $t_i$ and cool-down $t_f$) while masked weights keep updating so the model recovers from early masking choices; for BERT I freeze the embeddings and prune the transformer layers and the task head; and because distillation is orthogonal to the mask rule, I can add a knowledge-distillation loss from a fine-tuned BERT-base teacher as a convex combination with the task loss without touching how the scores are defined. The heart of the implementation is a masked linear layer whose mask comes from the scores through a straight-through top-$v$ or threshold, so scores receive gradients even when masked.

```python
import torch, torch.nn as nn

class TopVSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, keep_ratio):
        mask = torch.zeros_like(scores)
        k = int(keep_ratio * scores.numel())
        idx = scores.flatten().argsort(descending=True)[:k]   # top-v% by score
        mask.flatten()[idx] = 1.0
        return mask
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None          # straight-through: grad -> scores

class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, tau): return (scores > tau).float()
    @staticmethod
    def backward(ctx, grad_out):  return grad_out, None

class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, keep_ratio=1.0, soft=False, tau=0.0):
        super().__init__(in_f, out_f)
        self.score = nn.Parameter(torch.zeros_like(self.weight))   # learned, init 0
        self.keep_ratio, self.soft, self.tau = keep_ratio, soft, tau
    def forward(self, x):
        M = (ThresholdSTE.apply(self.score, self.tau) if self.soft
             else TopVSTE.apply(self.score, self.keep_ratio))
        return nn.functional.linear(x, self.weight * M, self.bias)

def sparsity(t, v_i, v_f, t_i, n_steps, delta_t, total_steps, t_f=0):
    if t < t_i:
        return v_i
    if t >= total_steps - t_f:
        return v_f
    progress = (t - t_i - t_f) / (n_steps * delta_t)
    return v_f + (v_i - v_f) * (1 - progress) ** 3

def fine_prune(model, loader, opt, T, sched, lam_mvp=0.0, teacher=None, soft=False):
    for t, (x, y) in zip(range(T), loader):
        if not soft: set_keep_ratio(model, 1 - sparsity(t, **sched))
        out = model(x); loss = task_loss(out, y)
        if teacher is not None:
            loss = 0.5 * loss + 0.5 * distill_loss(out, teacher(x))
        if soft:
            loss = loss + lam_mvp * sum(torch.sigmoid(m.score).sum()
                                        for m in masked_layers(model))
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```
