# Context

The setting is compressing large pretrained Transformer language models (BERT) for task-specific deployment, around 2019–2020. The dominant NLP recipe is *transfer learning*: pretrain a large model (BERT-base, ~84M parameters in the encoder) on a massive generic corpus, then fine-tune it on a smaller annotated end-task dataset. Accuracy scales with model and pretraining size, but the resulting models are expensive to ship and run, and most of the energy on edge devices goes to fetching parameters from storage into memory. Reducing model size at small accuracy cost is therefore valuable, and unstructured weight pruning is one of the simplest, most effective levers.

## Research question

Magnitude pruning — keep the weights with the largest absolute value, prune the rest — is the standard, and it works very well for models trained from scratch on the end task. But in the *transfer learning* regime it is much less effective, especially at high sparsity. Why, and what should replace it? The crux: when a model is trained from scratch, its final weight values are determined by the end task, so "large weight" genuinely means "important for this task." But when fine-tuning a pretrained model, the weights are *mostly inherited from pretraining* and only nudged by fine-tuning — they stay close to their pretrained values. So which weights end up large is decided largely by pretraining, not by the end task. A magnitude criterion therefore prunes based on pretraining values, and you can predict before fine-tuning even begins which weights it will discard. A solution must let the *fine-tuning process itself* decide which weights to keep — including allowing a weight that is large to be pruned (if fine-tuning drives it toward zero) and a weight that is small to be kept (if fine-tuning drives it away from zero).

## Background

- **Score-based pruning, shared notation.** For a weight matrix $\mathbf{W}\in\mathbb{R}^{n\times n}$, introduce a parallel matrix of importance scores $\mathbf{S}$ and a binary mask $\mathbf{M}\in\{0,1\}^{n\times n}$; inference is $\mathbf{a}=(\mathbf{W}\odot\mathbf{M})\mathbf{x}$. A common strategy keeps the top-$v\%$ of weights by score: $\mathrm{Top}_v(\mathbf{S})_{i,j}=1$ iff $S_{i,j}$ is in the top $v\%$, else 0. **Magnitude pruning** is the special case $\mathbf{S}=(|W_{i,j}|)$, a *zeroth-order* (value-only) criterion.
- **Iterative magnitude pruning** (Han et al. 2015): train to convergence, remove lowest-magnitude weights, retrain with them fixed at 0, repeat — the basis of the lottery-ticket line (Frankle & Carbin 2019).
- **Automated gradual pruning** (Zhu & Gupta 2018): instead of a hard post-hoc cut, raise the sparsity level $v$ gradually during training on a cubic schedule, $v^{(t)} = v_f + (v_i - v_f)\big(1 - \tfrac{t-t_i}{N\Delta t}\big)^3$, while letting masked weights still be updated, so the model can recover from earlier masking choices. The model is pruned and trained jointly.
- **Higher-order saliency.** Optimal Brain Damage (LeCun et al. 1990) and Optimal Brain Surgeon (Hassibi & Stork 1993) use the Hessian of the loss to choose deletions — accurate but requires costly second-order information. Other lines use the absolute or squared value of the gradient as importance (Theis et al. 2018; Ding et al. 2019; Lee et al. 2019, SNIP).
- **Learned masks via score matrices.** Piggyback (Mallya et al. 2018) and "What's hidden in a randomly weighted network" (Ramanujan et al. 2019) keep the weights fixed and learn a parallel score matrix to select a good sub-network, using a threshold or top-$k$ mask. They never fine-tune $\mathbf{W}$ itself.
- **$L_0$ regularization** (Louizos et al. 2018): make the mask stochastic via a *hard-concrete* distribution so an expected-$L_0$ penalty is differentiable, and train weights and gates end-to-end to minimize $\mathcal{L}+\lambda\,\mathbb{E}(L_0)$. A first-order, learned-sparsity method, but with extra distributional machinery (Gumbel-style sampling, temperature, stretch parameters).
- **Straight-through estimator** (Bengio et al. 2013): when a forward operation has zero or undefined gradient (like a hard top-$k$ or threshold), approximate its backward pass by passing the gradient straight through as if it were the identity.
- **Empirical fact about fine-tuning** (observed by Gordon et al. 2020): fine-tuned weights stay close in absolute value to their pretrained values.
- **Distillation** (Bucilă et al. 2006; Hinton et al. 2015): a smaller/compressed student can be trained to match a teacher's output distribution, a complementary boost to any compression scheme.

## Baselines

- **Magnitude pruning** (with automated gradual pruning + cubic schedule). Strong at low sparsity (≥70% remaining) on transfer tasks, but degrades fast at high sparsity because the keep/prune decision is essentially fixed by the pretrained values, not the end task.
- **$L_0$ regularization.** A first-order, learned-mask method that does adapt during fine-tuning, but carries the overhead of the hard-concrete reparameterization and is harder to tune. Leaves open whether a simpler first-order criterion suffices.
- **Learned-score masking on frozen weights** (Piggyback, hidden-networks). Adapt a mask but keep $\mathbf{W}$ fixed — not designed for the fine-tunable-$\mathbf{W}$ transfer setting.
- **Structured pruning / smaller pretrained models** (LayerDrop; mini-BERT). Remove whole heads/layers or pretrain a smaller model; a different size/speed trade-off than unstructured weight pruning.

## Evaluation settings

- **Model:** `BERT-base-uncased` (~84M encoder params). Freeze the embeddings; prune/fine-tune the Transformer layers and the task head. Sparsity percentages are relative to BERT-base and correspond exactly to model size, including for baselines.
- **Tasks:** question answering SQuAD v1.1 (~8K train), natural-language inference MNLI (~393K), sentence similarity QQP (~364K). Metrics: EM/F1 for SQuAD, accuracy/MM-accuracy for MNLI, accuracy/F1 for QQP.
- **Protocol:** fine-tune the same number of updates (≈6–10 epochs) across pruning methods; cubic sparsity schedule with optional cool-down steps at the end. Compare against $L_0$ regularization, RPP, LayerDrop, and mini-BERT. Optionally add a knowledge-distillation loss from a fine-tuned BERT-base teacher (convex combination of task loss and distillation loss). All runs fit on a single 16 GB V100.

## Code framework

A pruning method is a masked linear layer plus a rule for the scores, the mask, and how scores get gradients. The pre-method scaffold leaves those slots open.

```python
import torch, torch.nn as nn

class MaskedLinear(nn.Linear):
    """A linear layer whose weights are multiplied by a learned binary mask."""
    def __init__(self, in_f, out_f, keep_ratio):
        super().__init__(in_f, out_f)
        self.keep_ratio = keep_ratio
        # TODO: an importance score per weight, of whatever order the method needs.
        self.score = nn.Parameter(torch.zeros_like(self.weight))  # placeholder slot

    def mask_from_scores(self):
        # TODO: turn scores into a {0,1} mask (e.g. top-v% or threshold) such that
        #       the chosen importance signal can still receive gradients.
        raise NotImplementedError

    def forward(self, x):
        M = self.mask_from_scores()
        return nn.functional.linear(x, self.weight * M, self.bias)

def sparsity_schedule(t, v_i, v_f, t_i, N, dt):
    # cubic automated-gradual-pruning schedule, already standard
    if t < t_i: return v_i
    return v_f + (v_i - v_f) * (1 - (t - t_i) / (N * dt)) ** 3

# Fine-tuning loop on a pretrained encoder (embeddings frozen).
def fine_prune(model, loader, opt, total_steps, v_i, v_f, t_i, N, dt):
    for t, (x, y) in zip(range(total_steps), loader):
        set_keep_ratio(model, 1 - sparsity_schedule(t, v_i, v_f, t_i, N, dt))
        loss = task_loss(model(x), y)   # (optionally + distillation loss)
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```
