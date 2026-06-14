# Context: gradient-based meta-learning and the cost of its inner loop (circa 2019)

## Research question

Gradient-based meta-learning has become the dominant recipe for few-shot learning: learn an
initialization for a neural network such that, faced with a brand-new task and only one to five
labelled examples per class, a handful of ordinary gradient steps on those examples produces a
model that generalizes. The recipe has two nested optimization loops. The **outer loop** learns
the initialization (the *meta-initialization*) across a distribution of training tasks. The
**inner loop** performs *adaptation*: starting from that initialization, it takes a few gradient
steps on a new task's support set. This works remarkably well, but it is expensive and it is
opaque. Expensive, because the inner loop is run *per task, inside every meta-iteration*, over
**every parameter of the network**, and differentiating the meta-loss through that inner gradient
step requires second-order (Hessian-vector) information for the whole network. Opaque, because
nobody can say *what the inner loop is actually doing* — whether the meta-initialization is a
launch point conditioned for large, efficient changes in the network's internal representations
during adaptation, or whether it already contains good representations that adaptation mostly
leaves alone.

The precise problem is twofold and the two halves are linked. First, **diagnostic**: determine
empirically which of those two stories is true for a trained meta-learner — does adaptation
substantially change the network's learned features for each task, or not? Second,
**algorithmic**: if the inner loop turns out to be doing far less than its cost implies, find the
*minimal* inner-loop adaptation rule that retains full few-shot accuracy while cutting the
per-task computation. A solution would have to (1) preserve benchmark accuracy on standard
few-shot classification (and ideally RL), (2) materially reduce the per-task cost of training and
inference relative to adapting the whole network, and (3) remain a *gradient-based* adaptation
rule that plugs into the existing two-loop harness without new model machinery. The contribution
sought is the *inner-loop adaptation rule itself* — which parameters change during adaptation,
and how — not changes to the data pipeline, the backbone, or the outer-loop schedule.

## Background

**The two-loop structure.** Let the base network be $f_\theta$ with parameters
$\theta = (\theta_1, \dots, \theta_l)$ for its $l$ layers. A task distribution $\mathcal{D}$
yields tasks; each task $T_b$ has a *support set* $\mathcal{S}_{T_b}$ (for inner-loop adaptation)
and a *target/query set* $\mathcal{Z}_{T_b}$ (for the outer update). Writing $\theta^{(b)}_m$ for
the parameters after $m$ inner steps on task $b$ (with $\theta^{(b)}_0 = \theta$), the inner loop
is plain gradient descent on the support loss,
$$\theta^{(b)}_m = \theta^{(b)}_{m-1} - \alpha\,\nabla_{\theta^{(b)}_{m-1}}\,
\mathcal{L}_{\mathcal{S}_{T_b}}\!\big(f_{\theta^{(b)}_{m-1}}\big),$$
the meta-loss is the query loss at the *adapted* parameters summed over a batch of tasks,
$\mathcal{L}_{\text{meta}}(\theta) = \sum_b \mathcal{L}_{\mathcal{Z}_{T_b}}(f_{\theta^{(b)}_m})$,
and the outer loop is $\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}_{\text{meta}}(\theta)$.
Because the adapted parameters $\theta^{(b)}_m$ depend on $\theta$ both directly and *through the
inner gradient*, the meta-gradient carries a second-order term: for one inner step it is
$(I - \alpha\,\nabla^2_\theta \mathcal{L}^{\text{sup}})^\top \nabla_{\theta'} \mathcal{L}^{\text{qry}}$,
a Hessian-vector product, obtained by retaining the inner step in the autodiff graph.

**The head/body distinction is structural, not a choice.** In an $N$-way episode the network's
$N$ output neurons are mapped to an *arbitrary, task-specific* set of classes: in one task the
five outputs might mean (dog, cat, frog, cupcake, phone) and in the next (airplane, frog, boat,
car, pumpkin). A single fixed last layer cannot encode all such alignments at once — whatever the
adaptation rule, the final layer (the *head*) must be free to take on each task's labelling. The
earlier layers (the *body*) compute class-agnostic features (edges, textures, parts) that are, in
principle, shared across tasks. So the head and the body sit in genuinely different positions with
respect to task specificity, and the rapid-learning-vs-feature-reuse question is really a question
about the *body*: it is the body where the two hypotheses predict different things.

**Diagnostic instruments for "did the representation change?".** Two representational-similarity
tools existed for comparing the latent functions computed by two layers. **Canonical Correlation
Analysis (CCA)** of neuron-activation vectors (SVCCA, Raghu et al. 2017; the CCA-insights variant,
Morcos et al. 2018) finds linear combinations of the two layers' neurons that are maximally
correlated over a set of inputs and summarizes them into a similarity score in $[0,1]$ ($1$
identical, $0$ unrelated); for convolutional layers one compares over channels, flattening the
spatial dimensions. **Centered Kernel Alignment (CKA)** (Kornblith et al. 2019) is a second,
independent similarity score in $[0,1]$. Either can be applied with $L_1$ the representation of a
layer *before* inner-loop adaptation and $L_2$ the same layer *after* adaptation, to measure how
much that layer's function moved during the inner loop.

**The diagnostic findings about MAML-trained networks.** These are measurements of the *existing*
system, on MiniImageNet (5-way) with the standard four-convolutional-block architecture, averaged
over three seeds, and they are the empirical heart of the setup:

- *Freezing.* After standard two-loop training, freeze a contiguous block of layers at test time
  (those layers get no inner-loop adaptation and must reuse the meta-initialization's features),
  and read off accuracy. Freezing successively more of the body changes accuracy remarkably little
  — even freezing *all four* convolutional layers (everything but the head) leaves accuracy almost
  unchanged. On 5-way-1-shot: no freezing $46.9 \pm 0.2$; freeze layer 1, $46.5 \pm 0.3$; freeze
  1–2, $46.4 \pm 0.4$; freeze 1–3, $46.3 \pm 0.4$; freeze 1–4, $46.3 \pm 0.4$. On 5-way-5-shot:
  $63.1 \pm 0.4$ down to $61.0 \pm 0.6$ at full freezing.
- *Representational similarity.* CCA similarity between each body layer's representation before and
  after adaptation is high ($>0.9$); CKA likewise near $1$. The head, by contrast, has CCA
  similarity below $0.5$ — it moves a lot. So during adaptation the body's *function* barely
  changes while the head's changes substantially.
- *Timing.* The same pattern (high body similarity, near-unchanged accuracy under freezing) holds
  from early in training, e.g. from $\sim$10{,}000 iterations onward, not just at convergence.
- *Weight movement.* The per-layer average Euclidean distance $\frac{1}{N}\sum_b \|\theta_l -
  (\theta_l)^{(b)}_m\|$ between initialization and finetuned weights is tiny for every layer except
  the last, *despite the body layers having more parameters than the head*.

These are facts about how a fully-trained MAML network behaves under its own inner loop. They say
the inner loop, run over the body, induces almost no functional change in the body.

## Baselines

**MAML (Finn, Abbeel & Levine, ICML 2017).** The inner loop is differentiable SGD with a fixed
scalar learning rate $\alpha$ applied to *all* parameters; the outer loop optimizes only the
initialization. Its meta-gradient is the exact bilevel derivative
$(I - \alpha\,\nabla^2_\theta \mathcal{L}^{\text{sup}})^\top \nabla_{\theta'} \mathcal{L}^{\text{qry}}$,
computed by keeping the inner step in the graph (`create_graph=True`) so a second backward pass
yields the Hessian-vector product without ever materializing the Hessian. **Gap:** the inner loop
and its second-order term run over the *entire* network for every task in every meta-iteration,
which dominates per-task cost; and the algorithm gives no insight into which parameters the
adaptation actually needs to move — it moves all of them by construction.

**First-order MAML / Reptile (Finn et al. 2017; Nichol & Schulman 2018).** Drop the Hessian: use
the query-loss gradient evaluated at the *adapted* point as if it were the gradient at $\theta$
(`first_order`/detach the inner step); Reptile is a related first-order scheme that moves the
initialization toward each task's adapted weights. **Gap:** cheaper than full MAML, but it still
runs the inner loop over *all* parameters, and it discards *all* curvature information, so on some
benchmarks it underperforms full MAML.

**Meta-SGD (Li, Zhou, Chen & Li, 2017).** Replace the single scalar inner learning rate with a
*per-parameter* learnable rate vector $\boldsymbol{\alpha}$, meta-trained jointly with the
initialization; the inner step becomes
$\theta' = \theta - \boldsymbol{\alpha} \odot \nabla_\theta \mathcal{L}^{\text{sup}}$. **Gap:**
it tunes *how far* each parameter moves but still moves every parameter in the inner loop, and it
adds roughly $|\theta|$ extra learnable values; it does not reduce — it slightly increases — the
per-task inner-loop work and the parameter count.

**Metric-based few-shot methods (Matching Networks, Vinyals et al. 2016; Prototypical Networks,
Snell et al. 2017; Relation Net, Sung et al. 2018).** Learn an embedding and classify a query by
comparing it (cosine or Euclidean similarity, or prototypes) to the labelled support set — for
Matching Networks, $\hat{y} = \sum_j a(g(\hat{x}), g(x_j))\,y_j$ with $a$ a softmax over cosine
similarities. There is no per-task parameter optimization at all. **Gap:** this is a
classification-specific construction (no obvious "prototype" of a regression function or a
policy), and it answers a different question than the optimization-based line; it is relevant here
because it provides a parameter-free way to turn frozen support representations into predictions.
A telling pre-existing observation in this line: on MiniImageNet-5way-1shot, Matching Networks
with joint encoding of the support set reaches $44.2\%$ while independent encoding reaches
$41.2\%$ — a small gap, suggesting that even metric methods lean heavily on the quality of fixed
features rather than on task-specific adaptation.

**Representational-similarity analysis tools (SVCCA, Raghu et al. 2017; CCA insights, Morcos et
al. 2018; CKA, Kornblith et al. 2019).** Not methods to beat but the measuring sticks above:
quantify how similar two layers' learned functions are, in $[0,1]$. Their limitation as
*algorithms* is that they only describe representations; they do not by themselves prescribe a new
adaptation rule. Their value here is diagnostic — they make "did adaptation change this layer?"
answerable.

## Evaluation settings

The standard few-shot yardsticks that already existed, used unchanged:

- **MiniImageNet 5-way 1-shot and 5-way 5-shot** (Ravi & Larochelle 2016): 64 training / 12
  validation / 24 test classes; each episode draws 5 classes and 1 or 5 support examples per class
  plus query examples; the standard backbone is four convolutional modules ($3\times3$ conv,
  $32$–$64$ filters, batch-norm, ReLU, $2\times2$ max-pool) and a linear head. Metric: mean
  classification accuracy over test episodes (with 95% confidence intervals).
- **CIFAR-FS 5-way 5-shot**: the few-shot split of CIFAR; same episodic 5-way protocol; same CNN
  backbone after resizing.
- **Omniglot 20-way 1-shot / 5-shot** (handwritten characters, 1600+ classes split by character):
  same family of CNN backbones at lower resolution.
- **RL benchmarks** (HalfCheetah-Direction, HalfCheetah-Velocity, 2D-Navigation): a two-layer MLP
  policy, average return as the metric, adapted with policy gradients in the inner loop.
- Protocol: meta-train for tens of thousands of iterations, a small meta-batch of tasks per outer
  update (e.g. 4), a few inner-loop steps during training (e.g. 5) and a few more at test (e.g.
  10), inner learning rate $\alpha$ on the order of $0.01$–$0.5$ depending on the setting, the
  outer loop driven by Adam. Models trained from several random seeds; comparisons read off mean
  accuracy with confidence intervals over test episodes. Per-iteration wall-clock time
  (train and inference) is also a natural yardstick, since cost is part of the problem.

## Code framework

The adaptation rule plugs into the fixed two-loop meta-learning harness. The outer loop, the data
pipeline, the CNN backbone, the meta-optimizer, the support/query split, and the evaluation loop
all already exist and are not in question. What is undetermined — the single empty slot — is the
inner-loop adaptation rule: given a freshly cloned model and its support set, how to update it
with differentiable operations so that gradients still flow to the outer loop. The harness clones
the model per task (so the rule may modify it in place), and the rule must use differentiable
gradient operations (`torch.autograd.grad`, keeping the step in the graph) rather than a
`torch.optim` optimizer, which would break the meta-graph. A differentiable in-place update
primitive `update_module(model, updates=[...])` is available: it swaps each parameter $p$ for
$p + u$ using a full-length list of per-parameter updates $u$, preserving differentiability.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import learn2learn as l2l

INNER_LR = 0.5  # default inner-loop learning rate provided by the harness


class InnerLoopOptimizer:
    """Inner-loop adaptation rule for gradient-based meta-learning.

    Defines HOW the cloned model's parameters are updated during fast
    adaptation to one task's support set. The outer loop (meta-optimizer),
    data pipeline, backbone, and evaluation are fixed; only this rule is open.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # TODO: inspect the model and set up whatever the adaptation rule needs.

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE (safe to modify in place).
        # Must use differentiable ops (torch.autograd.grad + update_module),
        # NOT torch.optim, so gradients flow to the outer loop.
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            # TODO: the adaptation rule we will design — decide which
            #       parameters move and by how much, compute differentiable
            #       updates, and apply them with update_module.
            pass
        return model

    def meta_parameters(self) -> List[Tensor]:
        # Any learnable optimizer state for the outer loop to optimize.
        # TODO: return the rule's learnable parameters, or [] if it has none.
        pass


# existing per-task meta-training step the rule plugs into (fixed)
def meta_train_step(model, inner_opt, meta_optimizer, taskset,
                    n_way, n_shot, meta_batch_size, inner_steps, device):
    meta_loss = 0.0
    for _ in range(meta_batch_size):
        learner = l2l.clone_module(model)                 # fresh clone per task
        data, labels = taskset.sample()
        sx, sy, qx, qy = split_support_query(data.to(device), labels.to(device),
                                             n_way, n_shot)
        learner = inner_opt.adapt(learner, sx, sy, inner_steps)  # inner loop
        meta_loss = meta_loss + F.cross_entropy(learner(qx), qy)  # query loss
    meta_loss = meta_loss / meta_batch_size
    meta_optimizer.zero_grad()
    meta_loss.backward()                                  # second-order if kept in graph
    meta_optimizer.step()                                 # outer update of the init
    return meta_loss.item()
```

The outer loop supplies one cloned model and its support set per task; `adapt()` is where the
inner-loop rule will live, and `meta_parameters()` exposes any learnable state it adds.
