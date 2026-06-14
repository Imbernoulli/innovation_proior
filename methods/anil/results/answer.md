# ANIL (Almost No Inner Loop), distilled

ANIL is an inner-loop adaptation rule for gradient-based meta-learning. It keeps the standard
two-loop MAML structure but **removes the inner-loop update for the body of the network and
adapts only the head** (the final classification layer), during both meta-training and
meta-testing. The body's features are learned by the outer loop and then reused as-is on every
task; only the head is task-adapted, because each few-shot task relabels the output neurons. ANIL
matches MAML's few-shot accuracy at a fraction of the per-task cost.

## Problem it solves

In gradient-based meta-learning, the inner loop runs per task inside every meta-iteration over
*all* network parameters, and the meta-gradient differentiates through that inner step
(second-order, Hessian-vector). This is expensive and it is unclear which parameters the
adaptation actually needs to move. ANIL identifies the minimal inner loop that retains full
accuracy: adapt only the head.

## Key idea

The success of a MAML-trained network is dominated by **feature reuse**, not rapid learning. Two
diagnostics on the trained network show this:

- **Freezing.** Freezing a contiguous block of body layers at test time (no inner-loop adaptation)
  barely changes accuracy — even freezing all four convolutional layers (everything but the head)
  leaves MiniImageNet-5way-1shot at $46.3 \pm 0.4$ vs $46.9 \pm 0.2$ unfrozen.
- **Representational similarity.** CCA similarity of each body layer's representation before vs
  after the inner loop is $>0.9$ (CKA near $1$); the head's is $<0.5$. The inner loop barely
  changes the body's function; the head's changes a lot. This holds from early in training.

The head must adapt per task because the $N$ output neurons map to an arbitrary, task-specific set
of classes — a single fixed last layer cannot encode every task's labelling. The body's features
are class-agnostic and reusable. Since the inner loop is functionally inert on the body, removing
the body's inner loop should cost nothing while cutting most of the per-task computation.

## The ANIL update

With $\theta = (\theta_1, \dots, \theta_l)$ (body $\theta_1, \dots, \theta_{l-1}$, head
$\theta_l$), the inner loop updates only the head:

$$\theta^{(b)}_m = \Big(\theta_1, \dots, \theta_{l-1},\;
(\theta_l)^{(b)}_{m-1} - \alpha\,\nabla_{(\theta_l)^{(b)}_{m-1}}
\mathcal{L}^{\text{sup}}_{S_b}(f_{\theta^{(b)}_{m-1}})\Big).$$

The body components stay at the meta-initialization across all inner steps. The outer loop is
unchanged: meta-loss = query loss at the adapted parameters summed over the task batch,
$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}_{\text{meta}}$. The body parameters
are still learned — only by the outer loop, never task-adapted.

## ANIL still carries second-order terms (it is not first-order MAML)

ANIL and first-order MAML cut along different axes. First-order MAML keeps the full inner loop
(adapts all parameters) but drops the Hessian. ANIL keeps the second-order machinery and shrinks
*which* parameters get an inner loop down to the head. Crucially, **a second-order term survives**,
flowing through the head's inner update. On the minimal two-layer linear net
$\hat{y}(x;\theta) = \theta_2(\theta_1 x)$ ($\theta_2$ = head), the body outer-gradient
$\partial L(\hat{y}(x_2^{(t)}; \theta^{(t)}_{\text{ANIL}}), y_2^{(t)}) / \partial \theta_1$
differentiates the query prediction

$$\hat{y}(x_2^{(t)}; \theta^{(t)}_{\text{ANIL}}) =
\Big[\theta_2 - \tfrac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_2}\Big]
\cdot \theta_1 \cdot x_2,$$

and the first bracket — the head's inner-loop update — still depends on $\theta_1$ through the
support forward pass, so differentiating it yields a second derivative. (Contrast full MAML, where
the second bracket $[\theta_1 - \partial L^{\text{sup}}/\partial \theta_1]$ also depends on
$\theta_1$; ANIL drops *that* term by freezing the body, but keeps the one through the head.) This
retained curvature is why ANIL matches full MAML's accuracy where first-order MAML can lag.

## Computational benefit

Because adaptation only touches the head, training and inference per task are much cheaper. At
inference the body is a single forward pass and only a tiny linear head iterates, so the speedup
is largest there (roughly a $4\times$ inference speedup and a $\sim 1.7\times$ training speedup
over full MAML on standard benchmarks).

## Working code (harness inner-loop rule)

Faithful to the differentiable-update primitives of `learn2learn`: gradient w.r.t. the head only,
kept in the graph (`create_graph=True`) for the surviving second-order term; head parameters
stepped by $-\alpha g$, body parameters by zero; head re-identified each step because
`update_module` replaces parameter objects.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import learn2learn as l2l

INNER_LR = 0.5


class InnerLoopOptimizer:
    """ANIL: adapt only the final classification head in the inner loop;
    freeze the body and reuse the meta-initialization's features. Body
    weights are still learned, but only by the outer loop."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Head = final linear classifier; record its parameter NAMES
        # (stable across update_module's object swaps).
        self._head_param_names = set()
        for name, _ in model.named_parameters():
            if "classifier" in name:
                self._head_param_names.add(name)

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            # Re-identify head params each step: update_module replaces the
            # parameter objects, so references from a previous step are stale
            # and would make every update silently zero.
            head_params, head_ids = [], set()
            for name, p in model.named_parameters():
                if name in self._head_param_names:
                    head_params.append(p)
                    head_ids.add(id(p))

            loss = F.cross_entropy(model(support_x), support_y)
            # Grad w.r.t. the head only; create_graph=True keeps the inner
            # step in the meta-graph so the second-order term through the
            # head reaches the outer backward pass.
            grads = torch.autograd.grad(loss, head_params, create_graph=True)
            grad_map = {id(p): g for p, g in zip(head_params, grads)}

            # Full-length update list: -alpha*g for head, zero for body.
            updates = [
                -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
            l2l.update_module(model, updates=updates)  # differentiable p <- p + u
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []  # ANIL adds no learnable optimizer state
```

## NIL (No Inner Loop) — pushing feature reuse to its limit at test time

If test performance is determined by feature quality, the head can be dropped *at test*: pass the
support set through the frozen body to get penultimate representations, then classify a query by
cosine similarity to the support representations, weighting support labels by similarity (the
Matching-Networks rule $\hat{y} = \sum_j a(g(\hat{x}), g(x_j))\,y_j$). No inner loop at all at
test; it matches ANIL/MAML accuracy.

```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def nil_predict(body, support_x, support_y, query_x, n_way):
    # Frozen body features; classify queries by cosine similarity to support.
    s = F.normalize(body(support_x).flatten(1), dim=1)      # [n_sup, d]
    q = F.normalize(body(query_x).flatten(1), dim=1)        # [n_qry, d]
    sims = q @ s.t()                                        # cosine similarities
    onehot = F.one_hot(support_y, n_way).float()            # [n_sup, n_way]
    logits = F.softmax(sims, dim=1) @ onehot                # similarity-weighted labels
    return logits.argmax(dim=1)
```

The head can be removed at test but **not at training**: the per-task head during training is what
forces the body to learn task-discriminative features (training without a per-task head — e.g.
multitask training — yields worse features, even worse than random initialization). So the
asymmetry is: cut the body's inner loop everywhere; keep the head's inner loop at training (it
disciplines the body) and at test (it aligns to each task's labels), and optionally replace the
head at test with the cosine rule.
