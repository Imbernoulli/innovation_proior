# Context: gradient-based meta-learning and the inner-loop cost question

## Research question

Gradient-based meta-learning for few-shot learning trains an initialization that can be adapted to a new task from only a few labelled examples. The standard setup has two nested loops. The outer loop learns the meta-initialization across tasks; the inner loop starts from that initialization and takes gradient steps on the support set of the current task. This recipe works well on few-shot image classification and small reinforcement-learning domains, but it is expensive because every sampled task runs a differentiable inner optimization inside every meta-iteration.

The unresolved question is why the inner loop is useful. One possible story is rapid learning: the initialization is conditioned so a few gradient steps make large, task-specific representational changes. Another possible story is feature reuse: the initialization already contains broadly useful features, and the inner loop mainly handles the task-specific output alignment. These two stories imply very different algorithms. If the whole network genuinely changes its representation for each task, the full inner loop is load-bearing. If most of the representation is reused, much of the inner loop may be unnecessary computation.

## Meta-Learning Mechanics

Let the model be $f_\theta$, with parameters partitioned by layers as $\theta=(\theta_1,\ldots,\theta_l)$. A task distribution $\mathcal D$ yields tasks $T_b$. Each task has a support set $\mathcal S_{T_b}$ for adaptation and a target/query set $\mathcal Z_{T_b}$ for the meta-objective. With $\theta^{(b)}_0=\theta$, the standard inner update is
$$
\theta^{(b)}_m
= \theta^{(b)}_{m-1}
- \alpha \nabla_{\theta^{(b)}_{m-1}}
\mathcal L_{\mathcal S_{T_b}}\!\left(f_{\theta^{(b)}_{m-1}}\right).
$$
The meta-loss is
$$
\mathcal L_{\mathrm{meta}}(\theta)
= \sum_b \mathcal L_{\mathcal Z_{T_b}}\!\left(f_{\theta^{(b)}_m}\right),
$$
and the outer update is $\theta \leftarrow \theta-\eta\nabla_\theta \mathcal L_{\mathrm{meta}}(\theta)$.

For one inner step, the meta-gradient differentiates through the inner gradient. In the simplest all-parameter case it contains the factor
$$
\left(I-\alpha \nabla_\theta^2 \mathcal L_{\mathcal S}^{\mathrm{sup}}\right)^\top
\nabla_{\theta'} \mathcal L_{\mathcal Z}^{\mathrm{qry}},
$$
so ordinary autodiff must retain the inner step graph and compute a Hessian-vector product. This is precisely the computational burden to inspect: the inner update is not just a forward pass or a normal backward pass, and it is repeated per task.

## Head, Body, And Diagnostics

Few-shot classification episodes have a structural asymmetry between the final layer and the earlier layers. In an $N$-way episode, the $N$ output coordinates are assigned to a fresh set of classes. One task might map the five output neurons to dog, cat, frog, cupcake, and phone; another might map them to airplane, frog, boat, car, and pumpkin. The final classifier therefore has a task-specific alignment problem that the earlier feature extractor does not have.

The earlier layers compute features that can plausibly be shared across many tasks, while the final layer maps those features to the current episode's labels. The rapid-learning-vs-feature-reuse question should therefore be asked mainly about the body: do the earlier layers actually change their represented function during adaptation, or do they mostly reuse the meta-initialization?

Two diagnostics can answer that without changing the training objective. First, after training a standard meta-learner, freeze contiguous blocks of body layers during test-time adaptation and compare accuracy with the unfrozen model. Second, compare each layer's activations before and after adaptation using representational similarity tools. SVCCA and related CCA-based scores compare linear subspaces of neuron activations and return a similarity score in $[0,1]$; CKA gives an independent kernel-alignment score in the same spirit. These instruments measure whether a layer's function changes during the inner loop.

## Baselines And Gaps

**MAML (Finn, Abbeel, Levine, 2017).** The inner loop applies differentiable SGD to every parameter, and the outer loop optimizes the initialization through the adapted query loss. This is the clean reference point: general, model-agnostic, and second-order when implemented exactly. Its gap is cost and opacity. It pays for full-network task adaptation without revealing which parameters truly have to move.

**First-order MAML and Reptile.** First-order variants reduce cost by dropping second-order terms or by moving the initialization toward task-adapted weights. They still adapt the whole network in the inner loop. The gap is that they cut curvature rather than identifying which parts of the model require task-specific adaptation.

**Meta-SGD.** Meta-SGD learns a per-parameter inner learning-rate vector, replacing $\alpha$ with a learned $\boldsymbol\alpha$ in $\theta'=\theta-\boldsymbol\alpha\odot\nabla_\theta\mathcal L^{\mathrm{sup}}$. It can learn that different parameters should move at different scales, but it still computes gradients for all parameters and adds optimizer state of roughly the same size as the model.

**Metric-based few-shot methods.** Matching Networks, Prototypical Networks, and Relation Networks learn an embedding and classify queries by comparing them with the support set. They avoid per-task parameter optimization, but their machinery is specialized to classification and does not directly answer which part of a gradient-based meta-learner is doing useful work.

**Representation-similarity tools.** SVCCA, CCA-insights, and CKA are not competing meta-learning algorithms. They are measurement tools for deciding whether two layers compute similar representations before and after a perturbation such as adaptation.

## Evaluation And Code Scaffold

The relevant evaluation settings are the standard few-shot benchmarks already used for this line of work: MiniImageNet 5-way 1-shot and 5-way 5-shot with a four-convolutional-block backbone and linear classifier; Omniglot 20-way 1-shot and 5-shot; CIFAR-FS; and simple policy-gradient reinforcement-learning tasks such as HalfCheetah-Direction, HalfCheetah-Velocity, and 2D-Navigation. Metrics are mean few-shot classification accuracy or average return, with wall-clock time for training and inference because the target of the design is inner-loop computation.

The code framework fixes the outer loop, data pipeline, model backbone, support/query split, and meta-optimizer. The open component is only the differentiable inner-loop adaptation rule. A cloned task model may be modified, but the update has to be expressed with `torch.autograd.grad` and `learn2learn.update_module`, not with a normal `torch.optim` step, so the query loss can still backpropagate to the meta-initialization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import learn2learn as l2l

INNER_LR = 0.5


class InnerLoopOptimizer:
    """Defines the differentiable fast-adaptation rule inside the meta-learner."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Inspect the cloned model and identify which parameters the rule updates.

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            # Compute differentiable parameter updates and apply them with
            # l2l.update_module(model, updates=...).
            pass
        return model

    def meta_parameters(self) -> List[Tensor]:
        # Return learnable optimizer state, if the adaptation rule has any.
        return []
```
