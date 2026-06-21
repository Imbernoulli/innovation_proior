# Context: gradient-based few-shot meta-learning (circa 2017)

## Research question

The goal is to make a high-capacity neural network learn a brand-new task from only a handful
of labeled examples — classify novel image classes after one or five examples per class, fit a
new continuous function from a few points, reach a new goal after a few rollouts — and do it
*fast*, ideally in a couple of gradient steps. The setting is not "train on this one task"; it
is "have already meta-trained on *many* related tasks drawn from a distribution `p(T)`, and
exploit that experience to adapt to the next one." Each task `T` brings a small training
(support) set and a held-out test (query) set; the unit of generalization is the task, not the
datapoint. The hard constraint is data: with only `K` examples (often `K=1` or `K=5`) against a
network with far more parameters than `K`, plain training from scratch overfits the support set
and generalizes to nothing.

The precise object we get to optimize ahead of time is *the adaptation procedure itself*: how a
learner that has already meta-trained on many related tasks should move its parameters when given
the support set of a brand-new task.

## Background

The problem is formalized as **few-shot meta-learning**. A distribution over tasks `p(T)`;
each task `T_i` supplies a support set `train(T_i)` and a query set `test(T_i)` and a loss
`L_{T_i}`. During *meta-training* the system sees tasks sampled from `p(T)`; at *meta-test* it
faces held-out tasks and must adapt from `K` examples. The conceptual move, from the
meta-learning tradition (Schmidhuber 1987; Bengio, Bengio & Cloutier 1991; Thrun & Pratt 1998),
is to lift learning from data to tasks: gradual learning *across* tasks produces a fast learner
that operates *within* each task. Matching the meta-training conditions to the meta-test
conditions — episodic sampling of `N`-way `K`-shot tasks with a held-out query set scored each
episode (Vinyals et al. 2016) — is the training protocol that makes this work and is widely
adopted.

A learner here is a parametric, differentiable map `f_θ: X → Y`. The standard way to adapt it to
a task is iterated gradient descent on the support loss,

```
θ^t = θ^{t-1} − α ∇L_T(θ^{t-1}),    L_T(θ) = (1/|T|) Σ_{(x,y)∈T} ℓ(f_θ(x), y),
```

with `α` the learning rate (set by hand) and `θ^0` the initialization (usually random). Three
ingredients fully specify such an optimizer: the **initialization**, the **update direction**,
and the **learning rate**. With abundant data the rules of thumb — random init, follow the
gradient, small/decayed `α` — are common practice.

The obvious cheap baseline — pretrain one network on pooled data, then fine-tune on the new task —
adapts poorly here, because when tasks demand contradictory outputs for the same input (different
sine waves, different goal directions) pooling drives the network toward the average response,
informative about the output range but not about any single task. The structure worth learning is
*task-to-task*, not in any single pooled model.

Two framings inform what "learn the optimizer" could mean. First, hand-designed adaptive
optimizers — AdaGrad (Duchi et al. 2011), RMSProp (Tieleman & Hinton 2012), Adam (Kingma & Ba
2014), AdaDelta (Zeiler 2012) — show that scalar SGD is not the only useful step geometry: they
adapt step magnitudes from gradient history (its `L2` norm and, for Adam, the gradient itself).
They are still fixed recipes tied to a history, not learning rules trained from a task
distribution. Second, recurrent networks can model adaptive optimization procedures (Cotter &
Conwell 1990; Younger et al. 1999; Hochreiter et al. 2001 found LSTM best among RNN
architectures), which opens the door to *learning* an optimizer rather than hand-designing it.

## Baselines

These are the prior methods a new few-shot meta-learner would be measured against and react to.

**MAML — Model-Agnostic Meta-Learning (Finn, Abbeel & Levine, ICML 2017).** Meta-learn only the
*initialization* `θ`; keep the inner loop as ordinary SGD with a scalar step-size hyperparameter
`α`. For task `T_i`, one inner step gives `θ'_i = θ − α ∇L_{T_i}(f_θ)` (or a few such steps).
The meta-objective optimizes the *query* loss of the *adapted* parameters across tasks,

```
min_θ  Σ_{T_i ~ p(T)}  L_{T_i}(f_{θ'_i}),   θ'_i = θ − α ∇L_{T_i}(f_θ),
```

and the meta-update `θ ← θ − β ∇_θ Σ_i L_{T_i}(f_{θ'_i})` differentiates *through* the inner
gradient step — a gradient-through-a-gradient, requiring a Hessian-vector product (a
first-order variant, FOMAML, drops it). It is architecture-agnostic, adds no parameters beyond
`θ`, and works across supervised and RL task forms.

**Meta-Learner LSTM — "Optimization as a model for few-shot learning" (Ravi & Larochelle, ICLR
2017),** extending **"Learning to learn by gradient descent by gradient descent"
(Andrychowicz et al., NeurIPS 2016).** Meta-learn the *entire* update rule with an LSTM whose
cell state *is* the learner's parameters. Writing the LSTM cell update `c_t = f_t ⊙ c_{t-1} +
i_t ⊙ c̃_t` and identifying `c_t = θ_t`, `c̃_t = −∇L_t`, the input gate `i_t = α_t` becomes a
*learned* learning rate and the forget gate `f_t` a *learned* shrink/decay of the previous
parameters: `α_t = σ(W_I·[∇L_t, L_t, θ_{t-1}, i_{t-1}] + b_I)`, `f_t = σ(W_F·[∇L_t, L_t,
θ_{t-1}, f_{t-1}] + b_F)`. The initial cell state `c_0` (the learner's initialization) is also
learned. To avoid an explosion of meta-parameters the same small LSTM is *shared across all
coordinates* (coordinate-wise), with the Andrychowicz log-magnitude-and-sign gradient
preprocessing; training is by BPTT through the unrolled inner loop under a gradient-independence
simplification (treat `∇L_t` as not depending on the meta-parameters, avoiding second
derivatives). BPTT stores every intermediate LSTM state, a space cost `O(T · #states · dim(θ))`
(typically `T≈8`, `#states≈20`).

**Matching Networks (Vinyals et al., NeurIPS 2016)** and earlier **Siamese Networks (Koch
2015).** Metric/attention meta-learners: learn an embedding and classify a query by a
(soft-)nearest-neighbor comparison to the support embeddings, trained episodically. They learn a
distance-based comparison rather than adapting the parameters of a parametric learner.

## Evaluation settings

The natural yardsticks already in use at the time (these datasets/protocols predate any new
method):

- **Few-shot image classification.** *Omniglot* (Lake et al. 2011): 1623 characters from 50
  alphabets, 20 instances each, conventionally 1200 characters for meta-training and the rest
  for meta-test; 5-way and 20-way, 1-shot and 5-shot, images downsampled to 28×28.
  *MiniImagenet* (Ravi & Larochelle 2017 split): 100 classes × 600 color images, 64 / 16 / 20
  classes for meta-train / meta-val / meta-test; 5-way (and 20-way) 1-shot and 5-shot, images
  84×84. Episodic `N`-way `K`-shot sampling with a held-out query set per episode (Vinyals et
  al. 2016); metric is mean classification accuracy over many test episodes with 95% confidence
  intervals. Standard backbone: four conv modules (3×3 conv → batch-norm → ReLU → 2×2 max-pool),
  e.g. 32 filters for MiniImagenet, following the MAML setup; meta-batches of a few tasks.
- **Few-shot regression.** Sine curves `y(x) = A sin(ω x + b)` with `A ∈ [0.1, 5.0]`,
  `ω ∈ [0.8, 1.2]`, `b ∈ [0, π]`, inputs in `[−5, 5]`; `K ∈ {5, 10, 20}` support points, MSE
  loss; a small MLP (1 → 40 → 40 → 1, ReLU). Tests whether a learner can recover an unseen sine
  from a few points, even when they cover only half the input range.
- **Few-shot reinforcement learning.** 2D point-navigation: reach a sampled goal in the unit
  square (fixed or random start); state = position, action = velocity, reward = negative
  distance to goal; a small policy MLP (2 → 100 → 100 → 2), vanilla policy-gradient for the
  inner step and a trust-region outer update; return averaged over many test tasks.
- Protocol throughout: meta-train across sampled task batches, meta-validate for
  hyperparameters, report on held-out meta-test tasks; one or a few inner adaptation steps;
  the comparison is against MAML, Meta-Learner LSTM, Matching/Siamese Nets under matched splits.

## Code framework

The method plugs into a standard gradient-based meta-learning harness that already exists. The
*outer* loop is fixed: sample a batch of tasks; for each, clone the learner from the shared
initialization, run an inner adaptation on the support set, score the adapted clone on the
query set, and meta-update across tasks by back-propagating the summed query loss *through* the
inner adaptation. The data pipeline (episodic `N`-way `K`-shot tasksets), the backbone, the
cross-entropy loss, the differentiable parameter-update primitive (a functional update that
re-routes parameter tensors so gradients flow back through the inner step to the
meta-parameters), and the meta-optimizer are all given. What is *not* settled is the inner
adaptation rule itself — exactly what is to be designed.

The inner rule lives behind a single object with a constructor, an `adapt` method that takes a
*cloned* learner and the support set and returns the adapted learner using *differentiable* ops
(`torch.autograd.grad`, not `torch.optim`, so the graph survives), and a `meta_parameters`
accessor returning any optimizer state that should be trained by the outer loop. The starting
default is the plainest possible inner rule — ordinary SGD with one fixed scalar rate and no
trainable optimizer state — with one neutral empty slot where the designed update rule will go.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5  # default scalar inner-loop step size (a hand-set hyperparameter)


class InnerLoopOptimizer(nn.Module):
    """Inner-loop adaptation rule for gradient-based meta-learning.

    Defines HOW the cloned learner's parameters move during fast adaptation on a
    task's support set. The outer (meta) loop is fixed and differentiates THROUGH
    whatever this object does, so adapt() must use differentiable ops and any
    learnable optimizer state must be exposed via meta_parameters().
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE — safe to modify; must stay differentiable so the
        # outer loop can backprop through the inner step.
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, create_graph=True
            )
            model = self._apply_update(model, grads)
        return model

    def _apply_update(self, model: nn.Module, grads: List[Tensor]) -> nn.Module:
        # TODO: define the adaptation rule.
        raise NotImplementedError

    def meta_parameters(self) -> List[Tensor]:
        return list(self.parameters())
```

The outer loop calls `adapt` on each task's clone and meta-optimizes the shared initialization
together with whatever `meta_parameters()` returns; the single empty slot is the inner update
rule.
