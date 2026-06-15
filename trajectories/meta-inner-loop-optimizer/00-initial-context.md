## Research question

Gradient-based meta-learning learns a model initialization that adapts to a brand-new task from a
handful of labeled examples through a few gradient steps. The split is two-loop: an **inner loop**
adapts the model on a task's small support set, an **outer loop** optimizes the initialization (and
any optimizer state) across a distribution of tasks. The single thing being designed here is the
**inner-loop adaptation rule** — *which* parameters change during adaptation, *how* the support
gradient is scaled or transformed before it is applied, and *what* state (if any) is carried across
inner steps and meta-learned by the outer loop. Everything else — the data loader, the CNN4 backbone,
the meta-batch schedule, the number of inner steps, the outer optimizer — is fixed. The contribution
is the adaptation rule itself, not the pipeline around it.

## Prior art before the first rung (fast-adaptation lineage)

The first rung — vanilla MAML — is itself the resolution of a line of few-shot learners. These
precede the ladder; the fixed substrate below is what they converged to.

- **Pretrain-and-fine-tune.** Pool data across tasks, train one network, and at test time take a few
  gradient steps on the new task's K examples. The trouble is the starting point: the initialization
  was chosen to minimize *average* pooled loss, which for a symmetric task family lands on a
  compromise (the mean target) that adaptation must fight its way out of. Gap: the initialization was
  never *built* to be a good launch point for fast adaptation.
- **Metric learners (Matching Networks, Vinyals et al. 2016; Prototypical Networks, Snell et al.
  2017).** Learn an embedding and classify a query by soft nearest-neighbor to the labeled support
  set, $\hat y=\sum_j a(g(\hat x),g(x_j))\,y_j$. Beautiful few-shot *classification* numbers, but
  "embed and compare against support points" is a classification construct — there is no prototype of
  a regression function or a control policy, and it produces no adapted *network*. Gap: not an
  adaptation rule for a parametric learner, and not architecture/task-form general.
- **Learned optimizers / Meta-Learner LSTM (Andrychowicz et al. 2016; Ravi & Larochelle 2017).**
  Replace gradient descent with a recurrent network that reads the gradient and emits the update,
  meta-learning init, direction, and rate together. The right ambition, but the recurrence costs
  backprop-through-time memory over the unrolled inner loop that does not scale to a conv learner, and
  it is hard to train. Gap: the recurrence — not the ambition — makes it unscalable.

## The fixed substrate

The two-loop meta-learner is frozen and must not be touched. The backbone is **CNN4** (four conv
blocks — Conv → BatchNorm → ReLU → MaxPool — at `hidden_size=64`, feature dim $64\cdot5\cdot5=1600$,
then a linear classifier to `N_WAY=5`), built by `make_model`. Meta-training runs **60,000**
iterations at 1-shot (15,000 at 5-shot), **4 tasks per meta-batch**, with the outer optimizer
`Adam(meta_lr=0.003)` over *all* meta-learnable parameters — the model's weights **plus** whatever
the inner-loop optimizer exposes. The inner loop takes **5 steps during training, 10 during
evaluation**, with a default `inner_lr=0.5`. Each task is cloned with `l2l.clone_module`, split into a
support set (K examples per class) and a 15-shot query set; the inner loop adapts on support, the
meta-loss is the query cross-entropy of the adapted clone, summed over the meta-batch. The loop also
provides the differentiable update primitives a rule may use: `l2l.algorithms.maml.maml_update(model,
lr, grads)` (one differentiable SGD step over every parameter) and `l2l.update_module(model,
updates=[...])` (a differentiable per-parameter `p ← p + u` swap, one entry per parameter in order).
Both replace the parameter *objects* each call — a fresh tensor per step — which is load-bearing
plumbing for any rule that re-reads parameters across inner steps.

## The editable interface

Exactly one region is editable — the `InnerLoopOptimizer` class in `learn2learn/custom_maml.py`
(lines 177–254). Every method on the ladder is a fill of this same three-method contract:
`__init__(model, inner_lr)` (inspect parameter shapes, **create any learnable optimizer parameters
here**), `adapt(model, support_x, support_y, n_steps)` (perform the inner adaptation on a *clone* —
must use `torch.autograd.grad` with the graph kept so the outer loop can backprop through it, never
`torch.optim`; return the adapted model), and `meta_parameters()` (the optimizer's own learnable
state handed to the outer Adam — `[]` if it has none). A parameter-count budget caps the optimizer's
extra learnable state at roughly the model size (one scalar per parameter, Meta-SGD's footprint).

The starting point is the scaffold default, which **is** vanilla MAML: differentiable SGD with a
fixed scalar learning rate applied to *every* parameter, and no learnable optimizer state. Each later
method replaces exactly this class and nothing else.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — default fill (vanilla MAML)
class InnerLoopOptimizer:
    """Inner-loop adaptation: differentiable SGD, fixed scalar LR, all parameters."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        # inner_lr: default inner-loop learning rate. No learnable optimizer state.
        self.inner_lr = inner_lr

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE (via l2l.clone_module) — safe to modify in-place.
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(            # differentiable; graph kept
                loss, model.parameters(), create_graph=True
            )
            model = l2l.algorithms.maml.maml_update( # one SGD step over ALL params
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []                                   # fixed LR => no meta-parameters
```

## Evaluation settings

Three few-shot benchmarks spanning the difficulty range — **miniImageNet 5-way 1-shot** (the hard,
60k-iteration setting where one support example per class makes full-network adaptation most fragile),
**miniImageNet 5-way 5-shot**, and **CIFAR-FS 5-way 5-shot** — each over three seeds {42, 123, 456}.
The metric is **mean classification accuracy over 600 test episodes** (higher is better on all three),
each episode adapting a fresh clone with 10 inner steps on the support set and scoring on a held-out
15-shot query set.
