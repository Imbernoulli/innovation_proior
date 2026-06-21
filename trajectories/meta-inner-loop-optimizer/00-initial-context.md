## Research question

Gradient-based meta-learning learns an initialization that adapts to a new task from a handful of labeled examples through a few gradient steps. The design object is the **inner-loop adaptation rule** — which parameters change during adaptation, how the support gradient is scaled or transformed before it is applied, and what state (if any) is carried across inner steps and meta-learned by the outer loop. Everything else — the data loader, the CNN4 backbone, the meta-batch schedule, the number of inner steps, the outer optimizer — is fixed. The contribution is the adaptation rule itself.

## Prior art / Background / Baselines

Existing few-shot learners leave different gaps:

- **Pretrain-and-fine-tune.** Pool data across tasks, train one network, and at test time take a few gradient steps on the new task's support set. Gap: the initialization minimizes average pooled loss, which for structured task families lands on a compromise that adaptation must escape; it is not built as a launch point for fast adaptation.
- **Metric learners (Matching Networks, Prototypical Networks).** Learn an embedding and classify a query by soft nearest-neighbor to the support set. Gap: the "embed-and-compare" construction is tied to classification; it produces no adapted parametric network for regression, control, or other task forms.
- **Learned optimizers / Meta-Learner LSTM.** Replace gradient descent with a recurrent network that reads each gradient and emits the parameter update, meta-learning initialization, direction, and step size jointly. Gap: backpropagating through the unrolled recurrent inner loop is memory-hungry and unstable, so the approach has not scaled to standard convolutional few-shot learners.

## Fixed substrate / Code framework

The two-loop meta-learner is frozen and must not be touched. The backbone is **CNN4** (four Conv → BatchNorm → ReLU → MaxPool blocks at `hidden_size=64`, feature dim 1600, then a linear classifier to 5-way), built by `make_model`. Meta-training runs **60,000** iterations for 1-shot (**15,000** for 5-shot), **4 tasks per meta-batch**, with outer optimizer `Adam(meta_lr=0.003)` over all meta-learnable parameters — the model's weights plus whatever the inner-loop optimizer exposes. The inner loop takes **5 steps during training, 10 during evaluation**, with default `inner_lr=0.5`.

Each task is cloned with `l2l.clone_module`, split into a support set (K examples per class) and a 15-shot query set; the inner loop adapts on support, and the meta-loss is the query cross-entropy of the adapted clone, summed over the meta-batch. The framework also exposes differentiable update primitives: `l2l.algorithms.maml.maml_update(model, lr, grads)` (one differentiable SGD step over every parameter) and `l2l.update_module(model, updates=[...])` (a differentiable per-parameter `p ← p + u` swap, one entry per parameter in order). Both replace parameter objects each call, so rules that re-read parameters across steps must account for fresh tensors.

## Editable interface

Only the `InnerLoopOptimizer` class in `learn2learn/custom_maml.py` (lines 177–254) may change. The contract has three methods:

- `__init__(model, inner_lr)` — inspect parameter shapes and create any learnable optimizer parameters here.
- `adapt(model, support_x, support_y, n_steps)` — run the inner adaptation on a *clone*; must use `torch.autograd.grad` with the graph kept so the outer loop can backprop through it, and must never use `torch.optim`; return the adapted model.
- `meta_parameters()` — the optimizer's own learnable state handed to the outer Adam; return `[]` if it has none.

A parameter-count budget caps extra learnable state near the model size (about one scalar per parameter). The default fill is vanilla MAML: differentiable SGD with a fixed scalar learning rate applied to every parameter and no learnable optimizer state. Later fills replace only this class.

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

Three few-shot benchmarks: **miniImageNet 5-way 1-shot**, **miniImageNet 5-way 5-shot**, and **CIFAR-FS 5-way 5-shot**. Each is run over seeds `{42, 123, 456}`. The metric is **mean classification accuracy over 600 test episodes** (higher is better), with each episode adapting a fresh clone with 10 inner steps on the support set and scoring on a held-out 15-shot query set.
