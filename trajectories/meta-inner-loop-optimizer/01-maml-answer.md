**Problem.** Few-shot meta-learning: adapt a CNN4 to a 5-way task from 1 or 5 examples per class,
generalizing to held-out query points, after meta-training across a task distribution. The inner-loop
rule is the design surface; the floor is the simplest rule that adapts at all.

**Key idea (the floor).** Do not learn an optimizer or a metric — learn an **initialization**. Choose
$\theta$ so that a few *ordinary* gradient-descent steps on a new task's support set yield a model
that generalizes on that task; the post-adaptation *query* loss is the training signal for $\theta$.
The inner step is plain differentiable SGD over **all** parameters at a fixed scalar rate,
$\theta_i' = \theta - \alpha\,\nabla_\theta \mathcal{L}^{\text{sup}}_{\mathcal{T}_i}(f_\theta)$;
unrolling $n$ steps and backpropagating the query loss through them (with the inner gradient kept in
the graph, `create_graph=True`) gives the meta-gradient $(I-\alpha\nabla_\theta^2\mathcal{L}^{\text{
sup}})^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}$ — a Hessian-vector product, no Hessian
materialized. No learnable optimizer state, so `meta_parameters()` is empty.

**Why it is the floor.** The only thing meta-learned is *where to start*. The update direction is
locked to the raw gradient, the magnitude is one global scalar, and the same rate hits every parameter
— all four conv blocks, BatchNorm, and the head alike. Anything the task family knows about *how* to
move (per-coordinate scale, off-gradient direction) cannot be expressed. A good launch point, then a
rigid step.

**Hyperparameters (harness).** Outer `Adam(meta_lr=0.003)` over the model weights only; 5 inner steps
train / 10 eval; CNN4 (`hidden_size=64`). The one harness-specific deviation from the bare default:
the global `inner_lr=0.5` destabilizes *full-network* adaptation at 1-shot (single example per class,
high-variance support gradient), so use the standard 1-shot recipe $\alpha=0.01$ when `N_SHOT==1` and
keep $\alpha=0.5$ at 5-shot (the larger support set buffers the noise; matches the learn2learn
default). This is a property of the benchmark's chosen rate, not of MAML the method.

**What to watch.** 5-shot benchmarks should be competitive (the larger support set forgives the rigid
step). 1-shot should be the soft spot even at $\alpha=0.01$, where a learned per-coordinate step
should later pull ahead — that gap is what forces structure into the step at rung 2.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — step 1: vanilla MAML
class InnerLoopOptimizer:
    """MAML inner-loop optimizer (Finn et al., 2017).

    Vanilla SGD with a fixed learning rate applied uniformly to all
    model parameters. This is the standard MAML inner loop.

    Shot-aware LR override: the global INNER_LR=0.5 destabilizes
    full-network adaptation at 1-shot in the local harness. At 5-shot the larger
    support set buffers gradient noise so 0.5 is fine (matches
    learn2learn benchmark default). Use the common 1-shot recipe
    (0.01) only when N_SHOT=1, keep 0.5 for 5-shot.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = 0.01 if N_SHOT == 1 else 0.5

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            model = l2l.algorithms.maml.maml_update(
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```
