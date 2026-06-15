**Problem.** MAML meta-learns only *where* adaptation starts; the inner step is a single shared scalar
rate over every parameter, with the direction locked to the raw gradient. At 1-shot that rigid step is
both weakest and shakiest — one rate must be small enough not to blow up the most sensitive coordinate,
so it under-moves the discriminative ones. Learn the *step*, not just the start.

**Key idea.** Of an optimizer's three ingredients — initialization, direction, rate — keep MAML's
learned initialization and add the cheapest learnable object that rescales *per coordinate* and tilts
the step *off* the gradient: a **diagonal preconditioner**, i.e. one learned learning-rate vector
$\alpha$ the same shape as $\theta$. The inner step is the elementwise product
$\theta_i' = \theta - \alpha \odot \nabla\mathcal{L}^{\text{sup}}_i(\theta)$. Both $\theta$ and
$\alpha$ are meta-trained jointly in the *same* query-loss-after-adaptation objective, by ordinary
backprop through the inner step (graph kept) — no recurrent optimizer, no BPTT over an optimizer.

**Why the diagonal.** For a nonzero gradient with $\ge2$ active coordinates, $\alpha\odot g$ stays
parallel to $g$ only if the active entries of $\alpha$ are equal; once they differ, the step *tilts*
off the gradient. So a single vector buys both a per-coordinate rate and a learned direction. A
*per-layer* scalar cannot move two weights within a block differently; a *full matrix* costs
$\dim\theta^2$ and reintroduces per-step cost. The diagonal is the unique middle: linear memory,
per-coordinate, off-gradient. Setting every entry of $\alpha$ to one constant recovers MAML exactly —
this strictly generalizes it.

**Hyperparameters (harness).** `lrs` is an `nn.ParameterList` of `ones_like(p) * inner_lr` (uniform
start at 0.5 = MAML-like, then meta-training pulls the entries apart). The inner step uses
`l2l.update_module(model, updates=[-lr*g ...])` — a *list* of per-parameter products, the one change
from MAML's scalar `maml_update`. `meta_parameters()` returns `list(self.lrs.parameters())`, so the
outer `Adam(0.003)` optimizes $\alpha$ alongside $\theta$; the extra footprint (one scalar per
parameter) is exactly the budget the harness reserves. 5 inner steps train / 10 eval.

**What to watch.** Biggest expected win at **1-shot**, where MAML's single rate was lowest and most
seed-dependent: mean above MAML's 0.4365 and a tighter spread. **5-shot** is the open question — MAML
was already strong there (the larger support set forgives a rigid step), so the extra meta-parameters
may not help and could overfit. If 5-shot stalls, the next axis is *which* parameters to adapt.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — step 2: Meta-SGD
class InnerLoopOptimizer:
    """Meta-SGD inner-loop optimizer (Li et al., 2017).

    Learns a per-parameter learning rate vector that is meta-optimized
    by the outer loop. Each model parameter gets a corresponding learnable
    learning rate tensor of the same shape, initialized to inner_lr.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Create per-parameter learnable learning rates
        self.lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            updates = [-lr * g for g, lr in zip(grads, self.lrs)]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return list(self.lrs.parameters())
```
