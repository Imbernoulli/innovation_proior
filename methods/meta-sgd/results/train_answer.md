The problem starts from what actually breaks. I have a high-capacity network and a new task with a handful of labeled examples, sometimes one. If I do the obvious thing — random initialization, run SGD on those few points until the support loss is small — I fit the points and generalize to nothing, because the gradient of the support loss is a great direction for *fitting the support set*, which in the few-shot regime is exactly the thing that overfits. And I am left with a pile of knobs to set by hand from almost no data: where to start, what learning rate, how many steps, when to stop. None of that is reliable when the data is this thin, and pooling many tasks into one pretrained model fails too, because when tasks demand contradictory outputs for the same input the average response is informative about the output range but about no single task. So learning from scratch is out. The leverage I do have is that I am never facing one task in isolation: there is a whole distribution $p(\mathcal{T})$ of related tasks I can meta-train on first, each with a support set to adapt from and a held-out query set to be scored on, and the unit of generalization is the task, not the datapoint. The real design question is what to *extract* from that distribution so that a new task takes only a few steps and does not overfit. An adaptation procedure for a parametric learner $f_\theta$ is fully specified by three things — where it starts (the initialization), which way it moves (the update direction), and how far (the learning rate) — and with five examples every one of the defaults (random start, follow the gradient, small hand-set rate) is a liability. So abstractly I want to learn all three from the task distribution; the only question is the machinery, because the machinery is where everything has gone wrong before.

The two existing answers stake out opposite corners and the tension between them is the whole problem. MAML keeps the inner loop trivial and meta-learns only the starting point: the inner update is plain SGD, $\theta'_i = \theta - \alpha\,\nabla L_{\mathcal{T}_i}(f_\theta)$ with $\alpha$ a hand-set scalar, and the elegance is that $\theta$ is trained not to fit any task but so that *after* adaptation the query loss is small across tasks, with the meta-update differentiating through the inner step (a gradient-through-a-gradient, i.e. a Hessian-vector product). It is architecture-agnostic, adds no parameters beyond $\theta$, and works for classification, regression, and policy gradients alike. But the only thing meta-learned is $\theta^0$. The way the learner moves on a new task is frozen: every coordinate steps along the raw gradient at one scalar rate that must be hand-tuned and can need wildly different values per problem and per step. A great starting point, a dumb step. The Meta-Learner LSTM goes the other way and meta-learns the entire update rule with a recurrent network, exploiting that an LSTM cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ becomes gradient descent when the cell state is the parameters, the candidate is $-\nabla L$, the input gate is a learned learning rate, and the forget gate is a learned shrink. That is the right ambition — init, direction, and rate learned together — but the recurrence is the problem: backprop-through-time over the unrolled inner loop must store every intermediate state at a cost on the order of $O(T \cdot \#\text{states} \cdot \dim(\theta))$, which simply does not fit a convolutional learner, and the one tiny LSTM shared across all coordinates pushes each parameter around independently, ignoring how they relate. So one corner is simple, scalable, and rigid; the other is expressive and unscalable. Crucially, the recurrence — not the ambition — is what costs the memory and the optimization pain.

I propose Meta-SGD. The move is to keep the prize, "learn the initialization, the update direction, and the learning rate," and drop the recurrence by replacing the network-that-emits-the-step with the cheapest learnable object that can still do what a scalar cannot. Sitting between the corners, the simple step is $-\alpha\,\nabla L$ with $\alpha$ a scalar; the most general linear thing is a full matrix preconditioner $-A\,\nabla L$, which can rotate the step arbitrarily off the gradient but costs $|\theta|^2$ to store and, if it varied per step, reintroduces exactly the per-step-per-coordinate cost that sank the LSTM. The unique middle ground is the diagonal: let $A = \mathrm{diag}(\alpha)$ with $\alpha$ a vector the size of $\theta$, so the inner adaptation on a task's support set is a single elementwise-scaled gradient step,
$$\theta' = \theta - \alpha \odot \nabla L_{\mathcal{T}}(\theta),$$
where $\odot$ is the Hadamard product. This buys both ingredients the scalar corner could not reach. Per-coordinate rate is immediate: entry $\alpha_j$ is coordinate $j$'s own learning rate, so different coordinates move at completely different scales. And the direction comes for free, which is the load-bearing observation: for a nonzero gradient with at least two active coordinates, $\alpha \odot \nabla L$ stays parallel to $\nabla L$ only when the active entries of $\alpha$ are all the same scalar; once those entries differ, the coordinate rescaling *tilts* the vector off the raw-gradient direction. So the norm of $\alpha \odot \nabla L$ is the effective step size and its orientation is the effective update direction, both rolled into one vector — an off-gradient step and a per-parameter rate without a matrix and without a recurrence, since $\alpha$ is just a tensor the same shape as $\theta$. The diagonal is the right rank precisely because it is the unique object that is linear in $\dim(\theta)$ to store and meta-learn, capable of a distinct rate for every coordinate, and capable of an off-gradient direction; a per-layer scalar is cheaper but cannot move two weights within a layer differently, which is where the interesting structure lives, and the full matrix is more expressive but costs $|\theta|^2$.

The other decision is where $\alpha$ comes from. The hand-designed adaptive optimizers — AdaGrad, RMSProp, Adam — also put a per-coordinate rescaling on the gradient, which tells me the *shape* is right, but they read it from a gradient history by a fixed formula, and in one-step few-shot adaptation there is no history to accumulate. What I have instead is the task distribution, so I *learn* $\alpha$ as a free vector of meta-parameters, tuned across tasks so that one step of $-\alpha \odot \nabla L$ lands somewhere that generalizes; $\alpha$ becomes a learned encoding of how far and which combined way each coordinate should move when the support gradient points a given way. Because $\alpha$ is differentiable, I fold it straight into the same query-loss-after-adaptation objective and meta-train $\theta$ and $\alpha$ jointly,
$$\min_{\theta,\,\alpha}\ \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\big[\,L_{\text{test}(\mathcal{T})}\big(\theta - \alpha \odot \nabla L_{\text{train}(\mathcal{T})}(\theta)\big)\,\big].$$
For one task, writing $g = \nabla_\theta L_{\text{train}}(\theta)$, $H = \nabla^2_\theta L_{\text{train}}(\theta)$, $\theta' = \theta - \alpha \odot g$, and $v = \nabla_{\theta'} L_{\text{test}}(\theta')$, the chain rule gives
$$\frac{\partial L_{\text{test}}(\theta')}{\partial \alpha} = -\,v \odot g, \qquad \frac{\partial L_{\text{test}}(\theta')}{\partial \theta} = \big(I - \mathrm{diag}(\alpha)\,H\big)^{\!\top} v.$$
The $\theta$-gradient is the same through-the-inner-gradient Hessian-vector path as in MAML — the cost of asking how the post-adaptation query loss responds to perturbing the pre-adaptation initialization — while the $\alpha$-gradient is simpler because $\alpha$ enters $\theta'$ linearly through the elementwise product. The whole thing is optimized by ordinary SGD over task batches, $(\theta,\alpha) \leftarrow (\theta,\alpha) - \beta\,\nabla_{(\theta,\alpha)} \sum_i L_{\text{test}}(\theta'_i)$, with Adam the practical choice for the meta-update; a first-order variant detaches $g$, keeps the identity path through $\theta$, and drops the $\mathrm{diag}(\alpha)H$ term. There is no BPTT over an LSTM, no stored recurrent states, no coordinate-sharing hack — the thing that made the recurrent route unscalable is simply gone, replaced by a static learned vector and an elementwise multiply, so the memory stays linear.

Two construction details follow. I initialize every entry of $\alpha$ to a common small constant (around the scalar rate I would have hand-picked), so meta-training begins from behavior I trust — MAML-like uniform steps — and pulls the entries apart as it discovers which coordinates want bigger or differently-signed steps. And I want as few inner steps as possible: the point of learning $\alpha$ is that one step can carry the structure a rigid scalar rule would need several careful steps to approximate, but if I do iterate, I reuse the same learned $\alpha$ at each step (it is a property of the optimizer, not of the step index) and keep the graph differentiable across steps so the outer gradient still flows. This is the right generalization rather than a third unrelated gadget: freeze $\alpha$ to a single constant in every coordinate and stop learning it, and $\theta' = \theta - \alpha\,\nabla L$ is exactly MAML; replace the static vector with recurrent gates and I am back to the expensive LSTM. Meta-SGD sits exactly between — strictly more capacity than the scalar-rate method, far cheaper than the recurrent one — and drops into the standard meta-learning harness as one extra vector of meta-parameters.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5  # initial value for every entry of alpha


class InnerLoopOptimizer(nn.Module):
    """Meta-SGD inner-loop optimizer: a learned per-parameter learning-rate vector
    alpha (the `lrs`), meta-optimized by the outer loop jointly with the learner
    initialization. Inner step:  theta <- theta - alpha (.) grad   (elementwise)."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR,
                 first_order: bool = False):
        super().__init__()
        self.inner_lr = inner_lr
        self.first_order = first_order
        # one learnable LR tensor per parameter, same shape, initialized uniform.
        self.lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        second_order = not self.first_order
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            # second_order keeps the Hessian-vector path through the inner gradient.
            grads = torch.autograd.grad(
                loss,
                model.parameters(),
                retain_graph=second_order,
                create_graph=second_order,
            )
            model = self._apply_update(model, grads)
        return model

    def _apply_update(self, model: nn.Module, grads: List[Tensor]) -> nn.Module:
        """Differentiable p <- p - lr * grad update using learnable rates."""
        for p, lr, g in zip(model.parameters(), self.lrs, grads):
            p.grad = g
            p._lr = lr

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and p.grad is not None:
                    module._parameters[name] = p - p._lr * p.grad
                    p.grad = None
                    p._lr = None

            for name in module._buffers:
                buff = module._buffers[name]
                if buff is not None and buff.grad is not None and getattr(buff, "_lr", None) is not None:
                    module._buffers[name] = buff - buff._lr * buff.grad
                    buff.grad = None
                    buff._lr = None

            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        # alpha is meta-optimized alongside the initialization.
        return list(self.lrs.parameters())
```
