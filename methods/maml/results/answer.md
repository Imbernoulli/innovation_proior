# Model-Agnostic Meta-Learning (MAML)

## Problem

Few-shot learning: adapt a high-capacity model to a new task from only $K$
examples ($K=1$ or $5$), generalizing to held-out data of that task, after
meta-training on a distribution of related tasks $p(\mathcal{T})$. Do it without
extra learned parameters, without constraining the architecture, and across task
forms (classification, regression, reinforcement learning).

## Key idea

Don't learn an optimizer or a metric — learn an **initialization**. Choose the
model's initial parameters $\theta$ so that *a few ordinary gradient-descent
steps* on a new task's small support set yield a model that generalizes on that
task. The post-adaptation loss on a held-out **query** set is the training signal
for $\theta$. Adaptation at test time is plain fine-tuning, so it can use any
number of steps and any amount of data.

## Algorithm

Inner adaptation on task $\mathcal{T}_i$ (one step; multiple steps unroll the
same update on the support set):
$$\theta_i' = \theta - \alpha\, \nabla_\theta\, \mathcal{L}_{\mathcal{T}_i}^{\text{sup}}(f_\theta).$$

Meta-objective, evaluated on each task's query set:
$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{\text{qry}}\!\left(f_{\theta_i'}\right),
\qquad
\theta \leftarrow \theta - \beta\, \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}).$$

Meta-gradient (chain rule through the inner step), per task:
$$\nabla_\theta\, \mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'})
= \left(I - \alpha\, \nabla_\theta^2\, \mathcal{L}_{\mathcal{T}_i}^{\text{sup}}(f_\theta)\right)^{\!\top}
\nabla_{\theta'}\mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}).$$
The Hessian is of the **support** loss at $\theta$; the gradient is of the
**query** loss at the adapted point $\theta_i'$. The transpose is the chain-rule
Jacobian transpose; for smooth scalar losses the Hessian factor is symmetric.
The product is computed as a Hessian-vector product, so no Hessian is
materialized, and autodiff produces it by differentiating through an inner step
kept in the graph.

**Pseudocode.** Random init $\theta$; while not done: sample tasks
$\mathcal{T}_i\sim p(\mathcal{T})$; for each, compute $\theta_i'$ from a $K$-shot
support set, sample a query set; update
$\theta \leftarrow \theta - \beta\nabla_\theta\sum_i \mathcal{L}^{\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'})$.

**First-order MAML (FOMAML).** Drop the curvature term
($\partial\theta_i'/\partial\theta \approx I$):
$$\nabla_\theta\, \mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}) \approx \nabla_{\theta'}\mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}).$$
Still evaluate the query gradient at the **post-update** $\theta_i'$ (evaluating
at $\theta$ would reduce to ordinary joint pretraining). The approximation is
reasonable when $\alpha H^\top v$ is small for
$v=\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, a condition often plausible for
locally near-linear ReLU networks but not guaranteed by ReLUs alone. It saves the
extra backward pass while keeping the post-adaptation query signal.

**Domains.** Supervised: $\mathcal{L}$ is MSE (regression) or cross-entropy
(classification), $H=1$. Reinforcement learning: if
$J_i(\phi)=\mathbb{E}_{\tau\sim\pi_\phi,q_i}[\sum_t r_i(x_t,a_t)]$, then
$\mathcal{L}_i(\phi)=-J_i(\phi)$ and the inner loss step is reward ascent,
$\theta_i'=\theta+\alpha\nabla_\theta J_i(\theta)$. Both inner and outer
gradients use the policy-gradient estimator, query trajectories are sampled
on-policy from the adapted policy, the outer step uses a trust region (TRPO),
and Hessian-vector products use finite differences to avoid third derivatives.

## Code

The one model requirement is a *functional forward* that runs on an explicitly
supplied weight dict, so adaptation can produce a fresh $\theta_i'$ without
mutating $\theta$. The code keeps the operational structure: a manual weight
dict, an inner step `fast_weights = weights - alpha * grad`, the meta-loss
evaluated through `functional_forward(query, fast_weights)`, and a
`first_order` flag that detaches the inner gradient.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Learner(nn.Module):
    def __init__(self, dim_in, hidden, dim_out):
        super().__init__()
        sizes = [dim_in] + hidden + [dim_out]
        self.params = nn.ParameterDict()
        for i in range(len(sizes) - 1):
            self.params[f"w{i}"] = nn.Parameter(
                torch.randn(sizes[i + 1], sizes[i]) * 0.01)
            self.params[f"b{i}"] = nn.Parameter(torch.zeros(sizes[i + 1]))
        self.n_layers = len(sizes) - 1

    def functional_forward(self, x, weights):
        h = x
        for i in range(self.n_layers):
            h = F.linear(h, weights[f"w{i}"], weights[f"b{i}"])
            if i < self.n_layers - 1:
                h = F.relu(h)
        return h

    def init_weights(self):
        return {k: v for k, v in self.params.items()}


def mse_loss(pred, y):
    return ((pred - y) ** 2).mean()


def adapt_parameters(model, loss_fn, x_s, y_s, alpha, n_steps, first_order=False):
    fast_weights = model.init_weights()
    for _ in range(n_steps):
        support_pred = model.functional_forward(x_s, fast_weights)
        support_loss = loss_fn(support_pred, y_s)
        # create_graph=True keeps the inner step in the graph so the meta-grad
        # includes the Hessian-vector term; first_order drops that path.
        grads = torch.autograd.grad(
            support_loss, fast_weights.values(), create_graph=not first_order)
        fast_weights = {
            name: w - alpha * (g.detach() if first_order else g)
            for (name, w), g in zip(fast_weights.items(), grads)
        }
    return fast_weights


def meta_update_step(model, loss_fn, meta_opt, tasks, alpha,
                     n_steps=1, first_order=False, grad_clip=None):
    meta_opt.zero_grad()
    meta_loss = 0.0
    for x_s, y_s, x_q, y_q in tasks:
        fast_weights = adapt_parameters(model, loss_fn, x_s, y_s,
                                        alpha, n_steps, first_order)
        query_pred = model.functional_forward(x_q, fast_weights)
        meta_loss = meta_loss + loss_fn(query_pred, y_q)
    meta_loss = meta_loss / len(tasks)
    meta_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    meta_opt.step()
    return meta_loss.detach().item()


def meta_train(model, sample_tasks, steps=70000, alpha=0.01,
               meta_lr=1e-3, n_steps=1, first_order=False):
    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    for _ in range(steps):
        tasks = sample_tasks()
        meta_update_step(model, mse_loss, meta_opt, tasks, alpha,
                         n_steps=n_steps, first_order=first_order)
    return model
```

Weights are explicit dictionaries so the inner step can produce $\theta_i'$
functionally; the inner update is `w - alpha * grad`; the meta-loss is the query
loss through the adapted weights; `first_order` detaches the inner gradient; Adam
drives the supervised outer loop; and optional gradient clipping is available for
deeper convolutional models. For classification, swap `mse_loss` for cross-entropy
and use the four-block conv stack (conv, batch norm, ReLU, pooling); for RL,
replace the supervised loss with the policy-gradient surrogate and the Adam
outer step with a trust-region update.
