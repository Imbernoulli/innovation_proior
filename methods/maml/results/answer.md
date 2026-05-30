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
= \left(I - \alpha\, \nabla_\theta^2\, \mathcal{L}_{\mathcal{T}_i}^{\text{sup}}(f_\theta)\right)
\nabla_{\theta'}\mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}).$$
The Hessian is of the **support** loss at $\theta$; the gradient is of the
**query** loss at the adapted point $\theta_i'$. It is a Hessian-vector product —
one extra backward pass, no Hessian materialized — and is produced automatically
by differentiating through an inner step that is kept in the autodiff graph.

**Pseudocode.** Random init $\theta$; while not done: sample tasks
$\mathcal{T}_i\sim p(\mathcal{T})$; for each, compute $\theta_i'$ from a $K$-shot
support set, sample a query set; update
$\theta \leftarrow \theta - \beta\nabla_\theta\sum_i \mathcal{L}^{\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'})$.

**First-order MAML (FOMAML).** Drop the curvature term
($\partial\theta_i'/\partial\theta \approx I$):
$$\nabla_\theta\, \mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}) \approx \nabla_{\theta'}\mathcal{L}_{\mathcal{T}_i}^{\text{qry}}(f_{\theta_i'}).$$
Still evaluate the query gradient at the **post-update** $\theta_i'$ (evaluating
at $\theta$ would reduce to ordinary joint pretraining). Because ReLU networks
are locally near-linear, the support-loss Hessian is small, so $(I-\alpha H)\approx I$
and accuracy is nearly unchanged while one backward pass is saved.

**Domains.** Supervised: $\mathcal{L}$ is MSE (regression) or cross-entropy
(classification), $H=1$. Reinforcement learning: $\mathcal{L}$ is negative
expected return; both the inner and meta gradients use the policy-gradient
(REINFORCE) estimator, the outer step uses a trust region (TRPO), and
Hessian-vector products use finite differences to avoid third derivatives.

## Code

The one model requirement is a *functional forward* that runs on an explicitly
supplied weight dict, so adaptation can produce a fresh $\theta_i'$ without
mutating $\theta$. Faithful to the canonical implementation (`cbfinn/maml`): a
manual weight dict, an inner step `fast_weights = weights - alpha * grad`, the
meta-loss evaluated through `forward(query, fast_weights)`, and a
`first_order` flag that detaches the inner gradient (the `stop_gradient` path in
the original).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Learner(nn.Module):
    """A plain GD-trainable model with a functional forward over a weight dict."""
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


def mse(pred, y):
    return ((pred - y) ** 2).mean()


def inner_adapt(model, loss_fn, x_s, y_s, alpha, n_steps, first_order=False):
    """theta'_i via n_steps of gradient descent on the support set."""
    fast_weights = model.init_weights()
    for _ in range(n_steps):
        loss_sup = loss_fn(model.functional_forward(x_s, fast_weights), y_s)
        # create_graph=True keeps the inner step in the graph so the meta-grad
        # picks up (I - alpha * H); first_order detaches to drop the Hessian.
        grads = torch.autograd.grad(
            loss_sup, fast_weights.values(), create_graph=not first_order)
        fast_weights = {
            name: w - alpha * (g.detach() if first_order else g)
            for (name, w), g in zip(fast_weights.items(), grads)
        }
    return fast_weights


def maml_step(model, loss_fn, meta_opt, tasks, alpha,
              n_steps=1, first_order=False, grad_clip=None):
    """One outer meta-update over a batch of tasks.
       tasks: list of (x_s, y_s, x_q, y_q)."""
    meta_opt.zero_grad()
    meta_loss = 0.0
    for x_s, y_s, x_q, y_q in tasks:
        fast_weights = inner_adapt(model, loss_fn, x_s, y_s,
                                   alpha, n_steps, first_order)
        q_pred = model.functional_forward(x_q, fast_weights)  # eval at theta'_i
        meta_loss = meta_loss + loss_fn(q_pred, y_q)          # query = meta signal
    meta_loss = meta_loss / len(tasks)
    meta_loss.backward()  # yields (I - alpha*H) . grad_{theta'} L^qry, summed
    if grad_clip is not None:                                  # used on MiniImagenet
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    meta_opt.step()       # theta <- theta - beta * meta-grad (Adam supplies beta)
    return meta_loss.item()


# --- meta-training loop (sinusoid regression instantiation) ---
def meta_train(model, sample_tasks, steps=70000, alpha=0.01,
               meta_lr=1e-3, n_steps=1, first_order=False):
    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)  # beta
    for _ in range(steps):
        tasks = sample_tasks()  # each: (x_support, y_support, x_query, y_query)
        maml_step(model, mse, meta_opt, tasks, alpha,
                  n_steps=n_steps, first_order=first_order)
    return model
```

Notes on fidelity to the canonical implementation: weights are an explicit dict
(not stored module state) so the inner step can produce $\theta_i'$ functionally;
the inner update is `w - update_lr * grad`; the meta-loss is the query loss
through the adapted weights; `first_order` mirrors the original `stop_grad` flag
(detach the inner gradient → FOMAML); Adam is the supervised meta-optimizer; and
gradient clipping is applied for the deeper MiniImagenet conv model. For
classification, swap `mse` for cross-entropy and `Learner` for the four-block
conv stack (conv → batch norm → ReLU → $2\times2$ pool); for RL, replace the
supervised loss with the policy-gradient surrogate and the Adam outer step with
a trust-region update.
