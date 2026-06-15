# Elastic Weight Consolidation (EWC), distilled

EWC is a regularization method for continual learning that prevents catastrophic forgetting by
adding a per-weight quadratic penalty to the loss of each new task. The penalty anchors each
weight to its value after the previous task and weights the anchor by that weight's *importance*
to the previous task — the diagonal of the Fisher information matrix. Important weights are held
nearly rigid; unimportant ones stay free to learn the new task. It uses a single fixed-capacity
network, no stored old data, and is linear in the number of parameters and training examples.

## Problem it solves

Train one network on a sequence of tasks A, B, C, ... with no access to a task's data once it is
finished, while retaining performance on the earlier tasks. Plain sequential SGD overwrites the
weights that mattered to old tasks (catastrophic forgetting); a uniform L2 anchor to the old
weights cannot distinguish important weights from free ones, so it either freezes the network or
fails to protect the old task; replaying old data costs memory that grows with the number of
tasks.

## Key idea

Read training as MAP estimation, `log p(theta|D) = log p(D|theta) + log p(theta) - log p(D)`,
where the log-likelihood is the negative loss. Split the data into the finished task A and the
current task B and apply Bayes again:

```
log p(theta | D_A, D_B) = log p(D_B | theta) + log p(theta | D_A) - const.
```

All knowledge of task A is contained in the posterior `p(theta | D_A)`. Approximate it (Laplace
approximation) by a Gaussian centered at the task-A optimum `theta*_A`: since the gradient
vanishes at the optimum, the second-order Taylor expansion of `-log p(theta|D_A)` has no linear
term, giving a Gaussian whose precision is the Hessian of the old negative log posterior at
`theta*_A`. Its data-likelihood curvature is the task-specific part used for per-weight
importance; prior curvature can be dropped for a broad prior or absorbed into the scalar
regularization strength.

Replace that intractable likelihood curvature by the **Fisher information** `F`, whose
model-distribution form equals the expected Hessian of the NLL, is computable from first-order
gradients, and is positive semi-definite by construction. Keep only its diagonal (factorized
Gaussian). The objective for training task B is then

```
L(theta) = L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_{A,i})^2,
```

a quadratic "spring" anchoring each weight to `theta*_A` with per-weight stiffness `F_i`. The
`1/2` is the Gaussian quadratic-form factor; `lambda` trades old vs. new task (in the exact
derivation it absorbs the task-A sample size `N`, which scales the Fisher, plus the
overconfidence of a diagonal point-estimate Laplace approximation, and is set by search).

## Why the Fisher equals the curvature (the load-bearing identity)

For `p = p_theta(y|x)`, with the expectation over `y ~ p_theta`:

```
E[ d^2 log p / dtheta^2 ] = sum_y p * [ (1/p) d^2 p/dtheta^2 - (d log p/dtheta)^2 ]
                          = d^2/dtheta^2 (sum_y p) - E[(d log p/dtheta)^2]
                          = 0 - E[(d log p/dtheta)^2],
```

since `sum_y p = 1` for all theta. Hence `E[-d^2 log p/dtheta^2] = E[(d log p/dtheta)^2] = F`:
under the model distribution, the expected Hessian of the NLL equals the expected squared score
(the Fisher), so the likelihood curvature proxy is an average of squared first-order gradients —
no second derivatives, and PSD because it is a sum of squares.

## Computing the diagonal Fisher

`F_i = E_x E_{y ~ p_theta(y|x)} [ (d log p_theta(y|x)/d theta_i)^2 ]` at `theta*_A`. Average over
100-200 sampled inputs; for each input, take the exact inner expectation over the model's *own*
predictive distribution by summing the squared per-class score weighted by the predicted class
probability `p_k`. Using `y ~ p_theta` (not the data labels) is what makes `F` the expected
Hessian. Using a supplied label gives a label-based empirical Fisher; compact implementations
often take that label to be the model's argmax class, which is a deterministic pseudo-label
shortcut rather than the probability-weighted expected Fisher. Sampling a label from `p_theta`
would be a Monte Carlo estimate of the expected Fisher; argmax is not.

## Multiple tasks

For task C after A and B, keep one quadratic penalty per past task and sum them (the sum of
quadratics is a quadratic), each anchored at the optimum reached when that task finished and
weighted by that task's stored diagonal Fisher; memory then grows with the number of tasks.

## Working code

Two functions filling the continual-learning regularization harness — `summarize_finished_context`
(called once after each context: store the anchor and diagonal Fisher) and `extra_training_loss`
(called each step: sum the quadratic records and scale by `lambda`):

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def summarize_finished_context(model, dataset, device):
    """EWC: diagonal Fisher information via probability-weighted squared gradients.

    F_i = E_x E_{y ~ p_theta(y|x)} [ (grad_i log p_theta(y|x))^2 ] at the optimum:
    the model-distribution expected Hessian of the NLL -> per-weight curvature.
    """
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    anchor = {n: p.detach().clone() for n, p in params.items()}
    est_fisher = {n: torch.zeros_like(p) for n, p in params.items()}

    mode = model.training
    model.eval()                                       # no dropout/BN noise in the estimate

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)             # average over a sample of inputs

    for idx, (x, _) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1).detach()
        for label_index in range(output.shape[1]):     # exact y-expectation: sum over classes
            label = torch.tensor([label_index], device=device, dtype=torch.long)
            negloglikelihood = F.cross_entropy(output, label)   # -log p_theta(y=k|x)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for n, p in params.items():
                if p.grad is not None:
                    est_fisher[n] += label_weights[0, label_index] * p.grad.detach().pow(2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    model.train(mode=mode)
    return {"fisher": est_fisher, "anchor": anchor}


def extra_training_loss(model, summary_state, strength=1.0):
    """EWC penalty: 0.5 * sum_i F_i (theta_i - theta*_i)^2.

    Sum one quadratic record per past context. `strength` is lambda unless a
    record carries its own value. Cheap: no data pass.
    """
    device = next(model.parameters()).device
    if not summary_state:
        return torch.zeros((), device=device)

    records = summary_state if isinstance(summary_state, (list, tuple)) else [summary_state]
    total = torch.zeros((), device=device)
    for record in records:
        fisher = record["fisher"]
        anchor = record["anchor"]
        scale = record.get("strength", strength)
        record_total = torch.zeros((), device=device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in fisher:
                record_total = record_total + (fisher[n] * (p - anchor[n]) ** 2).sum()
        total = total + 0.5 * scale * record_total
    return total
```

Compact PyTorch structure (per-task Fisher and anchor stored once, penalty summed each step).
This mirrors the local compact implementation pattern: it uses the model's argmax class as a
pseudo-label shortcut rather than the exact probability-weighted expected Fisher above:

```python
from copy import deepcopy
import torch
from torch.nn import functional as F


class EWC:
    """Stores theta*_A (anchor) and the diagonal Fisher for a finished task."""

    def __init__(self, model, dataset):
        self.model = model
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._anchor = {n: p.detach().clone() for n, p in self.params.items()}  # theta*_A
        self._fisher = self._diag_fisher(dataset)                               # F_i

    def _diag_fisher(self, dataset):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        for x in dataset:
            self.model.zero_grad()
            output = self.model(x).view(1, -1)
            label = output.max(1)[1].view(-1)              # deterministic argmax pseudo-label
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 / len(dataset)
        return fisher

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._anchor[n]) ** 2).sum()
        return loss  # caller scales by importance; the 1/2 factor is absorbed there
```
