Training a single network on a sequence of tasks with ordinary SGD causes catastrophic forgetting: the gradient for each new task freely overwrites the weights that encoded the previous ones, so performance on earlier tasks collapses. The obvious fixes are either ruled out by the problem statement or fail in practice. Interleaving old and new data would remove forgetting entirely, but it requires storing and replaying an ever-growing episodic memory, which violates the constraint that the stored record must stay compact. A uniform L2 anchor pulling every weight back toward its old value is compact, yet it has only one global stiffness: set it high enough to preserve the old task and the whole network freezes, leaving no capacity for the new task; set it low enough to learn the new task and the old one is lost. Dropout and other standard regularizers help only by effectively enlarging capacity, not by protecting specific knowledge. Even path-integral importance estimators can grow unbounded along a trajectory and over-constrain the network. What is needed is a per-weight measure of importance that is bounded, positive, and cheap to compute, so that some weights can be held rigid while others remain free.

The method I propose is Elastic Weight Consolidation (EWC). It reads sequential learning as Bayesian inference. After training on task A, all knowledge of A is summarized by the posterior p(theta | D_A). When task B arrives, the target posterior is p(theta | D_A, D_B) ∝ p(D_B | theta) p(theta | D_A), so fitting B while preserving A means keeping the term log p(theta | D_A) in the objective. That posterior is intractable, but near the task-A optimum theta*_A a Laplace approximation replaces it by a Gaussian centered at theta*_A whose precision is the Hessian of the negative log posterior there. The data-likelihood part of that Hessian is the curvature we need for per-parameter importance. Rather than forming the full Hessian, which is infeasible for a deep network, EWC uses the Fisher information matrix F, which equals the expected Hessian of the negative log-likelihood under the model distribution and can be estimated from first-order gradients alone. Keeping only the diagonal gives a factorized Gaussian approximation and a scalar importance F_i for every weight. Important weights, with large F_i, are anchored strongly to their old values; unimportant weights are left loose. The objective for the new task is L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_{A,i})^2, a quadratic spring with per-coordinate stiffness. For a third, fourth, and later task, one keeps a separate anchor theta*_t and diagonal Fisher F_t for each past task and sums the corresponding penalties.

The key identity making this practical is that the expected Hessian of the negative log-likelihood equals the expected outer product of the score under the model's predictive distribution: E[-∇^2 log p] = E[(∇ log p)(∇ log p)^T] = F. The derivation is a single line: differentiate log p twice, take expectation over y ~ p_theta, and use that sum_y p = 1 so the second-derivative-of-the-density term integrates to zero. This means curvature can be estimated by squared gradients, avoiding second derivatives, and the outer-product form is positive semi-definite by construction.

The diagonal Fisher is computed by sampling a small set of inputs from the finished task, running a forward pass for each, and taking the exact expectation over classes by summing the squared per-class gradient weighted by the predicted class probability. This is more faithful than using the data label or the argmax label. The per-step penalty is then just an elementwise operation over the current parameters, the stored anchors, and the stored diagonal Fishers.

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
    model.eval()  # no dropout/BN noise in the estimate

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)  # average over a sample of inputs

    for idx, (x, _) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1).detach()
        for label_index in range(output.shape[1]):  # exact y-expectation: sum over classes
            label = torch.tensor([label_index], device=device, dtype=torch.long)
            negloglikelihood = F.cross_entropy(output, label)  # -log p_theta(y=k|x)
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
