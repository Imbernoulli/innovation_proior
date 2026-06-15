**Problem.** SI's path-integral importance is unbounded and grows along the trajectory, so on the long
domain-incremental Permuted-MNIST sequence it both over-constrains the fixed-capacity net (later contexts
unlearnable) and stops holding the early ones (the per-context accuracies slide to chance). The estimator
itself has to change: a bounded, anchored, PSD per-parameter curvature instead of a running sum.

**Key idea.** Derive importance as the precision of an approximate posterior. Reading training as MAP,
all of a finished context's knowledge lives in `p(theta | D_A)`; Laplace-approximate it as a Gaussian at
the optimum `theta*_A`, whose precision is the Hessian of the negative log posterior. Replace the
intractable likelihood Hessian by the **diagonal Fisher information**
`F_i = E_x E_{y~p_theta}[(grad_i log p_theta(y|x))^2]`, which equals the expected Hessian of the NLL
(the squared-score identity), is computable from first-order gradients, and is PSD by construction. The
penalty is the Gaussian quadratic spring `0.5 * sum_i F_i (theta_i - theta*_i)^2`.

**Why it beats SI.** The Fisher is evaluated *at one endpoint* with bounded curvature — nothing
compounds with the number of steps in a context, and a sum of squares cannot go negative (no floor
needed). The `0.5` is the honest Gaussian factor (SI dropped its half via the `Delta^2` normalization).

**Harness note (vs. paper EWC).** Canonical EWC keeps a separate Fisher *and* anchor per past context.
Here the loop instead **sums** each returned Fisher into `_custom_importance` and re-snapshots
`_custom_prev_params` to the *latest* boundary — so the accumulated stiffness is anchored at the most
recent optimum, not at each context's own. Identical for two contexts; the divergence from the third on
is exactly what the next rung exploits.

**Hyperparameters.** Fisher over `min(len, 200)` single-example passes, eval mode, exact all-class inner
expectation weighted by `softmax(output)`. Strength `lambda` = the per-benchmark `reg_strength` (default,
`CONFIG_OVERRIDES` left empty). Cross-context accumulation done by the loop.

```python
# EDITABLE region of custom_regularization.py — step 2: EWC (diagonal Fisher + quadratic penalty)
def estimate_importance(model, dataset, prev_params, device):
    """EWC: Diagonal Fisher Information matrix via squared gradients."""
    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, y) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    model.train(mode=mode)
    return est_fisher


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """EWC: 0.5 * sum(fisher * (param - prev_param)^2)."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
