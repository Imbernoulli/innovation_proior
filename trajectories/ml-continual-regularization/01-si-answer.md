**Problem.** A fixed-capacity net trains on contexts in sequence and forgets the old ones because
descending the current loss moves the shared weights that encoded them. The fix is a per-parameter
quadratic penalty anchored at each weight's boundary value, with stiffness equal to that weight's
*importance* for past contexts — and importance must be cheap, local, and constant in memory.

**Key idea.** Read importance off the *training trajectory* instead of an after-the-fact Fisher.
Over a context, each parameter's contribution to the loss drop is the per-coordinate line integral of
the gradient against the parameter's motion, `omega_k = -sum_steps g_k * delta_theta_k` — which is
exactly what the loop's `_custom_W` accumulator already maintains for free. Demanding a quadratic
surrogate that reproduces the same loss drop over the same motion forces dividing by the squared net
motion: importance `= W_k / (Delta_k^2 + epsilon)`, with `epsilon` (`model.epsilon`, default 0.1)
flooring the blow-up when a weight barely moved. The penalty is the no-half quadratic
`sum_k Omega_k (theta_k - theta_tilde_k)^2`.

**Why it starts the ladder.** It costs nothing extra over the loop the harness already runs (the
default Fisher fill needs a separate post-context backward-per-class sweep; this reuses gradients
training computed). On a quadratic loss the `Delta^2` normalization recovers the right Hessian
curvature — and unlike the endpoint Fisher, which is zero at a minimum, the path integral captured
curvature along the way.

**Hyperparameters.** `epsilon = model.epsilon` (0.1) damping; strength `c` = the per-benchmark
`reg_strength` (`CONFIG_OVERRIDES` left empty, scale 1.0). Cross-context summation is done by the loop,
so `estimate_importance` returns the raw per-context increment.

```python
# EDITABLE region of custom_regularization.py — step 1: SI (path-integral importance)
def estimate_importance(model, dataset, prev_params, device):
    """SI: Compute omega from accumulated path integral W and parameter changes.

    omega_k = W_k / (delta_k^2 + epsilon)

    where W_k is the accumulated per-step gradient-weighted parameter change
    (tracked in model._custom_W by the training loop) and delta_k is the
    total parameter change over the context.
    """
    epsilon = getattr(model, 'epsilon', 0.1)
    omega = {}

    # Get the accumulated W from the per-step tracking
    W = getattr(model, '_custom_W', {})

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_current = p.detach().clone()
                p_prev = prev_params.get(n, p_current)
                p_change = p_current - p_prev
                w_val = W.get(n, torch.zeros_like(p_current))
                omega[n] = w_val / (p_change ** 2 + epsilon)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """SI: sum(omega * (param - prev_param)^2)."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((omega * (p - prev) ** 2).sum())
    if losses:
        return sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
CONFIG_OVERRIDES = {}
```
