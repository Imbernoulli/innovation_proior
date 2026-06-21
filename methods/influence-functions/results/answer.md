# Influence functions for black-box predictions

## Problem

Attribute a model's test-time prediction back to the training data: for a training point `z` and a test point `z_test`, estimate how the prediction (the test loss) would change if `z` were removed, or if its input were perturbed — without retraining the model. The gold standard, leave-one-out retraining, costs one full retraining per training point and is infeasible for deep networks.

## Key idea

Reframe "remove a training point" as the smooth counterfactual "upweight it by ε," with deletion corresponding to ε = −1/n. Differentiating the empirical-risk-minimizer's optimality condition gives, in closed form, how the parameters and any test-point loss respond to that upweighting. The response involves the inverse Hessian, which is intractable to form at scale — but it is only ever needed as its action on a vector, so it is computed implicitly through Hessian-vector products and reused across all training points. The same approximation, computed on smoothed losses or with a damped Hessian, remains useful even when the model is non-convex, non-converged, or non-differentiable.

## Final form

Let `θ̂ = argmin_θ (1/n) Σ_i L(z_i, θ)` with empirical-risk Hessian `H_θ̂ = (1/n) Σ_i ∇²_θ L(z_i, θ̂)` (assumed PD in the convex case).

**Influence of upweighting `z` on the parameters** (classic M-estimation / influence-function result):
```
I_up,params(z) = dθ̂_{ε,z}/dε |_{ε=0} = − H_θ̂^{-1} ∇_θ L(z, θ̂)
```
This is one Newton step on the quadratic model of the risk, in the direction of `z`'s gradient. Deletion: `θ̂_{-z} − θ̂ ≈ −(1/n) I_up,params(z)`.

**Influence of upweighting `z` on the loss at a test point** (chain rule):
```
I_up,loss(z, z_test) = − ∇_θ L(z_test, θ̂)^⊤ H_θ̂^{-1} ∇_θ L(z, θ̂)
```
Estimated change in test loss from removing `z`: `≈ −(1/n) I_up,loss(z, z_test)`.

**Influence of perturbing a training input** `z = (x,y) ↦ (x+δ, y)`:
```
I_pert,loss(z, z_test) = − ∇_θ L(z_test, θ̂)^⊤ H_θ̂^{-1} ∇_x ∇_θ L(z, θ̂)   ∈ ℝ^d
```
so `I_pert,loss(z, z_test) · δ` is the first-order effect on the test loss; moving `δ` along `I_pert,loss^⊤` maximally raises it. The ε-mass-moving derivation holds for arbitrary (including discrete) `δ`.

**Why the two non-trivial factors matter** (logistic regression, `y∈{−1,1}`, `σ(t)=1/(1+e^{−t})`):
```
I_up,loss(z, z_test) = − y_test y · σ(−y_test θ^⊤x_test) · σ(−y θ^⊤x) · x_test^⊤ H_θ̂^{-1} x
```
versus the bare similarity `x·x_test`: the `σ(−yθ^⊤x)` factor up-weights high-loss outliers; `H_θ̂^{-1}` measures the other points' "resistance," amplifying gradients in low-variation directions and correcting the sign errors that pure input similarity makes (e.g. with `x ⪰ 0`, `x·x_test ≥ 0` falsely marks all same-label points helpful).

## Scaling it: never form H

All formulas need `H^{-1}` only as `H^{-1}v`. Define once per test point `s_test = H_θ̂^{-1} ∇_θ L(z_test, θ̂)`; then for every training point `I_up,loss(z, z_test) = − s_test · ∇_θ L(z, θ̂)` (a dot product), collapsing `n` inversions into one.

- **Hessian-vector product (Pearlmutter):** `Hv = ∇_θ(v · ∇_θ L)` — two reverse-mode passes, exact, `O(p)`, no explicit `H`, no finite differencing.
- **Solve route 1 — conjugate gradients:** since `H ≻ 0`, `H^{-1}v = argmin_t {½ t^⊤H t − v^⊤t}`; minimize with Newton-CG using only `Ht`.
- **Solve route 2 — LiSSA stochastic estimator:** from `H^{-1} = Σ_{i≥0}(I−H)^i` (valid when `0 ≺ H ⪯ I`), the unbiased recursion `H̃_0^{-1}v = v`, `H̃_j^{-1}v = v + (I − ∇²_θ L(z_{s_j}, θ̂)) H̃_{j-1}^{-1}v` with `z_{s_j}` sampled uniformly; run to depth `t`, average `r` runs. Enforce `∇²L ⪯ I` by scaling the loss down (the `scale` hyperparameter; rescale the result); add a damping `λI` to keep the curvature PD off the minimum.

Total cost to score all training points for one test point: `O(np + rtp)`, linear in `n` and `p`; empirically `rt = O(n)` suffices.

## Robustness to broken assumptions

- **Non-convex / non-converged `θ̃`:** use a damped convex quadratic model with `H_θ̃ + λI` (equivalently L2 regularization). A Newton step from `θ̃` after upweighting `z` decomposes as `−H_θ̃^{-1}g − ε H_θ̃^{-1}∇L(z,θ̃)`: a `z`-independent drift plus `ε · I_up,params(z)`, so relative influence is still tracked.
- **Non-differentiable loss (hinge):** the hinge's second derivative is zero, so it carries no margin information and influence overestimates. Swap in `SmoothHinge(s,t) = t·log(1 + exp((1−s)/t)) → Hinge(s)` as `t→0` only for influence computation; it restores a curvature that encodes closeness to the margin.

## Code (autodiff implementation)

Faithful to the canonical `influence-release` engine (HVP via double-backprop; CG via Newton-CG; LiSSA Neumann recursion with `scale`/`damping`/`num_samples`/`recursion_depth`; influence as the dot product `s_test · grad`, with the removal effect being `−(1/n) I_up,loss = (1/n) s_test · grad`), expressed with `torch.autograd`.

```python
import torch

def grad_params(scalar_loss, params, create_graph=False):
    """Gradient of a scalar loss w.r.t. params (one reverse pass)."""
    return list(torch.autograd.grad(scalar_loss, params, create_graph=create_graph))

def hvp(loss, params, v):
    """Hessian-vector product H v, exact and O(p): H v = grad( v . grad L ). Never forms H."""
    g = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((gi * vi.detach()).sum() for gi, vi in zip(g, v))
    return list(torch.autograd.grad(dot, params, retain_graph=True))

def _hvp_over_data(model, train_data, loss_fn, params, v, batch_size=None):
    """Empirical-risk HVP H v = (1/n) sum_i grad^2 L(z_i) v, accumulated over the dataset."""
    total, nb = [torch.zeros_like(p) for p in params], 0
    for xb, yb in train_data.batches(batch_size):
        part = hvp(loss_fn(model, (xb, yb)), params, v)
        total = [t + p for t, p in zip(total, part)]; nb += 1
    return [t / nb for t in total]

def inverse_hvp_cg(model, train_data, loss_fn, params, v, damping=0.0):
    """s = H^{-1} v via Newton-CG on  1/2 t^T H t - v^T t  (H > 0). Uses only H t."""
    import numpy as np
    from scipy.optimize import fmin_ncg
    shapes = [p.shape for p in params]; sizes = [p.numel() for p in params]
    def to_list(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(torch.tensor(x[i:i+n], dtype=params[0].dtype).reshape(s)); i += n
        return out
    def flat(ts): return np.concatenate([t.detach().cpu().numpy().ravel() for t in ts])
    def Hx(x):
        hv = _hvp_over_data(model, train_data, loss_fn, params, to_list(x))
        return flat([h + damping * xi for h, xi in zip(hv, to_list(x))])
    f      = lambda x: 0.5 * np.dot(Hx(x), x) - np.dot(flat(v), x)
    fprime = lambda x: Hx(x) - flat(v)
    res = fmin_ncg(f=f, x0=flat(v), fprime=fprime,
                   fhess_p=lambda x, p: Hx(p), avextol=1e-8, maxiter=100)
    return to_list(res)

def inverse_hvp_lissa(model, train_data, loss_fn, params, v,
                      scale=10.0, damping=0.0, num_samples=1, recursion_depth=5000,
                      batch_size=1):
    """s = H^{-1} v via the stochastic Neumann recursion (LiSSA).
       e_0 = v;  e_j = v + (I - H_sample/scale - damping I) e_{j-1};  return e_t / scale."""
    result = None
    for _ in range(num_samples):
        cur = [vi.clone() for vi in v]
        for _ in range(recursion_depth):
            xb, yb = train_data.sample_batch(batch_size)      # unbiased H sample
            Hcur = hvp(loss_fn(model, (xb, yb)), params, cur)
            cur = [vi + (1 - damping) * ci - hi / scale
                   for vi, ci, hi in zip(v, cur, Hcur)]
        contrib = [ci / scale for ci in cur]
        result = contrib if result is None else [r + c for r, c in zip(result, contrib)]
    return [r / num_samples for r in result]

def influence_on_test_loss(model, train_data, test_point, loss_fn, params,
                           approx="cg", **kw):
    """I_up,loss(z, z_test) = - s_test . grad L(z) for every training z. Removal effect ~ -(1/n) I."""
    v = grad_params(loss_fn(model, test_point), params)        # grad L(z_test)
    s_test = (inverse_hvp_cg if approx == "cg" else inverse_hvp_lissa)(
        model, train_data, loss_fn, params, v, **kw)
    out = []
    for z in train_data.each_example():
        gz = grad_params(loss_fn(model, z), params)
        out.append(-sum((si * gi).sum() for si, gi in zip(s_test, gz)).item())
    return out

def grad_influence_wrt_input(model, x, y, s_test, loss_fn, params):
    """Returns d/dx ( s_test . grad_theta L(z) ) = s_test^T grad_x grad_theta L(z) = - I_pert,loss(z, z_test).
       To raise the test loss, step x along + I_pert,loss^T, i.e. x <- Pi(x - a*sign(returned))."""
    x = x.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(model, (x, y)), params, create_graph=True)
    scalar = sum((gi * si.detach()).sum() for gi, si in zip(g, s_test))
    return torch.autograd.grad(scalar, x)[0]
```
