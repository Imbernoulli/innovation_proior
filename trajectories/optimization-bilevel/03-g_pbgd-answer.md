## G-PBGD (gradient-norm penalty-based bilevel gradient descent), distilled

**Problem.** Solve the bilevel problem with one coupled first-order loop — no inner optimizer, no value
tracking, no Hessian inverse — when the inner level is only PL / non-convex.

**Key idea.** Fold lower-level optimality into a single joint objective
`F_gamma(x, y) = f(x, y) + (gamma/2)||grad_y g(x, y)||^2` and descend it on `(x, y)` jointly. The
penalty gradient is *exactly* computable as Hessian-vector products: compute `grad_y g` with a retained
graph (`create_graph=True`), then one backward of `(gamma/2)||grad_y g||^2` yields
`gamma grad_yy g . grad_y g` and `gamma grad_xy g . grad_y g` (the `1/2` cancels the square's `2`).
Under `(1/sqrt(mu))`-PL the squared gradient norm is a squared-distance bound, so a finite
`gamma = Theta(delta^{-1/2})` suffices. Fragility: a local minimum can be spurious wherever `grad_yy g`
degenerates (the diagnostic `y=2pi/3` example), which is why the value-gap rung is the more robust
sibling.

**Why it fits this task / harness.** Point `algorithm` at `run_g_pbgd`, which forms `gxy`, takes
`dgdy = autograd.grad(gxy, params, create_graph=True)`, sets `objective = min(1/(gamma+eps),1) * (fy +
0.5*gamma*||dgdy||^2)`, and steps `x_opt` (`lrx`) and `y_opt` (`lry`) with `gamma` annealed linearly
from `gamma_init` to `gamma_max` over `gamma_argmax_step`. The `min(1/gamma,1)` rescale keeps the step
stable as `gamma` grows. It does *not* use the exposed `inner_grad/outer_grad` callables — the HVP is
built directly from the model and `sigmoid(x)`. In **toy** mode `run_g_pbgd` dispatches to
`_toy_pbgd_step("g_pbgd")`, which uses `toy_gpbgd_penalty_grad` (the gradient of `||grad_y g||^2`), so
the toy descends a stiffer objective and is expected to need *more* steps and a larger residual than the
value-gap-style rungs.

**Hyperparameters.** Toy: `gams=(10.0,), alpha0=0.1`. Hyper-cleaning, linear: `lrx=0.3, lry=0.5,
gamma_max=37.0, gamma_argmax_step=5000, outer_itr=40000`; mlp: `lrx=0.5, lry=0.5, gamma_max=37.0,
gamma_argmax_step=30000, outer_itr=50000`. No `lr_inner / T / K` — there is no inner loop.

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.3,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 5_000,
        "outer_itr": 40_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.5,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 30_000,
        "outer_itr": 50_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_g_pbgd(state, hparams, grad_fns)
```
