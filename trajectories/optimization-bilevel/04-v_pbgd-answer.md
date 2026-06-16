## V-PBGD (value-function penalty-based bilevel gradient descent), distilled

**Problem.** Solve the bilevel problem with first-order steps when the inner level is only PL /
non-convex, *without* the gradient-norm penalty's fragility at degenerate lower-level curvature.

**Key idea.** Penalize the lower-level **value gap** `p = g(x, y) - v(x)`, `v(x) = min_y g`, and descend
`F_gamma = f + gamma p`. Its local stationarity in `y` is `grad_y f + gamma grad_y g = 0` (since `v` has
no `y`-dependence), so `||grad_y g|| <= L/gamma` *directly* — no `grad_yy g`, no singular-value
condition — and under `(1/mu)`-PL the residual is `p <= mu L^2/gamma^2` with one PL chain. The only
subtlety, `grad v(x)`, is removed by the PL Danskin lemma: at a lower minimizer `grad_y g(x, y*) = 0`, so
`grad v(x) = grad_x g(x, y*)` for any `y* in S(x)`. Estimate `y*` by a short warm-started inner loop; the
value-gap direction is `h = grad g(x, y) - (grad_x g(x, y_hat), 0)` (subtract only in `x`).

**Why it fits this task / harness.** Point `algorithm` at `run_v_pbgd`, which keeps a persistent
auxiliary `net_inner` (init from the model), runs `inner_itr` SGD steps on the `sigmoid(x)`-weighted
training loss at frozen `sigx`, then forms `fy=CE(val,clean)`, `gxy=(sigmoid(x)*CE_train).mean()` (main
model) and `vx=(sigmoid(x)*CE_inner).mean()` with the inner outputs **detached** so the gradient flows
only through `sigmoid(x)` (= `grad_x g(x,y_hat)`). Objective `min(1/(gamma+eps),1)*(fy + gamma*(gxy-vx))`,
`gamma` ramped 0->`gamma_max`. The single warm-started inner step is far cheaper than RHG's
500-step-from-scratch unroll. It ignores the exposed `inner_grad/outer_grad/inner_val` callables (builds
gradients from the two nets + `sigmoid(x)`). In **toy** mode it dispatches to `_toy_pbgd_step("v_pbgd")`,
which uses `grad g` (value gap with `v=0`), so the toy returns to the well-behaved 260.7-step regime.

**Hyperparameters.** Toy: `gams=(10.0,), alpha0=0.1`. Hyper-cleaning, linear: `lrx=0.1, lry=0.1,
lr_inner=0.01, inner_itr=1, gamma_max=0.2, gamma_argmax_step=30000, outer_itr=40000`; mlp: `lrx=0.1,
lry=0.01, lr_inner=0.01, inner_itr=1, gamma_max=0.1, gamma_argmax_step=10000, outer_itr=80000`. The
gentle `gamma_max` (0.1-0.2 vs G-PBGD's 37) is possible because the value gap is the more faithful
surrogate.

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.1,
        "lry": 0.1,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.2,
        "gamma_argmax_step": 30_000,
        "outer_itr": 40_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.1,
        "lry": 0.01,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.1,
        "gamma_argmax_step": 10_000,
        "outer_itr": 80_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_v_pbgd(state, hparams, grad_fns)
```
