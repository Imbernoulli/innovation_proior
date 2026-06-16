## RHG (Reverse-mode Hyper-Gradient, full unroll), distilled

**Problem.** The exact bilevel hypergradient of a `T`-step inner training run,
`d_x F = grad_x f + (d y_T/d x)^T grad_y f`, with no truncation bias — at the cost of storing the whole
inner trajectory.

**Key idea.** The same adjoint recursion as T-RHG, run over the *entire* trajectory: `alpha_T =
grad_y f`, `alpha_{t-1} = alpha_t . A_t`, `h += alpha_t . B_t` for `t = T ... 1`, with `A_t = I -
lr_inner * grad_yy g`, `B_t = -lr_inner * grad_xy g`. This is back-propagation through time over the
optimizer's steps, derived as the Lagrangian stationarity conditions (a multiplier per inner
constraint). Full RHG is the `K = T` case of T-RHG; it adds back exactly the deep trajectory terms,
each bounded by `(1 - lr_inner*alpha)^{T-t}`, that truncation dropped. In the converged limit the full
backward sum is the Neumann series of the inverse Hessian, so RHG equals implicit differentiation
computed by summation — its quality is capped by how well the 500-step inner loop solves the inner
problem.

**Why it fits this task / harness.** Identical to T-RHG: call `run_rhg_family`, which keeps the suffix
of `K+1 = T+1` inner states and runs the fixed `hg.reverse(..., [fp_map]*K, ...)` sweep. The single
change from step 1 is `K = T = 500` (store/differentiate the full trajectory). It ignores the exposed
`outer_grad/inner_grad/inner_val` callables (the adjoint needs the `create_graph=True` inner map the
helper builds internally). In **toy** mode it dispatches to the shared projected penalized step
(`S(x)={-x}`, nothing to unroll), so the toy numbers are expected to match every other rung.

**Hyperparameters.** Toy: `gams=(10.0,), alpha0=0.1`. Hyper-cleaning: `lr=0.001` outer,
`lr_inner=0.1` (linear) / `0.4` (mlp), `outer_itr=100`, `T=500`, `K=500`, `eval_interval=1`. Only `K`
differs from T-RHG (500 vs 100).

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lr": 0.001,
        "lr_inner": 0.1,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
    "mlp": {
        "lr": 0.001,
        "lr_inner": 0.4,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_rhg_family(state, hparams, grad_fns)
```
