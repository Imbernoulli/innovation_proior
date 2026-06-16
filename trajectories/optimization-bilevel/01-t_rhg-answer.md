## T-RHG (Truncated Reverse-mode Hyper-Gradient), distilled

**Problem.** Compute the bilevel hypergradient `d_x F = grad_x f + (d y_T/d x)^T grad_y f` when
`y_T(x)` is the output of `T` inner gradient-descent steps and `x` (per-example weights) is
high-dimensional — without forming any parameter-by-parameter Jacobian and without storing the whole
inner trajectory.

**Key idea.** Unroll the inner dynamical system and contract the chain-rule product from the left (the
adjoint / reverse-mode recursion): `alpha_T = grad_y f`, sweep `t` down, accumulate `h += alpha_t . B_t`
and propagate `alpha_{t-1} = alpha_t . A_t`, with `A_t = I - lr_inner * grad_yy g` and `B_t = -lr_inner *
grad_xy g`. This is back-propagation through time over the optimizer's steps, `O(T)` time independent of
`dim(x)`. To dodge the `O(T*dim(y))` trajectory-storage wall, back-propagate through only the last `K`
of `T` transitions: because the inner map contracts near a well-conditioned minimizer, the dropped
deep terms carry a factor `(1 - lr_inner*alpha)^{T-t}`, so the truncation bias is geometrically small in
`K`. Full RHG is `K = T`; here `K = 100`, `T = 500`, storing only the last `K+1` states.

**Why it fits this task / harness.** The scaffold's `run_rhg_family` already implements the full forward
inner loop and the fixed `hg.reverse` adjoint sweep; the truncation is exactly the `K` hyperparameter.
The method calls the helper and ignores the exposed `outer_grad/inner_grad/inner_val` callables (the
adjoint needs a `create_graph=True` inner map, which the helper builds internally). In **toy** mode
`run_rhg_family` dispatches to the same projected penalized step as the value-gap method (`S(x)={-x}` so
there is nothing to unroll), so `TOY_HPARAMS` matches the other rungs and the toy numbers are expected
to coincide. T-RHG distinguishes itself only on hyper-cleaning, only through the truncation `K`.

**Hyperparameters.** Toy: `gams=(10.0,), alpha0=0.1`. Hyper-cleaning: `lr=0.001` outer,
`lr_inner=0.1` (linear) / `0.4` (mlp), `outer_itr=100`, `T=500`, `K=100`, `eval_interval=1`. The only
difference from full RHG is `K` (100 vs 500).

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
        "K": 100,
        "reg": 0.0,
        "eval_interval": 1,
    },
    "mlp": {
        "lr": 0.001,
        "lr_inner": 0.4,
        "outer_itr": 100,
        "T": 500,
        "K": 100,
        "reg": 0.0,
        "eval_interval": 1,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_rhg_family(state, hparams, grad_fns)
```
