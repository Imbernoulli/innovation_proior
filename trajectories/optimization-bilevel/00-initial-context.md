## Research question

A bilevel problem couples an outer objective `f(x, y)` to an inner problem `min_y g(x, y)` whose minimizer set `S(x)` drifts as `x` moves. The task is to find a single shared first-order update rule that performs well on both a toy instance and a data-hyper-cleaning instance, using only the gradient callables the loop provides.

The **toy** instance uses a one-dimensional upper variable `x` projected to `[0, 3]` and a one-dimensional `y`, sampled from 1000 random inits. The outer objective is `f(x, y) = cos(4y+2)/(1 + exp(2-4x)) + 0.5*log((4x-2)^2 + 1)`, the inner objective is `g(x, y) = (x+y)^2 + x*sin(x+y)^2`, and `S(x) = {-x}`. The primary metric is `convergence_steps` (lower is better); secondaries are residual and success rate.

The **data hyper-cleaning** instance corrupts 50% of MNIST training labels and learns a per-example weight vector `x` (weight `= sigmoid(x_i)` in `(0, 1)`) so that an inner classifier `y` trained on the weighted corrupted set generalizes to a clean validation split. The primary metric is clean `test_accuracy` (higher is better); secondaries are `f1_score` of the recovered clean/corrupt split and runtime-to-best.

## Prior art / Background / Baselines

- **Implicit differentiation.** Core idea: for strongly convex inner problems, compute the exact hypergradient by inverting the inner Hessian at the unique minimizer. Gap: it requires a Hessian inverse and fails when the inner problem is only PL / non-convex or `S(x)` is a flat valley, and it is second-order.

- **Reverse-mode hypergradients / ITD.** Core idea: unroll `T` inner gradient-descent steps, treat the final iterate as a differentiable surrogate for `y*(x)`, and back-propagate the validation loss through the inner trajectory by the adjoint recursion. Gap: the reverse pass stores every inner iterate, so memory grows as `O(T * dim(y))`, and the finite-time relation to the true bilevel solution is fragile.

- **Penalty reformulations.** Core idea: fold lower-level optimality into a single joint objective `F_gamma(x, y) = f(x, y) + gamma * p(x, y)` and descend on `(x, y)` with plain first-order steps, where `p` measures lower-level optimality. Gap: the choice of penalty `p`, penalty weight `gamma`, and update geometry are unsettled, and naive penalties can stall at spurious points.

## Fixed substrate / Code framework

The driver, dataset split (5000 train / 5000 validation / 10000 test), the 50% label-pollution protocol, the metrics, and both model architectures (a linear `784 -> 10` classifier and a 2-layer MLP `784 -> 300 -> 10` with a sigmoid hidden layer) are frozen. The toy problem definition is fixed: the same `f`, `g`, `x` in `[0, 3]`, and a fixed stationarity tolerance.

The scaffold exposes three reference helpers a method may call instead of re-deriving the update: `run_v_pbgd(state, hparams, grad_fns)` (penalty step on the value gap), `run_g_pbgd(...)` (penalty step on the gradient norm), and `run_rhg_family(...)` (reverse-mode unroll with a truncation knob `K`). Each dispatches on `state["task"]`: in toy mode it takes one projected penalized first-order step; in hyper-cleaning mode it runs the method's real inner/outer machinery and emits the `TRAIN_METRICS` / `FINAL_METRICS` lines the parser reads.

## Editable interface

Exactly one region of `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py` (lines 227–262, between the `BEGIN`/`END MLSBENCH_EDITABLE_ALGORITHM_REGION` markers) is editable. A method is a fill of this contract:

- `algorithm(state, hparams, grad_fns) -> dict` — one shared update used by both toy and hyper-cleaning runs. It receives the current `state` dict, a resolved `hparams` dict, and a `grad_fns` dict of callables, and returns the updated `state` after one outer update. Methods typically call the scaffold helpers rather than reimplement the gradient mechanics.
- `TOY_HPARAMS` — scalar knobs for the toy run.
- `HYPERCLEAN_HPARAMS` — scalar knobs for hyper-cleaning, optionally split into `linear` and `mlp` sub-dicts.

The starting point is the scaffold default shown below. A method replaces exactly this region — typically just the body of `algorithm` and the two hyperparameter dicts — and nothing else.

```python
# EDITABLE region of custom_strategy.py (lines 227-262) — default fill
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

## Evaluation settings

Three commands, each over seeds {42, 123, 456}: `toy-convergence` (1000 random inits, primary `convergence_steps`, lower better; secondaries `success_rate`, `final_residual`, `runtime_sec`), `hyperclean-linear` and the hidden `hyperclean-mlp` (primary `test_accuracy`, higher better; secondaries `f1_score`, cleaner precision/recall, runtime to best). The MLP command is the hidden, hardest target. The toy and linear runs report all three seeds; the MLP run reports seed 42.
