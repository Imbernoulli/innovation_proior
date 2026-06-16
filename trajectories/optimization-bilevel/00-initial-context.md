## Research question

A bilevel problem couples an outer objective `f(x, y)` to an inner problem `min_y g(x, y)` whose
minimizer set `S(x)` drifts as `x` moves. Two concrete instances are fixed here. The **toy /
numerical-verification** instance is Section 5.1/6.1 of Shen and Chen (2023): a one-dimensional
upper variable `x` projected to `[0, 3]`, a one-dimensional `y`, sampled from 1000 random inits, where
the metric is how many first-order steps it takes to reach a stationary point (`convergence_steps`,
lower is better) and at what residual. The **data hyper-cleaning** instance corrupts 50% of MNIST
training labels and learns a per-example weight vector `x` (weight `= sigmoid(x_i) in (0, 1)`) so that
an inner classifier `y` trained on the weighted corrupted set generalizes to a clean validation split;
the metric is clean `test_accuracy` (higher is better), with `f1_score` of the recovered clean/corrupt
split and a runtime-to-best as secondaries. The single thing being designed is **one first-order
update rule** shared by both instances: it must converge the toy faster and recover more clean MNIST
data, using only the gradient callables the loop hands it.

## Prior art before the first rung

The ladder reacts to the line of bilevel-gradient methods that the scaffold's fixed gradient interface
deliberately rules in or out. These are the ancestors; the first rung is the cheapest one that fits.

- **Implicit differentiation (Ghadimi and Wang, 2018).** When `g(x, .)` is strongly convex,
  `S(x) = {y*(x)}` is a singleton and the hypergradient is
  `grad_x f - grad_xy g (grad_yy g)^{-1} grad_y f`, evaluated at `y*(x)`. Exact and `T`-free, but it
  needs `grad_yy g` invertible everywhere (strong convexity) and an inverse-Hessian–vector product.
  Gap: undefined the moment the inner level is only PL / non-convex or `S(x)` is a flat valley — the
  regime this task lives in — and it is second-order.
- **Reverse-mode hypergradients / ITD (Franceschi et al., 2017; Maclaurin et al., 2015).** Unroll `T`
  inner gradient-descent steps `y_{t+1} = y_t - lr_inner * grad_y g`, treat `y_T` as a differentiable
  surrogate for `y*(x)`, and back-propagate the validation loss through the whole inner trajectory by
  the adjoint recursion. Tolerates non-convex `g` and needs no Hessian inverse — but the reverse pass
  must store every inner iterate, so memory grows as `O(T * dim(y))`, and the finite-time relation to
  the true bilevel solution is fragile. Gap: the trajectory-storage wall.
- **Penalty reformulations (the lineage the PBGD rung lands on).** Fold lower-level optimality into a
  single joint objective `F_gamma(x, y) = f(x, y) + gamma * p(x, y)` and descend it on `(x, y)` with
  plain first-order steps, where `p` measures lower-level optimality (the squared gradient norm
  `||grad_y g||^2`, or the value gap `g(x, y) - v(x)` with `v(x) = min_y g`). No inverse, no stored
  trajectory, one loop. Gap before the ladder: which penalty `p`, which `gamma`, and which geometry
  make a penalized solution an actual bilevel solution — naive penalties can stall at spurious points.

## The fixed substrate

The driver, dataset split (5000 train / 5000 validation / 10000 test), the 50% label-pollution
protocol, the metrics, and both model architectures (a linear `784 -> 10` classifier and a 2-layer MLP
`784 -> 300 -> 10` with a sigmoid hidden layer) are frozen. The toy problem definition is fixed:
`f(x, y) = cos(4y+2)/(1 + exp(2-4x)) + 0.5*log((4x-2)^2 + 1)`, `g(x, y) = (x+y)^2 + x*sin(x+y)^2`, with
`x` projected to `[0, 3]` and the stationarity tolerance fixed; here `S(x) = {-x}` so `v(x) = 0`, the
value gap reduces to `g` itself, and a step is counted whenever it advances `state["total_steps"]`.

The scaffold also exposes three reference helpers a method may simply call instead of re-deriving the
update: `run_v_pbgd(state, hparams, grad_fns)` (value-gap penalty step), `run_g_pbgd(...)`
(gradient-norm penalty step), and `run_rhg_family(...)` (reverse-mode unroll with a truncation knob
`K`). Each dispatches on `state["task"]`: in toy mode it takes one projected penalized first-order
step; in hyper-cleaning mode it runs the method's real inner/outer machinery and emits the
`TRAIN_METRICS` / `FINAL_METRICS` lines the parser reads.

## The editable interface

Exactly one region of `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py` (lines
227–262, between the `BEGIN`/`END MLSBENCH_EDITABLE_ALGORITHM_REGION` markers) is editable. A method
is a fill of this same contract:

- `algorithm(state, hparams, grad_fns) -> dict` — one shared update used by both toy and
  hyper-cleaning runs. It receives the current `state` dict, a resolved `hparams` dict, and a
  `grad_fns` dict of callables, and returns the updated `state` after one outer (or method-equivalent)
  update. For toy mode `grad_fns` provides `f, df, g, dg_dy, dg_dl, proj, init_state`. For
  hyper-cleaning mode `grad_fns` provides `outer_grad, inner_grad, inner_val, init_state` — first-order
  information for the validation loss, the weighted training loss, and the initial state.
- `TOY_HPARAMS` — scalar knobs for the toy run (penalty schedule `gams`, base step `alpha0`).
- `HYPERCLEAN_HPARAMS` — scalar knobs for hyper-cleaning, optionally split into `linear` and `mlp`
  sub-dicts (learning rates, penalty schedule or unroll horizon, outer iterations, eval interval).

The starting point is the scaffold default: `algorithm` delegates to `run_v_pbgd`, with the value-gap
penalty hyperparameters below. Each rung replaces exactly this region — typically just the body of
`algorithm` (which helper it calls) and the two hyperparameter dicts — and nothing else.

```python
# EDITABLE region of custom_strategy.py (lines 227-262) — default fill
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

## Evaluation settings

Three commands, each over seeds {42, 123, 456}: `toy-convergence` (1000 random inits, primary
`convergence_steps`, lower better; secondaries `success_rate`, `final_residual`, `runtime_sec`),
`hyperclean-linear` and the hidden `hyperclean-mlp` (primary `test_accuracy`, higher better;
secondaries `f1_score`, cleaner precision/recall, runtime to best). The MLP command is the hidden,
hardest target. The toy and linear runs report all three seeds; the MLP run reports seed 42.
