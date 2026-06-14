# SLDAgent, distilled

SLDAgent is an evolution-based coding agent that automatically discovers symbolic scaling laws from
a table of observed training runs. Its central move: the unit of evolution is not a formula but a
**pair of subroutines**, `(Expression, Optimization)` — the symbolic form `f_theta(x) -> y_hat` and
the routine that fits its coefficients — co-evolved together. An LLM mutates a parent program; the
child is executed, scored by goodness of fit on the seen data, and inserted into a quality-diversity
database (MAP-Elites over feature cells, with islands and migration). After a fixed budget of
iterations the highest-scoring program is returned. The held-out (extrapolation) split is never
touched during search.

## Problem it solves

Given a fixed dataset `D = {(x_i, group_i, y_i)}` of training-run descriptors `x` (e.g. `N`, `D`,
`V`, `l`, `b`, `E`, `U`) and a target `y` (loss), discover (1) a single symbolic functional form
shared across all groups, and (2) per-group fitted coefficients `theta_group`, such that the law
predicts *unseen, extrapolated* points (larger models/datasets/more extreme hyperparameters)
accurately. Measured by held-out `R^2` per benchmark.

## Key idea — why co-evolve the pair

A scaling law is two things: the symbolic form and how you fit it. Prior LLM-driven program search
(FunSearch) evolves only the form, with a fixed program skeleton and a hand-written optimizer. On
easy regimes (near-Chinchilla) a fixed BFGS fitter suffices, so the gap is invisible. On hard
regimes the candidate forms are awkward (multi-axis power products, log-quadratic basins,
interaction terms) and the *fitting* becomes the bottleneck — a plausible form with a naive fitter
can still be discarded because its parameters were fit badly. So the gene must be the pair: when the
LLM proposes a more aggressive form it proposes a tailored fitter in the same breath.

## The evolutionary engine

- **Gene** = `(scaling_law_func, fit_scaling_law)` inside an `# EVOLVE-BLOCK`. Seed = generic
  additive power law + naive BFGS.
- **Database** = MAP-Elites: programs binned by `[combined_score, complexity, diversity]`, 10 bins
  per axis, one elite per cell — keeps simple, complex, and novel candidates alive (parsimony often
  extrapolates best). Plus **5 islands** evolving in parallel, **migration every 25 generations**,
  population size 100 and archive size 50, to fight premature convergence.
- **Parent sampling** = **70% exploitation** (elite archive) / **20% exploration** (current-island
  random draw) / **10% residual path** (elite-style preservation in the algorithm description;
  random or fitness-weighted fallback in the local sampler), with archive and best-program tracking preserving elites. Prompts include top and
  diverse examples for best-shot-style context.
- **Prompt** carries: task context, inspiration programs, data statistics (ranges/means/variances),
  a **parameter budget or parameter-efficiency instruction** (for parsimony → less overfitting), a **ban on
  input-dependent features** (`median/min/max` of inputs leak the test distribution and break
  extrapolation), and **pinned signatures** so candidates load and run.
- **Mutation** = full rewrite of the EVOLVE block (programs are short; let the LLM restructure form
  and fitter at once).
- **Fitness** = on the seen split only, per group fit then score; `NMSE = mean((y - y_hat)^2)/Var(y)`,
  `R^2 = 1 - NMSE`, `combined_score = 1/(1 + NMSE)` (bounded, monotone in `R^2`). Crashes/timeouts
  get the failure floor. Pure numerical fitness (no LLM judge, no cascade); parallel evaluation,
  per-program timeout + retries.
- **Loop** = 50 iterations: seed → (sample parent+inspirations → build prompt → LLM full rewrite →
  execute & score on seen → insert) → return best. Test split untouched throughout.

## Representative output forms

- **lr & bsz** (no prior full-loss law): a log-quadratic surface in `x = ln l`, `y = ln b` with
  scale-dependent drift,
  `log(loss) = C(N,D) + b3 y + b4 x + b5 x^2 + b6 y^2 + b7 xy + b9 (ln N) y + b10 (ln D) x`.
  Linear in its coefficients → fit by closed-form ridge least-squares on the design matrix (uses
  *all* the data, vs Step Law's 17 best points). Quadratic → analytic optimum (below).
- **MoE**: `L = t1 N^t2 / (1 + t3 E^t4) + t5 N^{0.6 t2} + t6`. Diminishing-returns expert
  attenuation + reduced-rate term + irreducible floor `t6`. As `E -> inf`, `L -> t5 N^{0.6 t2} + t6`;
  with `t2 < 0`, `L -> t6` as `N -> inf` — bounded, principled asymptotics (the human
  exp-of-log-bilinear form can diverge if its interaction coefficient is positive).
- **SFT**: `L = t2 + t0 / (1 + (D/t3)^{t1})` — the offset enters as a *dimensionless ratio*, so `t3`
  is a true characteristic data scale (the human `t2 + t0/(D^{t1}+t3)` has exponent-dependent units).

## Closed-form hyperparameter optimum (lr & bsz)

Because `log(loss)` is quadratic in `(x, y)`, set the gradient to zero:

```
b4 + 2 b5 x + b7 y + b10 ln D = 0
b3 + b7 x + 2 b6 y + b9 ln N = 0
```

a 2x2 system `H [x; y] = -[b4 + b10 ln D ; b3 + b9 ln N]`, `H = [[2 b5, b7], [b7, 2 b6]]`. With
`Delta = det(H) = 4 b5 b6 - b7^2 > 0` (positive-definite → unique minimum), Cramer's rule gives

```
x* = ( -2 b6 (b4 + b10 ln D) + b7 (b3 + b9 ln N) ) / (4 b5 b6 - b7^2)
y* = (   b7 (b4 + b10 ln D) - 2 b5 (b3 + b9 ln N) ) / (4 b5 b6 - b7^2)
lr* = exp(x*),   bsz* = exp(y*)   (affine in ln N, ln D).
```

Fitted: `b3=0.0595, b4=0.1906, b5=0.0098, b6=0.0073, b7=-0.006, b9=-0.0089, b10=-0.0012`;
`Delta = 0.00025016`; `x* = -12.5510 + 0.070035 ln D + 0.213463 ln N`,
`y* = -9.23329 + 0.028782 ln D + 0.697314 ln N`. At `N = 2^30 ≈ 1.07e9`, `D = 1e11`:
`lr* ≈ 1.767e-3`, `bsz* ≈ 401.8` as the analytic target values before snapping to an admissible grid.

## Working code

The seed gene (generic power law + BFGS), the fitness on the seen split, the prompt, and the loop:

```python
import numpy as np
import uuid
from scipy.optimize import minimize
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


# ---- seed candidate program: BOTH form and fitter are inside the EVOLVE block ----
SEED_PROGRAM = r'''
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points))           # (N, F)
    _, F = X.shape
    params = np.asarray(params)
    if params.ndim == 1:
        params = params[None, :]
    T, _ = params.shape
    coeffs = params[:, :F]
    exps = params[:, F:2 * F]
    bias = params[:, -1]
    pred = (coeffs[None, :, :] * (X[:, None, :] ** exps[None, :, :])).sum(axis=2) + bias[None, :]
    return pred[:, 0] if pred.shape[1] == 1 else pred

def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points)); y = np.asarray(loss_values)
    P = 2 * X.shape[1] + 1
    y2d = y[:, None] if y.ndim == 1 else y
    T = y2d.shape[1]
    init = np.ones((T, P))
    def objective(flat_params):
        params = flat_params.reshape(T, P)
        return np.mean((scaling_law_func(X, params) - y2d) ** 2)
    res = minimize(objective, init.ravel(), method="BFGS")
    params_opt = res.x.reshape(T, P) if res.success else init
    return params_opt[0] if T == 1 else params_opt
# EVOLVE-BLOCK-END
'''


# ---- fitness on the SEEN split only; combined_score = 1/(1+NMSE), monotone in R^2 ----
def fitness_of(prog, seen_by_group):
    sse = sst = 0.0
    for _g, (X, y) in seen_by_group.items():
        try:
            theta = prog.fit_scaling_law(X, y)
            pred = np.asarray(prog.scaling_law_func(X, theta), dtype=float)
            if not np.all(np.isfinite(pred)):
                return 0.0
            sse += float(np.sum((y - pred) ** 2))
            sst += float(np.sum((y - np.mean(y)) ** 2))
        except Exception:
            return 0.0                                    # crash/timeout -> floor
    nmse = sse / sst if sst > 0 else np.inf
    return 1.0 / (1.0 + nmse)


# ---- the prompt: co-evolve form + fitter, with parsimony + anti-leak constraints ----
def build_prompt(parent_code, inspirations, ctx, stats, parameter_instruction):
    inspo = "\n\n".join(f"# score={s:.4f}\n{c}" for c, s in inspirations)
    return (f"Evolve BOTH scaling_law_func (the law) AND fit_scaling_law (its fitter) "
            f"for: {ctx}. {parameter_instruction} Do NOT use input-dependent stats "
            f"(median/min/max) in scaling_law_func. Keep the signatures. Edit only inside "
            f"# EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END.\nData stats: {stats}\n"
            f"High-scoring programs:\n{inspo}\n\nProgram to improve:\n{parent_code}\n")


# ---- evolutionary loop: MAP-Elites + islands, 70/20/10 sampling, 50 iterations ----
def add_scored_program(db, code, score, parent_id=None):
    program = Program(id=str(uuid.uuid4()), code=code, parent_id=parent_id,
                      metrics={"combined_score": float(score)})
    db.add(program)
    return program

def discover(task, n_iterations=50, n_islands=5,
             parameter_instruction="Keep the law parameter-efficient; use a task-specific cap where configured."):
    seen = load_seen_data(task)                           # {group: (X, y)}
    db_config = DatabaseConfig(population_size=100, archive_size=50, num_islands=n_islands,
                               feature_dimensions=["combined_score", "complexity", "diversity"],
                               feature_bins=10, exploitation_ratio=0.70,
                               exploration_ratio=0.20, elite_selection_ratio=0.10,
                               migration_interval=25, migration_rate=0.10)
    db = ProgramDatabase(db_config)
    seed_program = add_scored_program(db, SEED_PROGRAM, fitness_of(load_module(SEED_PROGRAM), seen))
    for _ in range(n_iterations):
        parent, inspirations = db.sample(num_inspirations=3)
        inspiration_items = [(p.code, p.metrics.get("combined_score", 0.0)) for p in inspirations]
        child = llm_propose(build_prompt(parent.code, inspiration_items,
                                         task.context, task.data_stats, parameter_instruction))
        try:
            score = fitness_of(load_module(child), seen)  # seen split only
        except Exception:
            score = 0.0
        add_scored_program(db, child, score, parent_id=parent.id)  # test split untouched
    return db.get_best_program()
```

The discovered lr&bsz pair (a form linear in its coefficients + a closed-form fit), plus the
analytic optimum:

```python
import numpy as np

def _cols(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected data_points of shape (N,4), got {X.shape}")
    if np.any(X <= 0):
        raise ValueError("All input features must be strictly positive for log transforms.")
    return np.log(X[:, 0]), np.log(X[:, 1]), np.log(X[:, 2]), np.log(X[:, 3])  # lr, bsz, D, N

def _design(l_lr, l_b, l_D, l_P):
    return np.column_stack([
        np.ones_like(l_lr), l_P, l_D, l_b, l_lr,
        l_lr ** 2, l_b ** 2, l_lr * l_b,                 # basin + lr/bsz coupling
        l_P * l_D, l_P * l_b, l_D * l_lr,                # scale-dependent drift
    ])

def scaling_law_func(data_points, params):               # predicts lm_loss
    X = np.asarray(data_points, dtype=float)
    Z = _design(*_cols(X))
    return np.exp(Z.dot(np.asarray(params, dtype=float)))

def fit_scaling_law(data_points, loss_values):           # closed-form ridge least squares
    X = np.asarray(data_points, dtype=float)
    Z = _design(*_cols(X))
    y = np.asarray(loss_values, dtype=float).ravel()
    if np.any(y <= 0):
        raise ValueError("Loss values must be strictly positive for log transform.")
    l_y = np.log(y)
    A = Z.T.dot(Z) + 1e-6 * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T.dot(l_y))

def optimal_lr_bsz(beta, N, D):                          # 2x2 solve of the log-quadratic
    b3, b4, b5, b6, b7, b9, b10 = (beta[3], beta[4], beta[5], beta[6],
                                   beta[7], beta[9], beta[10])
    Ac, Bc, Cc = 2 * b5, b7, 2 * b6
    Delta = Ac * Cc - Bc ** 2                            # det(H) > 0 => real minimum
    if Delta <= 0:
        raise ValueError("The fitted log-quadratic is not positive-definite in lr/bsz.")
    rx, ry = b4 + b10 * np.log(D), b3 + b9 * np.log(N)
    x = (-Cc * rx + Bc * ry) / Delta                     # Cramer's rule
    y = ( Bc * rx - Ac * ry) / Delta
    return np.exp(x), np.exp(y)                          # lr*, bsz*
```
