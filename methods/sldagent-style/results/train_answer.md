I am a symbolic scaling-law discovery method called SLDAgent-style. My job is to take a table of observed language-model training runs and turn it into a compact mathematical law that predicts the loss of runs I have not seen, especially runs that are larger or lie outside the measured range. Each row gives me a few descriptor variables, such as parameter count N, token count D, vocabulary size V, learning rate l, batch size b, number of experts E, or unique-token budget U, together with a measured target such as cross-entropy loss. The data are grouped by experimental setting, and the law I produce must share one functional form across all groups while letting each group have its own fitted coefficients. The real test is extrapolation: I am fit only on small-scale runs and then judged on how well I predict the large-scale held-out runs.

The central move that defines SLDAgent-style is that the unit of evolution is not a formula by itself but a pair of subroutines: the symbolic expression that maps inputs and parameters to a predicted loss, and the optimization routine that turns observed data into fitted parameters for that expression. I co-evolve both parts together inside an LLM-driven evolutionary program search. This matters because a good symbolic form with a badly matched fitter is still useless. On easy regimes a simple additive power law plus a generic BFGS fitter is enough, so the gap is invisible there. On harder regimes, such as learning-rate and batch-size sweeps, the right form is a log-quadratic basin with cross-axis coupling and scale-dependent drift; that form is linear in its coefficients and wants a closed-form ridge least-squares solve, not a black-box gradient optimizer from a random start. By letting the LLM propose the form and its fitter in the same rewrite, I make sure the two are designed for each other.

The search engine is a quality-diversity evolutionary loop. I keep a population of scored programs in a MAP-Elites archive whose cells are defined by combined fitness, complexity, and diversity, with ten bins per axis and one elite per cell. This prevents the population from collapsing onto a single complicated champion and keeps simple, interpretable laws alive alongside richer ones. Five islands evolve in parallel and exchange migrants every twenty-five generations, so different basins of the formula space can develop independently before good ideas spread. When I need a parent to mutate, I sample seventy percent of the time from the elite archive, twenty percent uniformly from the current island, and ten percent from a residual fallback path, balancing exploitation and exploration. The LLM prompt contains the task context, data statistics such as ranges and variances, a small set of high-scoring inspiration programs, and the current parent program. I tell the LLM to respect fixed function signatures, to keep the law parameter-efficient for parsimony, and never to use input-dependent statistics such as median, min, or max of the points it is predicting, because those would leak the test distribution and destroy extrapolation. The LLM then performs a full rewrite of the evolvable block, which contains both the law and its fitter.

Fitness is computed only on the seen split. For each group I call the candidate's own fit_scaling_law on the group's training rows, predict with scaling_law_func, and accumulate squared error and total variance across all groups. I convert this into a normalized mean squared error and then into a combined score equal to one over one plus NMSE, which is bounded, monotone in R-squared, and easy to bin. Programs that crash, time out, or return non-finite predictions receive a failure floor and are effectively discarded. The test split is never touched during search; it is used only for the final evaluation of the discovered law.

On the learning-rate and batch-size regime, the kind of law this process discovers is a log-quadratic surface. I work in x equals log learning rate and y equals log batch size, and model log loss as a quadratic in x and y whose intercept and curvature drift with log N and log D. Because the form is linear in its coefficients, the fitter builds a design matrix and solves a single ridge-regularized least-squares problem. This uses every row in the sweep, unlike prior optima-only laws that throw away all but a handful of best points. The same fitted quadratic gives closed-form optima by setting the gradient to zero and solving a two-by-two linear system, so I can read off analytic learning-rate and batch-size recommendations as functions of model size and token count.

The method generalizes to other regimes by allowing the evolutionary search to propose domain-appropriate structure. For vocabulary scaling it can add a cross term between vocabulary size and token count on top of a multiplicative power-law backbone. For data-constrained training with repetition it can introduce a multiplicative repeat-efficiency factor that attenuates the effective token count as D grows far beyond the unique-token budget U. For mixture-of-experts it can prefer forms whose asymptotics stay bounded rather than exponentiating a log-bilinear expression that can diverge. In every case the decisive advantage is the same: the symbolic form and the fitting procedure are discovered and adapted together, not designed separately.

Here is the engine itself, exactly as I run it. The gene, `(scaling_law_func, fit_scaling_law)`, lives inside an `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` marker so the LLM can rewrite the law and its fitter together but never touch the surrounding plumbing; the seed is the naive case, a generic additive power law fit by plain BFGS. `fitness_of` scores a candidate only on the seen split, per group, by calling the candidate's own fitter and predictor, accumulating squared error and variance, and turning the total into `combined_score = 1/(1+NMSE)`; any crash, timeout, or non-finite prediction returns the failure floor of zero. `build_prompt` is what the LLM actually sees each iteration: the task context, the data statistics, the parameter-efficiency instruction, the ban on input-dependent features, the pinned signatures, a handful of high-scoring inspiration programs, and the current parent to improve. `discover` ties it together: it seeds a `ProgramDatabase` configured as MAP-Elites over `(combined_score, complexity, diversity)` with ten bins per axis, five islands, migration every twenty-five generations, and 70/20/10 exploitation/exploration/residual sampling, then for fifty iterations samples a parent and inspirations, builds the prompt, asks the LLM for a full rewrite of the evolvable block, scores the child on the seen data only, and inserts it back into the database, finally returning the highest-scoring program found.

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

And here, concretely, is the pair this process converges on for the learning-rate/batch-size regime: the log-quadratic law linear in its own coefficients, the closed-form ridge solve that fits it, and the closed-form optimum derived from the same fitted quadratic.

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

SLDAgent-style is the approach of treating a scaling law as a co-evolved pair of symbolic form and parameter-fitting procedure, discovered by an LLM-guided quality-diversity evolutionary search with strict separation between the seen split used for fitness and the held-out extrapolation split used for final evaluation. It is the coupling of expression and optimization, rather than either one alone, that lets the method find laws general enough to extrapolate across scales and regimes.
