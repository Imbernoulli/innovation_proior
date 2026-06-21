I am a symbolic scaling-law discovery method called SLDAgent-style. My job is to take a table of observed language-model training runs and turn it into a compact mathematical law that predicts the loss of runs I have not seen, especially runs that are larger or lie outside the measured range. Each row gives me a few descriptor variables, such as parameter count N, token count D, vocabulary size V, learning rate l, batch size b, number of experts E, or unique-token budget U, together with a measured target such as cross-entropy loss. The data are grouped by experimental setting, and the law I produce must share one functional form across all groups while letting each group have its own fitted coefficients. The real test is extrapolation: I am fit only on small-scale runs and then judged on how well I predict the large-scale held-out runs.

The central move that defines SLDAgent-style is that the unit of evolution is not a formula by itself but a pair of subroutines: the symbolic expression that maps inputs and parameters to a predicted loss, and the optimization routine that turns observed data into fitted parameters for that expression. I co-evolve both parts together inside an LLM-driven evolutionary program search. This matters because a good symbolic form with a badly matched fitter is still useless. On easy regimes a simple additive power law plus a generic BFGS fitter is enough, so the gap is invisible there. On harder regimes, such as learning-rate and batch-size sweeps, the right form is a log-quadratic basin with cross-axis coupling and scale-dependent drift; that form is linear in its coefficients and wants a closed-form ridge least-squares solve, not a black-box gradient optimizer from a random start. By letting the LLM propose the form and its fitter in the same rewrite, I make sure the two are designed for each other.

The search engine is a quality-diversity evolutionary loop. I keep a population of scored programs in a MAP-Elites archive whose cells are defined by combined fitness, complexity, and diversity, with ten bins per axis and one elite per cell. This prevents the population from collapsing onto a single complicated champion and keeps simple, interpretable laws alive alongside richer ones. Five islands evolve in parallel and exchange migrants every twenty-five generations, so different basins of the formula space can develop independently before good ideas spread. When I need a parent to mutate, I sample seventy percent of the time from the elite archive, twenty percent uniformly from the current island, and ten percent from a residual fallback path, balancing exploitation and exploration. The LLM prompt contains the task context, data statistics such as ranges and variances, a small set of high-scoring inspiration programs, and the current parent program. I tell the LLM to respect fixed function signatures, to keep the law parameter-efficient for parsimony, and never to use input-dependent statistics such as median, min, or max of the points it is predicting, because those would leak the test distribution and destroy extrapolation. The LLM then performs a full rewrite of the evolvable block, which contains both the law and its fitter.

Fitness is computed only on the seen split. For each group I call the candidate's own fit_scaling_law on the group's training rows, predict with scaling_law_func, and accumulate squared error and total variance across all groups. I convert this into a normalized mean squared error and then into a combined score equal to one over one plus NMSE, which is bounded, monotone in R-squared, and easy to bin. Programs that crash, time out, or return non-finite predictions receive a failure floor and are effectively discarded. The test split is never touched during search; it is used only for the final evaluation of the discovered law.

On the learning-rate and batch-size regime, the kind of law this process discovers is a log-quadratic surface. I work in x equals log learning rate and y equals log batch size, and model log loss as a quadratic in x and y whose intercept and curvature drift with log N and log D. Because the form is linear in its coefficients, the fitter builds a design matrix and solves a single ridge-regularized least-squares problem. This uses every row in the sweep, unlike prior optima-only laws that throw away all but a handful of best points. The same fitted quadratic gives closed-form optima by setting the gradient to zero and solving a two-by-two linear system, so I can read off analytic learning-rate and batch-size recommendations as functions of model size and token count.

The method generalizes to other regimes by allowing the evolutionary search to propose domain-appropriate structure. For vocabulary scaling it can add a cross term between vocabulary size and token count on top of a multiplicative power-law backbone. For data-constrained training with repetition it can introduce a multiplicative repeat-efficiency factor that attenuates the effective token count as D grows far beyond the unique-token budget U. For mixture-of-experts it can prefer forms whose asymptotics stay bounded rather than exponentiating a log-bilinear expression that can diverge. In every case the decisive advantage is the same: the symbolic form and the fitting procedure are discovered and adapted together, not designed separately.

The small runnable illustration below shows the core idea on synthetic data. It generates a few groups of training runs where the true loss follows a Chinchilla-like base plus a log-quadratic penalty around problem-specific optima, then fits a single shared log-quadratic form to each group using ridge least squares and predicts on an extrapolated held-out point. The demo is self-contained and uses only NumPy and SciPy.

```python
import numpy as np
from scipy.optimize import least_squares

np.random.seed(0)

def true_loss(lr, bsz, D, N, group):
    # Shared Chinchilla-like base with group-specific quadratic basin in log(lr), log(bsz).
    E = 2.0
    base = E + 1.2 * N ** -0.34 + 0.8 * D ** -0.28
    x = np.log(lr)
    y = np.log(bsz)
    # Group-specific optimum and curvature.
    star = {0: (-8.0, -6.0), 1: (-7.5, -5.5), 2: (-7.0, -5.0)}[group]
    dx, dy = x - star[0], y - star[1]
    penalty = 0.05 * (dx ** 2 + dy ** 2 + 0.3 * dx * dy)
    return base + penalty + 0.02 * np.random.randn()

def make_design(X):
    lr, bsz, D, N = [np.clip(X[:, i], 1e-12, None) for i in range(4)]
    lx, ly = np.log(lr), np.log(bsz)
    lD, lN = np.log(D), np.log(N)
    return np.column_stack([
        np.ones_like(lx), lN, lD, ly, lx,
        lx ** 2, ly ** 2, lx * ly, lN * lD, lN * ly, lD * lx
    ])

def fit_scaling_law(X, y):
    Z = make_design(X)
    ly = np.log(np.clip(y, 1e-12, None))
    A = Z.T @ Z + 1e-6 * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ ly)

def predict_scaling_law(X, beta):
    return np.exp(make_design(X) @ beta)

# Three groups of small-scale observed runs.
groups = []
for g in range(3):
    n = 30
    lr = np.exp(np.random.uniform(-10, -5, n))
    bsz = np.exp(np.random.uniform(-8, -3, n))
    D = np.random.uniform(1e8, 5e9, n)
    N = np.random.uniform(1e7, 2e8, n)
    X = np.column_stack([lr, bsz, D, N])
    y = np.array([true_loss(*X[i], g) for i in range(n)])
    groups.append((g, X, y))

# Fit one shared form per group on seen data.
betas = {}
for g, X, y in groups:
    betas[g] = fit_scaling_law(X, y)

# Extrapolate to a much larger held-out configuration for group 1.
X_test = np.array([[3e-3, 512.0, 1e11, 1e9]])
y_true = true_loss(3e-3, 512.0, 1e11, 1e9, 1)
y_pred = predict_scaling_law(X_test, betas[1])

print(f"Group 1 extrapolation: true={y_true:.4f}, predicted={y_pred[0]:.4f}")

# Closed-form optimum for group 1 at this scale.
b = betas[1]
b3, b4, b5, b6, b7, b9, b10 = b[3], b[4], b[5], b[6], b[7], b[9], b[10]
Ac, Bc, Cc = 2 * b5, b7, 2 * b6
Delta = Ac * Cc - Bc ** 2
rx, ry = b4 + b10 * np.log(1e11), b3 + b9 * np.log(1e9)
x_star = (-Cc * rx + Bc * ry) / Delta
y_star = (Bc * rx - Ac * ry) / Delta
print(f"Recommended lr*={np.exp(x_star):.4e}, bsz*={np.exp(y_star):.1f}")
```

SLDAgent-style is the approach of treating a scaling law as a co-evolved pair of symbolic form and parameter-fitting procedure, discovered by an LLM-guided quality-diversity evolutionary search with strict separation between the seen split used for fitness and the held-out extrapolation split used for final evaluation. It is the coupling of expression and optimization, rather than either one alone, that lets the method find laws general enough to extrapolate across scales and regimes.
