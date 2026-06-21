## Research question

Given only observational i.i.d. data on `p` continuous random variables `X_1, ..., X_p` — no
experiments, no interventions, no time ordering — recover the *directed* acyclic graph (DAG) that
encodes which variable causally influences which. The data are assumed Markov with respect to some
underlying causal DAG `D^0`, all variables observed (no hidden confounders), no directed cycles.
"Recover the DAG" means recover the actual arrow directions, so that one can read off intervention
(do-) distributions: `p(x_j | do(X_k = x))`. The structural mechanisms are taken to be
*nonlinear* — each variable is some smooth, generally nonlinear function of its parents plus an
independent noise term — and the question is how to estimate the directed graph from such data,
including in the high-dimensional regime `p >> n` where each of the `p` mechanisms may have up to
`p-1` candidate predictors.

## Background

**Graphical models and structural equation models.** A causal DAG `D` over `X_1,...,X_p` says each
variable is generated from its parents: a structural equation model (SEM) writes
`X_j = f_j(X_{pa(j)}, eps_j)` with mutually independent noise `eps_1,...,eps_p`. The joint law is
then Markov with respect to `D` — each variable independent of its non-descendants given its
parents — and Pearl's do-calculus turns the DAG plus mechanisms into intervention distributions.
In the fully nonparametric case, or the multivariate-Gaussian case, the SEM view and the graphical
view are equivalent, and observation alone cannot orient all edges.

**Identifiability of the DAG.** The basic fact that frames everything: for
a *linear-Gaussian* SEM `X_j = sum_k beta_{jk} X_k + eps_j` with Gaussian noise, the DAG is *not*
identifiable from the distribution. The textbook example is two variables, `X_2 = a X_1 + N_2` with
`X_1, N_2` Gaussian: project `X_1` on `X_2` and, because uncorrelated jointly-Gaussian variables
are independent, you get a perfectly valid *backward* model `X_1 = a' X_2 + N_1'` with independent
noise. Forward and backward fit equally well, and the direction is not distinguished. Assuming
faithfulness, constraint- and score-based methods recover the *Markov equivalence class* (MEC) —
the set of DAGs with the same skeleton and v-structures — leaving many edges undirected.

Restricting the function or noise class changes this. Two strands establish identifiability of the
full DAG. First, *linear non-Gaussian*: if the mechanisms are linear but the noise is
non-Gaussian, the Darmois-Skitovic theorem makes the backward model fail, and the full DAG becomes
identifiable (Shimizu et al. 2006, via ICA). Second, and central here, *nonlinear additive noise*.
Consider the additive-noise model (ANM) `X_j = f_j(X_{pa(j)}) + N_j` with independent noise of
strictly positive density. Hoyer et al. (2008) and Peters, Mooij, Janzing & Scholkopf (2014)
showed that for such models the DAG is *generically identifiable from the observational
distribution*. The bivariate mechanism: a forward ANM `X_2 = f(X_1) + N_2` admits a backward ANM
`X_1 = g(X_2) + N_1` with independent residual *only* if the triple `(f, p_{X_1}, p_{N_2})`
solves a specific third-order differential equation in the log-densities. With
`xi = log p_{X_1}`, `nu = log p_{N_2}`, that condition is

```
xi''' = xi'' ( -nu''' f' / nu''  +  f'' / f' )
        - 2 nu'' f'' f'  +  nu' f'''  +  nu' nu''' f'' f' / nu''  -  nu' (f'')^2 / f'
```

(holding wherever `nu''(x_2 - f(x_1)) f'(x_1) != 0`). For a fixed `(f, p_N)` the set of input
densities `p_X` that satisfy it is confined to a low-dimensional space, so "most" triples
violate it and the direction is identifiable. A clean corollary: if `X_1` and `N_2` are Gaussian
and a backward additive-noise model exists, then `f` must be linear. Contrapositive:
**nonlinear `f` with Gaussian noise is identifiable**. This extends to the
multivariate case under a "restricted ANM" condition (the bivariate condition holds along every
edge conditioned on admissible sets), so the whole DAG — and in particular the set of correct
variable *orderings* — is identifiable. Concretely: the true orderings `Pi^0` are exactly the
permutations whose fully-connected DAG is a super-DAG of the (minimal) true graph; for nonlinear
ANMs this set is recoverable from the distribution, whereas for linear-Gaussian SEMs every ordering
is admissible and `Pi^0` carries no information.

**Topological ordering and triangular systems.** A DAG induces a topological ordering; given
a *correct* ordering `pi^0` of the variables, the model takes a triangular (autoregressive)
form

```
X^{pi}_j = sum_{k < j} f^{pi}_{j,k}(X^{pi}_k) + eps^{pi}_j ,   j = 1,...,p ,
```

in which each variable regresses only on those *earlier* in the order, so that per node what
remains is ordinary (nonlinear, multivariate) regression and variable selection among the earlier
variables. The role of an ordering in structure learning has been studied in the
structure-learning literature (e.g. Teyssier & Koller 2005; Schmidt et al. 2007).

**Nonparametric additive regression toolbox.** Fitting `X_j = sum_k f_{jk}(X_k) + eps` for smooth
`f_{jk}` is a solved problem: represent each `f_{jk}` in a spline / B-spline basis
`{b_1,...,b_{a_n}}`, fit by penalized least squares with smoothness penalties, or by boosting.
The number of basis functions `a_n` is the smoothness knob; classical theory uses `a_n ~ n^{1/5}`
for twice-differentiable functions. For high-dimensional additive models, sparse procedures select
which `f_{jk}` are nonzero: the Group Lasso treats each function's basis coefficients as a group
(Yuan & Lin 2006; Ravikumar et al. 2009), optionally with a sparsity-smoothness penalty (Meier et
al. 2009), and neighborhood-selection ideas from the linear case (Meinshausen & Buhlmann 2006)
carry over — regress each variable on all others and keep the selected predictors. These come with
screening guarantees (under compatibility / beta-min conditions, all truly relevant predictors are
retained with high probability).

## Baselines

**Constraint-based methods — PC / FCI (Spirtes, Glymour & Scheines 2000).** Test conditional
independencies `X_i ⫫ X_j | X_S` over subsets `S`, build the skeleton from the dependencies, then
orient v-structures and propagate Meek's rules. The assumptions are Markov + faithfulness, and the
output is the Markov equivalence class. Worst-case conditioning sets reach size `p-2`, and the
tests are conditional independence tests, including *nonparametric* ones.

**Score-based search — GES (Chickering 2002).** Assign each DAG a penalized-likelihood score
(BIC), greedily add / delete / reverse edges over the equivalence-class space until no neighbor
improves. For linear-Gaussian models the score is constant over a Markov equivalence class, so the
MEC is identified. The penalty performs two jobs — choosing the topology *and* selecting which
edges exist — within one search.

**LiNGAM (Shimizu et al. 2006).** Linear non-Gaussian SEM; recover the DAG via ICA on the data
matrix, using non-Gaussianity (Darmois-Skitovic) to orient edges, identifying the full DAG beyond
the MEC for linear mechanisms.

**Regression + residual-independence testing for ANMs (Mooij 2009; Peters, Mooij, Janzing &
Scholkopf 2014).** A practical algorithm that puts ANM identifiability to work. Its engine: a node
`X_i` is a *sink* of the true DAG iff its noise `N_i` is independent of all other variables;
estimate this by regressing each remaining variable on all the others and measuring the
*dependence* between the residual and the regressors (using a kernel independence test, HSIC, and
its p-value). The variable whose residual is *least* dependent is declared the current sink,
removed, and the procedure iterates — peeling off sinks back-to-front to build an ordering /
fully-connected DAG. A second phase then revisits each node and deletes incoming edges until the
residual stops being independent of the removed parent. With a consistent nonparametric regressor
and a perfect independence oracle this is provably correct, and it runs `O(p^2)` independence
tests. Each step uses a kernel independence test and a regression against all other variables.

## Evaluation settings

The natural yardsticks for observational causal discovery:

- **Simulated nonlinear ANMs.** Draw a random DAG (Erdos-Renyi with a target expected degree, or
  scale-free / Barabasi-Albert), sample smooth nonlinear mechanisms `f_{j,k}` (e.g. random
  functions / GP draws / sigmoids / polynomials), draw independent noise (Gaussian, exponential,
  Laplace; in the delicate Gaussian-noise regime), and generate `n` i.i.d. rows. Vary `p` (e.g.
  ~10-30 up to hundreds/thousands), graph topology, noise family, and the sample regime (including
  small-`n` / `p >> n`). The triangular structure makes the ground-truth ordering and edge set
  known.
- **Metrics on the directed edge set** (skeleton *and* direction must both be right): F1 of the
  directed edges (precision / recall over arrows), structural Hamming distance (SHD, the number of
  edge insertions / deletions / reversals to reach the truth, lower better), and — when the goal is
  intervention estimation rather than the exact graph — the structural intervention distance (SID),
  which counts how many intervention distributions the estimated graph gets wrong.
- **Baselines to compare against**: PC, conservative PC, GES, LiNGAM, and regression-against-all-
  others variants — the methods listed above.
- **Real data**: gene-expression / perturbation datasets where some interventional ground truth
  exists, as a sanity check beyond simulation.

Protocol: identical data across methods; report F1 / SHD / SID over many random graph replicates
and across the regimes (topology x noise x sample size).

## Code framework

The estimator plugs into a standard additive-regression and graph harness. The pieces that already
exist before the method: a nonparametric additive-model fitter (penalized regression splines /
boosting — `gam` from `mgcv`, `gamboost` from `mboost`), the data matrix, and a DAG /
adjacency-matrix representation. What is *not* settled is the procedure that turns the `n x p` data
matrix into a directed graph. That is the single empty slot.

```python
import numpy as np
from typing import Callable


def fit_additive_model(y: np.ndarray, X_parents: np.ndarray, num_basis: int = 10):
    """Pre-existing nonparametric additive regression: fit
    y ~ sum_k f_k(X_parents[:, k]) with smooth f_k (penalized regression
    splines / boosting). Returns fitted values and per-smooth significance
    (p-values), as a standard GAM fitter (e.g. mgcv::gam) already provides."""
    # ... existing GAM/spline machinery ...
    pass


def residual_variance(y: np.ndarray, X_parents: np.ndarray, num_basis: int = 10) -> float:
    """Variance of y after additive regression on its candidate parents.
    Empty parent set -> Var(y)."""
    if X_parents.shape[1] == 0:
        return float(np.var(y, ddof=1))
    fitted, _ = fit_additive_model(y, X_parents, num_basis)
    return float(np.var(y - fitted, ddof=1))


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """Input:  X of shape (n_samples, n_variables), observational, nonlinear ANM.
    Output: a p x p adjacency matrix of the estimated DAG.

    The pipeline that maps data -> DAG is unspecified here. Fill the slot below."""
    n, p = X.shape
    B = np.zeros((p, p))
    # Open slot: derive the data-to-DAG procedure.
    return B
```

The harness offers a nonparametric additive fitter and a graph container; the method fills the one
`run_causal_discovery` slot.
