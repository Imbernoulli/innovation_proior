## Research question

Given a data matrix `X ∈ R^{n×d}` of `n` i.i.d. observations of a `d`-dimensional random
vector `X = (X_1, ..., X_d)`, recover a directed acyclic graph (DAG, equivalently a Bayesian
network) for the joint distribution `P(X)`. We model `X` by a linear structural equation model
(SEM) parameterized by a weighted adjacency matrix `W ∈ R^{d×d}`: each variable is a linear
function of its parents plus independent noise, `X_j = w_j^T X + z_j`, where `w_j` is the `j`-th
column of `W` and the noise `z = (z_1, ..., z_d)` is *not* assumed Gaussian. The support of `W`
(its nonzero entries) is the graph: `W_{ij} ≠ 0` means there is an edge from node `i` to node
`j`. The task is to choose `W` so that (i) it fits the data under a score function, and (ii) the
induced graph is acyclic.

The acyclicity requirement is the central challenge. The score is a smooth function of a
real matrix and is easy to optimize on its own. But "the graph induced by `W` is a DAG" is a
*combinatorial* constraint on the support pattern of `W`, and the number of DAGs on `d` nodes
grows superexponentially in `d`, so one cannot enumerate or search this set directly. Learning a
DAG that minimizes a decomposable score is NP-hard (Chickering 1996; Chickering, Heckerman &
Meek 2004). The question is how to optimize a smooth score over the space of weighted adjacency
matrices while satisfying the acyclicity constraint.

## Background

The basic object is the linear SEM. With `W = [w_1 | ... | w_d]`, the model `X_j = w_j^T X + z_j`
says column `j` of `W` holds the incoming edge weights of node `j` (the regression coefficients
of `X_j` on the other variables). The matrix `A(W) ∈ {0,1}^{d×d}` with `[A(W)]_{ij} = 1 ⟺ W_{ij}
≠ 0` is the binary adjacency matrix of the directed graph `G(W)`. Learning the structure means
learning the support of `W`; learning the parameters means learning its values.

**The least-squares score and its statistical backing.** For a linear SEM the natural fit
measure is the squared reconstruction error of regressing every variable on all the others
simultaneously,

```
ℓ(W; X) = (1 / 2n) · ‖X − X W‖_F^2 ,
```

with gradient `∇ℓ(W) = −(1/n) X^T (X − X W)`. The statistical properties of this loss for
scoring DAGs are well established under the assumptions of those analyses: its minimizer recovers a true DAG with high probability in
finite samples and in high dimensions (`d ≫ n`), and the recovery is consistent for *both*
Gaussian SEM (van de Geer & Bühlmann 2013; Aragam, Amini & Zhou 2016) and non-Gaussian SEM
(Loh & Bühlmann 2014). These results also show the faithfulness assumption is not needed in this
setup. The upshot for method design: the statistical question of *which* `W` is right is settled
for the least-squares score; what remains open is the *computational* problem of actually finding
a good-scoring `W` whose graph is acyclic. To learn a sparse graph one adds an `ℓ1` penalty,
`F(W) = ℓ(W; X) + λ ‖W‖_1` with `‖W‖_1 = ‖vec(W)‖_1`.

**What makes the acyclicity constraint hard.** The space of DAGs is discrete and enormous
(superexponential in `d`; Robinson 1977), so the constraint `G(W) ∈ DAGs` cannot be written as a
simple algebraic condition on `W` that is amenable to gradient-based optimization. Existing
approaches therefore enforce acyclicity *operationally* — by searching the discrete space in a
way that only ever visits acyclic structures — rather than as a constraint a continuous solver
could handle.

**A combinatorial fact about walks.** There is a classical and elementary identity relating
matrix powers to graph structure (e.g. Harary & Manvel 1971): for a binary adjacency matrix `B`,
the entry `(B^k)_{ii}` counts the number of closed walks of length `k` that start and end at node
`i`, so `tr(B^k) = Σ_i (B^k)_{ii}` is the total number of length-`k` closed walks in the graph.
Consequently a directed graph is acyclic exactly when it has no closed walks of any length, i.e.
`tr(B^k) = 0` for every `k ≥ 1`. This is a true statement about graphs that long predates any
particular learning algorithm; it links the *combinatorial* property of acyclicity to *algebraic*
quantities computed from the adjacency matrix. How — or whether — such an identity can be turned
into something a continuous optimizer can use is not given by the identity itself.

**The analogy that motivates the whole enterprise.** Two neighboring problems were transformed
by being recast as closed-form continuous programs. For *undirected* graphical models (Markov
networks), structure learning over Gaussian data is a convex log-determinant program (Yuan & Lin
2007; Banerjee, El Ghaoui & d'Aspremont 2008), which can be handed to black-box convex solvers
and which sparked a wave of fast algorithms (the graphical lasso, Friedman, Hastie & Tibshirani
2008; QUIC, Hsieh et al. 2014). For deep networks, the major tool was stochastic gradient descent
on a closed-form differentiable objective. In both cases progress came from writing the learning
problem as a smooth/continuous program and delegating the heavy lifting to general-purpose
optimization.

## Baselines

**Exact score-based search (Singh & Moore 2005; Silander & Myllymäki 2006; Cussens 2012;
Cussens, Haws & Studený 2017).** Optimize a decomposable discrete score `Q(G)` (BDe, BGe, BIC,
MDL) over DAGs to *global* optimality via dynamic programming or integer linear programming
(e.g. the GOBNILP solver). Core idea: enumerate candidate parent sets per node and solve a
cutting-plane / DP problem for the best acyclic combination.

**Greedy / local search — greedy equivalence search and its fast variant (Chickering 2003;
Ramsey et al. 2016).** GES searches over Markov equivalence classes (CPDAGs) rather than DAGs:
a forward phase greedily inserts the single edge that most improves the score, each insertion
implemented as a move on the equivalence class; then a backward phase greedily deletes edges
until no deletion helps. Under score-equivalence and local consistency of the score, GES returns
the true equivalence class in the large-sample limit. The fast greedy variant (FGS) scales this
to very large problems and is the strongest practical baseline. It operates *locally*,
adding or removing one edge (or one equivalence-class move) at a time and checking acyclicity at
each step. Local methods of this family also typically lean on structural assumptions —
bounded in-degree, bounded treewidth, edge constraints — which real networks with highly
connected hub nodes (scale-free / small-world topologies; Watts & Strogatz 1998; Barabási &
Albert 1999) may violate.

**Order-based search (Teyssier & Koller 2005; Scanagatta et al. 2015, 2016).** Sidestep the
acyclicity constraint by searching over topological *orderings*: any fixed ordering admits only
acyclic structures, so the inner problem becomes parent selection per node.

**Coordinate-descent and penalized-regression DAG learners (Fu & Zhou 2013; Aragam & Zhou 2015;
Gu, Fu & Zhou 2018).** Use continuous penalties (e.g. concave or `ℓ1`) on the SEM coefficients,
optimized one block/coordinate at a time, with explicit acyclicity enforcement interleaved into
the descent (checking for and forbidding cycle-inducing updates).

**Constraint-based and non-Gaussian-specific methods (PC algorithm, Spirtes, Glymour & Scheines
2000; LiNGAM, Shimizu et al. 2006).** PC recovers a CPDAG from conditional-independence tests.
LiNGAM exploits non-Gaussian noise (via independent component analysis) to identify the full DAG
rather than just its equivalence class.

## Evaluation settings

The natural yardsticks for a structure-learning method on synthetic linear SEM data:

- **Random graph ensembles.** Erdős-Rényi (ER) graphs, edges added independently, simulated with
  `d`, `2d`, `4d` expected edges (ER-1/2/4); and scale-free (SF) graphs from the Barabási-Albert
  preferential-attachment process (with `4d` edges), which produce hub nodes and high in-degree
  — the regime that stresses local search. Node counts `d ∈ {10, 20, 50, 100}`.
- **Sample sizes** spanning low- and high-dimensional regimes, `n ∈ {20, 1000}` (so both `n ≫ d`
  and `n ≲ d`).
- **Noise models** for the SEM `X = W^T X + z`: standard Gaussian, Exponential(1), and
  Gumbel(0,1) — i.e. one Gaussian and two non-Gaussian families, to test distributional
  agnosticism. Edge weights drawn from `Unif([−2,−0.5] ∪ [0.5,2])`.
- **Metrics** computed against the true directed edge set, where both presence and orientation
  must be correct: structural Hamming distance (SHD = additions + deletions + reversals), false
  discovery rate, true positive rate, false positive rate. Reversed edges are scored as errors
  distinct from true positives. For a baseline that returns a CPDAG with undirected edges, those
  must be reconciled against the true directed graph before comparison.
- **A global-optimum reference.** For small graphs (`d = 10`), an exact ILP solver provides the
  globally optimal score, so an approximate method's score can be compared to the true optimum.
- **A real benchmark.** The Sachs et al. (2005) protein-signaling dataset (`n = 7466`, `d = 11`,
  with a biologically curated consensus network of ~20 edges), a standard graphical-models
  benchmark with a gold-standard graph.

## Code framework

The substrate is a generic continuous-optimization harness: a smooth score on a real matrix `W`,
fed to a standard numerical solver. What does *not* yet exist is any way to express the acyclicity
requirement so that the solver can respect it — that single piece is the open slot. The pieces
that already exist are the linear-SEM least-squares loss and its gradient, an `ℓ1` penalty, a
standard constrained-optimization driver (an augmented-Lagrangian / penalty loop wrapped around a
smooth unconstrained inner solver such as L-BFGS), and a final rounding step. The scalar
acyclicity constraint and its gradient are left as stubs.

```python
import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def continuous_dag_solver(X, lambda1, loss_type, max_iter=100, h_tol=1e-8,
                          rho_max=1e16, w_threshold=0.3):
    """Generic continuous constrained-optimization harness for a weighted adjacency matrix.
    Column j of W are the incoming weights of node j."""
    n = X.shape[0]
    d = X.shape[1]

    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _acyclicity_constraint(W):
        # TODO: define the scalar acyclicity constraint and its gradient.
        raise NotImplementedError

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        constr, G_constr = _acyclicity_constraint(W)
        obj = loss + 0.5 * rho * constr * constr + alpha * constr + lambda1 * w.sum()
        G_smooth = G_loss + (rho * constr + alpha) * G_constr
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est, rho, alpha, constr = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        while rho < rho_max:
            # TODO: once _acyclicity_constraint exists, solve the smooth bound-constrained subproblem.
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            constr_new, _ = _acyclicity_constraint(_adj(w_new))
            if constr_new > 0.25 * constr:
                rho *= 10
            else:
                break
        w_est, constr = w_new, constr_new
        alpha += rho * constr
        if constr <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0.0   # round the numerical solution
    return W_est
```

The empty slot is the missing bridge between a smooth score over real matrices and the discrete
acyclicity requirement; the surrounding optimization machinery is standard once that bridge exists.
