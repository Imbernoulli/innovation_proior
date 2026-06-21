# Context: learning directed acyclic graphs from observational data (circa 2018-2019)

## Research question

Given a data matrix `X ∈ R^{n×d}` of `n` i.i.d. observations of `d` variables
`X = (X_1, ..., X_d)`, recover the directed acyclic graph (DAG) that generated them — the
graph whose edge `X_k → X_j` means `X_k` is a direct cause (parent) of `X_j` in a structural
equation model `X_j = f_j(X_{pa(j)}) + z_j`. The data are purely observational (no
interventions), and the structural functions `f_j` are *nonlinear*: each variable is some
unknown nonlinear function of its parents plus independent noise.

The acyclicity requirement makes the feasible set the discrete space of DAGs, whose size grows
superexponentially in `d` (even a single fixed topological order admits `2^{d(d-1)/2}` acyclic
subgraphs, and the total number of DAGs is larger still). The problem is NP-hard. Practical
methods enforce acyclicity through discrete devices — sequentially adding edges and checking for
cycles, or searching over the `d!` topological orderings.

## Background

**Structural equation models and the DAG-learning problem.** A DAG model factorizes a joint
distribution `P(X) = ∏_j P(X_j | X_{pa(j)})`. A structural equation model (SEM) makes this
mechanistic: each variable is generated from its parents,
`E[X_j | X_{pa(j)}] = g_j(f_j(X))`, where `f_j` depends only on the parent coordinates and
`g_j` is a (typically known) link. The additive-noise special case is `X_j = f_j(X) + z_j`
with `z_j` independent of `f_j(X)` and `E f_j(X) = 0`. Learning the DAG means learning, for
each `j`, *which* coordinates `f_j` actually depends on — its parent set — subject to the whole
thing being acyclic.

**Identifiability — why observational nonlinear data carry direction at all.** It is a
classical fact that a jointly Gaussian linear SEM is *not* identifiable from observational data:
the `f_j` are linear, and many different DAGs (a whole Markov equivalence class) induce the same
Gaussian distribution, so direction cannot be read off the data. Two escapes from this
degeneracy were established before this work. With additive noise, if the `f_j` are *linear but
the noise is non-Gaussian*, the model is identifiable (the LiNGAM line:
Kagan-Linnik-Rao 1973; Shimizu et al. 2006; Loh & Bühlmann 2014). And if the `f_j` are
*nonlinear* with additive noise — the nonlinear additive noise model (ANM) of Hoyer, Janzing,
Mooij, Peters & Schölkopf (2009) and Peters, Mooij, Janzing & Schölkopf (JMLR 2014) —
then under mild conditions (the `f_j` three-times differentiable and not linear in any argument)
the DAG is generically identifiable; Corollary 31 of Peters et al. 2014 is the operative
statement. Intuitively, fitting `X_j` on `X_k` and `X_k` on `X_j` leaves residuals that are
independent of the regressor in only one direction when the dependence is genuinely nonlinear,
breaking the cause/effect symmetry that the Gaussian-linear case suffers from. So nonlinearity
is not just a modeling nuisance — it is the very thing that makes the direction recoverable from
observational data.

**A continuous handle on acyclicity exists for the linear case.** The single most important
piece of prior art for this problem is a *smooth, exact* algebraic characterization of
acyclicity over weighted adjacency matrices (Zheng, Aragam, Ravikumar & Xing, NeurIPS 2018). Its
construction is worth stating in full because it is the foundation everything else is built on.
For a binary adjacency matrix `B ∈ {0,1}^{d×d}`, the diagonal entry `(B^k)_{ii}` counts the
length-`k` closed walks from node `i` back to itself, so `B` is acyclic iff `(B^k)_{ii} = 0` for
all `k ≥ 1` and all `i`, i.e. `tr(B^k) = 0` for all `k`. One could try to collapse all powers
into one number with the Neumann series `tr((I-B)^{-1}) = tr(∑_{k≥0} B^k) = d + ∑_{k≥1} tr(B^k)`,
so `tr((I-B)^{-1}) = d` iff acyclic — but this needs the spectral radius `r(B) < 1`, which is a
strong, generally-violated condition, and the series is badly ill-conditioned because the
walk-counts `tr(B^k)` blow up. A finite version `∑_{k=1}^{d} tr(B^k) = 0` drops the spectral
condition but the entries of `B^k` overflow machine precision even for small `d`. The resolution
is the matrix exponential: `tr(e^B) - d = ∑_{k≥1} tr(B^k)/k!`, which is zero iff acyclic, is
defined and convergent for *every* square matrix, and is numerically tame because the `1/k!`
weights crush the long-walk terms that exploded before. To extend from a nonnegative matrix to a
real weighted matrix `W ∈ R^{d×d}` (whose nonzero pattern is the graph), one squares entrywise
via the Hadamard product so all "edge weights" are nonnegative:

```
h(W) = tr( e^{W ∘ W} ) - d,      [W ∘ W]_{kj} = w_{kj}^2,
```

which is `0` iff `G(W)` is a DAG. Its value quantifies "DAG-ness" — it counts *weighted* closed
walks, so a larger `h` means more or more-heavily-weighted cycles — and it has a clean gradient

```
∇h(W) = ( e^{W ∘ W} )^T ∘ 2W.
```

This single device replaces the combinatorial constraint `G(W) ∈ DAG` with one smooth equality
`h(W) = 0`, so the linear-SEM learning problem
`min_W (1/2n)||X - XW||_F^2 + λ||W||_1` s.t. `h(W) = 0` can be handed to a generic constrained
solver. The characterization lives on a weighted adjacency matrix `W`, which is the
regression-coefficient matrix for a linear SEM. A general nonlinear SEM `X_j = f_j(X) + z_j`
has no such coefficient matrix.

**Measuring nonlinear dependence.** Two strands of prior work bear on representing nonlinear
`f_j`. First, nonparametric variable selection via *partial derivatives*: a smooth function's
sensitivity to an input coordinate is captured by the size of its partial derivative in that
coordinate, an idea used for nonparametric sparsity (Rosasco, Villa, Mosci, Santoro & Verri,
JMLR 2013) and in sparse additive models with orthogonal-series bases (Ravikumar, Lafferty, Liu
& Wasserman, JRSS-B 2009). Second, flexible function approximation: a multilayer perceptron
`MLP(u; A^{(1)},...,A^{(h)}) = σ(A^{(h)} σ(··· σ(A^{(1)} u)))` with enough hidden units
approximates any sufficiently smooth function arbitrarily well, and is differentiable in both
its input and its weights — so quantities derived from it can be optimized by gradient methods.

## Baselines

**Linear NOTEARS (Zheng et al. 2018).** As above: the trace-exponential constraint `h(W) = 0`
plus least-squares score and `ℓ_1` penalty, solved by augmented Lagrangian with an L-BFGS-B inner
solver and a final hard-threshold on `|W|`. The SEM is assumed linear: `f_j(X) = w_j^T X`.

**Causal Additive Models, CAM (Bühlmann, Peters & Ernest, Annals of Statistics 2014).** Assumes
an additive structure `f_j(X) = ∑_{k∈pa(j)} f_{jk}(X_k)`, each `f_{jk}` a smooth univariate
function fit by spline/GAM regression. It runs in three separate stages: a preliminary
neighborhood selection to prune candidate parents, a greedy search over the variable *ordering*
maximizing the likelihood, and a final edge-pruning pass.

**Greedy equivalence search with generalized (kernel) scores, GS (Huang et al. 2018).** A
greedy search that adds/removes/reverses edges to improve a kernel-based generalized score, with
no parametric model assumption, using kernel measures of conditional dependence.

**DAG-GNN (Yu et al. 2019).** Learns a (noisy) nonlinear transform of a linear-SEM latent
structure with a variational graph-neural-network autoencoder, using a polynomial surrogate
`tr((I + A/d)^d) - d` for the acyclicity constraint.

**Fast greedy equivalence search, FGS (Ramsey et al. 2017).** A scalable greedy
equivalence-search variant assuming linear-Gaussian scores; returns a CPDAG (a Markov equivalence
class with possibly undirected edges).

## Evaluation settings

The natural yardsticks already in use for nonlinear DAG recovery, all pre-method facts:

- **Synthetic graphs.** Ground-truth DAGs drawn from two random-graph models: Erdős-Rényi (ER)
  and scale-free / Barabási-Albert (SF), with expected edge count `s_0 ∈ {d, 2d, 4d}` (denoted
  ER1/ER2/ER4 etc.), node counts `d ∈ {10, 20, 40}`, and sample sizes `n ∈ {1000, 200}` to span
  ample and scarce data.
- **Structural function families.** Given a sampled DAG, simulate `X_j = f_j(X_{pa(j)}) + z_j`
  in topological order with `z_j` i.i.d. noise (Gaussian in the simulations; the harness also
  varies exponential and Laplace), and four mechanisms for `f_j`: additive Gaussian-process
  draws (RBF kernel, length-scale 1), index models `∑_{m=1}^{3} h_m(β_{jm}^T X)` with
  `h ∈ {tanh, cos, sin}`, a randomly-initialized MLP (one hidden layer of 100 sigmoid units),
  and a full Gaussian-process draw. The point is to stress a method across nonlinearity types,
  topologies, noise laws, and sample regimes.
- **Metrics, on the directed edge set** (both skeleton and orientation must be right): structural
  Hamming distance (SHD, lower better), false discovery rate (FDR), true positive rate (TPR),
  false positive rate (FPR); the task harness scores F1 (primary), SHD, precision, recall.
  Methods that return a CPDAG with undirected edges are scored by orienting undirected edges
  favorably when possible.
- **Real data.** The Sachs et al. (2005) flow-cytometry protein-signaling dataset: `n = 7466`
  continuous measurements of `d = 11` proteins/phospholipids, with a community-accepted consensus
  network as ground truth.

## Code framework

The method will plug into a standard PyTorch training harness: a per-variable regression model
whose parameters are learned by a constrained-optimization loop. The generic machinery already
exists — a model object that maps the `d` inputs to `d` predicted values, a smooth least-squares
loss, an `ℓ_1` sparsity penalty, a bound-constrained quasi-Newton solver (L-BFGS-B) for the
inner unconstrained subproblem, and the augmented-Lagrangian outer loop that drives an equality
constraint to zero. The acyclicity function `h(·) = tr(e^{·∘·}) - d` is a known primitive and
is shown as a fixed component. The `NodeRegressors` class — the per-variable model family, its
forward pass, the nonnegative `d × d` adjacency it returns, and its sparsity penalty — is left
as the open slot.

```python
import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as slin


class TraceExpm(torch.autograd.Function):
    """Autograd primitive for tr(exp(A)); gradient wrt A is exp(A)^T."""
    @staticmethod
    def forward(ctx, input):
        E = slin.expm(input.detach().numpy())          # matrix exponential, O(d^3)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(np.trace(E), dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (E,) = ctx.saved_tensors
        return grad_output * E.t()


trace_expm = TraceExpm.apply


def h_acyclic(A: torch.Tensor) -> torch.Tensor:
    """Known smooth acyclicity measure: h(A) = tr(exp(A)) - d, zero iff A encodes a DAG.
    Expects a NONNEGATIVE d x d matrix A (entrywise) whose support is the candidate graph.
    (Zheng et al. 2018.)"""
    d = A.shape[0]
    return trace_expm(A) - d


class NodeRegressors(nn.Module):
    """Maps X -> X_hat, predicting each variable from all variables.
    The function family for each f_j, and the rule that turns its parameters into the
    nonnegative matrix fed to h_acyclic, are exactly what this method must design."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        # TODO: the per-variable model family we will design (parameters theta).
        pass

    def forward(self, x):                       # [n, d] -> [n, d]
        # TODO: predict each X_j from the other variables with the chosen model.
        pass

    def adjacency(self) -> torch.Tensor:        # -> [d, d], nonnegative
        # TODO: the object we will define here -- a real, nonnegative d x d matrix,
        #       built from the model parameters, that h_acyclic can consume.
        pass

    def l1_penalty(self) -> torch.Tensor:
        # TODO: sparsity penalty on the relevant parameters.
        pass


def squared_loss(output, target):
    n = target.shape[0]
    return 0.5 / n * torch.sum((output - target) ** 2)


def dual_ascent_step(model, X, lambda1, rho, alpha, h_prev, rho_max):
    """One augmented-Lagrangian subproblem: minimize
    loss + (rho/2) h^2 + alpha h + lambda1 * l1   over the model parameters,
    then update the penalty rho and the multiplier alpha. (Inner solver: L-BFGS-B.)"""
    optimizer = LBFGSB(model.parameters())      # bound-constrained quasi-Newton (existing)
    X_t = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            loss = squared_loss(model(X_t), X_t)
            h = h_acyclic(model.adjacency())
            penalty = 0.5 * rho * h * h + alpha * h
            obj = loss + penalty + lambda1 * model.l1_penalty()
            obj.backward()
            return obj
        optimizer.step(closure)
        with torch.no_grad():
            h_new = h_acyclic(model.adjacency()).item()
        if h_new > 0.25 * h_prev:               # constraint not shrinking fast enough
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def learn_dag(model, X, lambda1=0.01, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W = model.adjacency().detach().numpy()      # extract the learned weighted adjacency
    W[np.abs(W) < w_threshold] = 0              # round numerical near-zeros / cut weak edges
    return W
```

The open slots are the model family for each `f_j`, the map from its parameters to a real
nonnegative `d × d` matrix for `h_acyclic`, and the sparsity penalty.
