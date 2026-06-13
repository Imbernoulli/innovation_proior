## Research question

We observe a sample of continuous-valued vectors `x = (x_1, ..., x_m)` and we want the *causal*
structure that generated them — not just the joint distribution, but a directed graph that says
which variable is a cause of which, together with the strengths of those causal links. The
practical stakes are high: a causal model is what lets you predict the effect of an
*intervention* (forcing `x_j` to a value), whereas a purely associational model cannot.
Controlled experiments would settle causality directly, but in many domains they are impossible
or prohibitively expensive, so we are stuck with purely *observational* data and must ask: under
what assumptions, and by what procedure, can the full causal structure — direction *and*
coefficients — be recovered from observation alone?

The hard part is direction. The data-generating process is taken to be linear, recursive (it can
be written so that no later variable causes an earlier one, i.e. it is a directed acyclic graph),
with each variable a weighted sum of its causal parents plus its own independent noise, and with
no hidden common causes. The goal is a procedure that, given enough samples, returns the entire
weighted DAG with no undetermined parameters and *without* being told a time order or any other
ordering of the variables in advance, and that remains usable when the number of variables is in
the tens or hundreds rather than a handful.

## Background

The field has formalized causality with probability distributions on directed acyclic graphs
(Spirtes, Glymour & Scheines 2000; Pearl 2000). A DAG `G` over the variables encodes a set of
conditional-independence statements through d-separation, and the **causal Markov condition**
says the data distribution factorizes according to `G`. To go from observed independencies back
to a graph, one additionally assumes **faithfulness** (Spirtes et al. 2000): every conditional
independence in the distribution is one that `G`'s structure forces, none arising by accidental
cancellation of parameters. Under Markov + faithfulness, the **PC** and **GES** families of
algorithms search for a graph whose d-separations match the conditional independencies estimated
from data.

The structural-equation-model (SEM) tradition (Bollen 1989) writes the same linear recursive
process algebraically: stacking the equations `x_i = sum_{j} b_ij x_j + e_i` gives `x = Bx + e`,
where `B` holds the connection strengths and is, under the unknown causal order, *permutable to
strictly lower triangular* (lower triangular with an all-zero diagonal) — that strict-triangular
form is exactly the algebraic signature of acyclicity. Solving, `x = (I - B)^{-1} e`. In this
tradition the disturbances `e_i` are taken (explicitly or implicitly) to be Gaussian, and
estimation rests on second-order statistics — the data covariance matrix.

The load-bearing diagnostic fact about this landscape is what Gaussianity costs you. If the `e_i`
are Gaussian, the joint distribution of `x` is multivariate Gaussian, and *everything* observable
about it is in its mean and covariance matrix. But many distinct causal graphs produce the same
covariance. The cleanest case is two variables. Model 1: `x_1 = e_1`, `x_2 = 0.8 x_1 + e_2`,
with `var(e_1)=1`, `var(e_2)=0.36`, so `var(x_1)=var(x_2)=1` and `cov(x_1,x_2)=0.8`. Model 2:
`x_1 = 0.8 x_2 + e_1`, `x_2 = e_2`, with `var(e_1)=0.36`, `var(e_2)=1`, giving the *identical*
means (0), variances (1), and covariance (0.8). The two `B` matrices — one with `b_21=0.8`, the
other with `b_12=0.8` — are completely different causal stories, yet under Gaussian noise they
yield the same distribution. No method that reads only the covariance can prefer `x_1 -> x_2`
over `x_1 <- x_2`. More generally, with Gaussian noise the search returns only a *Markov
equivalence class* of graphs (summarized as a partially directed graph): some edges stay
undirected because their orientation cannot be determined, and the connection strengths are not
uniquely identified. Constraint-based discovery on Gaussian data hits this wall by construction.

Two further pieces of the landscape matter. First, there is evidence that the symmetry can be
broken if one looks past the covariance: with *non-Gaussian* noise, the two-variable cause/effect
direction becomes distinguishable using higher-order statistics (Dodge & Rousson 2001 relate the
asymmetry to the third moment in a regression; Shimizu & Kano 2006 use non-normality in SEM to
recover the direction of causation). Second, there is a well-developed body of statistical theory
about recovering a *linear mixture of independent non-Gaussian sources* — independent component
analysis — summarized next as a baseline technique.

## Baselines

These are the methods and techniques on the table that a new procedure would build on or be
measured against.

**Constraint-based discovery (PC), and score-based (GES) (Spirtes, Glymour & Scheines 2000;
Pearl 2000; Chickering 2002).** PC estimates the set of conditional independencies among the
variables (by repeated independence tests), recovers the undirected skeleton, then orients edges
where the independence pattern forces an orientation (colliders, then Meek-style propagation).
GES greedily adds/removes edges to optimize a penalized likelihood score. Both are principled and
scale to many variables. **Gap:** on continuous linear-Gaussian data they recover only the Markov
equivalence class. Where two graphs imply the same conditional-independence set (the
two-variable example above, and in general any pair of orientations not pinned by a collider),
they return an undirected or ambiguous edge; the connection strengths are not identified. Their
information source — conditional independence, equivalently the covariance in the Gaussian case —
simply does not contain the direction in those cases.

**Gaussian SEM / covariance-structure analysis (Bollen 1989).** Fit `x = Bx + e` by matching the
model-implied covariance `(I-B)^{-1} D (I-B)^{-T}` (with `D = cov(e)` diagonal) to the sample
covariance, typically by maximum likelihood under joint Gaussianity. **Gap:** the fit depends on
`B` only through the implied covariance, and many `B` (with different orientations) imply the same
covariance, so the model is fit-equivalent across causal directions; a pre-specified ordering or
other external knowledge is needed to pick one.

**Independent component analysis (ICA) (Comon 1994; Hyvärinen 1999; Hyvärinen, Karhunen & Oja
2001).** ICA models an observed vector as `x = A s` where the latent sources `s_j` are mutually
independent, at most one source is Gaussian, and `A` is an unknown invertible mixing
matrix; the task is to recover `A` (equivalently the separating matrix `W = A^{-1}`) and the
sources, using only the data. The crucial identifiability result (Comon 1994): under this source
condition and invertible `A`, `A` is identifiable **up to permutation, scaling, and sign of its columns**
— there is no rotational ambiguity, unlike the Gaussian case where any orthogonal rotation of `A`
gives the same distribution. The estimators maximize non-Gaussianity of the recovered components.
A standard, efficient one is **FastICA** (Hyvärinen 1999): whiten the data so `E{zz^T}=I`; for a
unit-norm `w`, measure the non-Gaussianity of `w^T z` by an approximation to negentropy
`J_G(w) = (E{G(w^T z)} - E{G(nu)})^2`, with `G` a non-quadratic contrast (the general-purpose
choice `G(u) = (1/a) log cosh(a u)`, derivative `g(u) = tanh(a u)`); maximize it by the
fixed-point Newton iteration `w+ = E{z g(w^T z)} - E{g'(w^T z)} w`, renormalize `w <- w+/||w+||`,
and estimate all components together with a symmetric decorrelation `W <- (W W^T)^{-1/2} W`.
FastICA estimates super- and sub-Gaussian components without knowing the source densities, and
converges fast (quadratically, even cubically for symmetric densities). **Gap as a causal tool:**
ICA alone does not solve a causal problem — it returns `W` with the rows in an arbitrary order and
each row arbitrarily scaled and signed, so on its own it gives no correspondence between recovered
components and observed variables, and no notion of acyclic ordering. And the non-Gaussianity
contrast is non-convex: a fixed-point run from a poor initialization can settle in a local
optimum. A standard combinatorial primitive is also available: the **linear assignment problem**,
which minimizes a sum of one-to-one matching costs and is solved in `O(m^3)` by the Hungarian
algorithm (Kuhn 1955; Burkard & Cela 1999).

**Two-variable non-Gaussian direction results (Dodge & Rousson 2001; Shimizu & Kano 2006).** For
*two* variables these works show the cause/effect direction is recoverable from non-Gaussianity
(via third-moment / non-normality asymmetries in the regression). **Gap:** they settle the
two-variable case but do not give a general procedure that returns the entire weighted DAG, with
ordering and all coefficients, for `m` variables at once.

## Evaluation settings

The natural yardstick is synthetic data drawn from a known ground-truth DAG so that the recovered
graph can be scored against truth. The standard protocol: randomly construct a strictly
lower-triangular coefficient matrix `B` over `m` variables at a chosen sparsity (varying the
number of zero edges and the dimension, e.g. `m` in `{3,5,10,20,100}`); draw each disturbance
`e_i` independently from a chosen non-Gaussian distribution (for instance Gaussian samples passed
through a sign-preserving power nonlinearity, or Laplace / exponential / uniform noise) and scale
to chosen variances; generate `n` samples through the recursive process (`n` from a few hundred to
a few thousand); then *randomly permute the variable order* in the data matrix to hide the
generating causal order from the algorithm. Standard graph families for the ground-truth DAG are
Erdős–Rényi random graphs and Barabási–Albert scale-free graphs. The recovered directed edge set
is compared to the true edge set; common metrics are the count of true/false positives and
negatives over the strictly-lower-triangular (directed) edges, summarized as precision, recall,
F1 (skeleton *and* orientation must both be correct for an edge to count), and structural Hamming
distance (SHD, total edge insertions/deletions/reversals to match truth).

## Code framework

A standard numerical/ML stack is already available: `numpy` for linear algebra, an off-the-shelf
**ICA** routine (`sklearn.decomposition.FastICA`) that returns a separating matrix, a
linear-assignment solver (`scipy.optimize.linear_sum_assignment`), and ordinary/penalized linear
regression (`sklearn.linear_model`) for fitting coefficients. The benchmark harness hands the
method an `n x m` data matrix and expects back an `m x m` adjacency matrix `B` with `B[i, j] != 0`
read as the directed edge `j -> i`.

What is *not* settled is the procedure that turns the raw data into that adjacency matrix — that
is the one empty slot. The scaffold is a single function with the I/O fixed and the body left to
design:

```python
import numpy as np


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """Recover a directed weighted DAG from observational data.

    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables),
            B[i, j] != 0 read as the directed edge j -> i.

    The available primitives (all pre-existing, generic):
      - an ICA routine that returns a separating matrix W (rows arbitrary
        in order, scale, and sign): sklearn.decomposition.FastICA
      - a linear-assignment solver: scipy.optimize.linear_sum_assignment
      - linear / penalized regression: sklearn.linear_model
    """
    # TODO: the procedure we will design — turn X into the directed
    #       weighted DAG using the primitives above.
    pass
```
