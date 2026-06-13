## Research question

Recover the **directed acyclic graph** from purely observational data drawn from a nonlinear
additive-noise model (ANM), `X_j = f_j(parents(j)) + e_j`, with `f_j` nonlinear and the `e_j`
mutually independent. The single thing being designed is the function `run_causal_discovery(X)`: it
takes the data matrix and must return the *oriented* edge set — both the skeleton and every arrow's
direction. Everything around it (data generation, scoring, the three evaluation scenarios) is fixed.

The hard part is direction, not association. From `P_X` alone, second-order machinery (PC, GES on a
Gaussian likelihood) bottoms out at the *Markov equivalence class*: a set of DAGs sharing every
conditional independence, with some edges left unoriented, because for a linear-Gaussian SEM forward
and backward models are observationally identical. The escape hatch is the data itself — the
mechanisms here are *nonlinear*, and nonlinear additive noise makes the DAG identifiable (regressing
the effect on the cause leaves a residual independent of the cause; the wrong direction does not).
The ladder below is a sequence of fills of one function, each exploiting that asymmetry differently.

## Prior art before the first rung

The first rung is a *linear* non-Gaussian method dropped onto nonlinear data, so the lineage it
reacts to is the line of observational structure learners that came before identifiable direction:

- **PC (Spirtes & Glymour, 1991).** Tests conditional independencies and orients only v-structures,
  returning a CPDAG. *Gap:* leaves the equivalence class — many edges undirected — and under
  Gaussianity reads everything off the covariance, which is direction-blind.
- **GES (Chickering, 2002).** Greedy score-based search over DAGs with a penalized likelihood that is
  *constant across an equivalence class*. *Gap:* same equivalence-class ceiling; the score cannot
  break the forward/backward Gaussian symmetry, and the search is combinatorial.
- **LiNGAM / ICA-LiNGAM (Shimizu et al., 2006).** The first crack in the ceiling: with non-Gaussian
  noise the linear SEM `x = (I-B)^{-1}e` is an ICA model, whose mixing is identifiable up to
  permutation/scale, so directions become recoverable. *Gap:* relies on a non-convex ICA optimization
  (local minima, init/step/stop knobs) and scale-dependent permutations that flip under
  normalization — and it is *linear*, so on nonlinear mechanisms it mis-scores.

The first rung is the direct successor of that last line — non-Gaussianity exploited *without* ICA —
but still linear, which is precisely why it is the floor here.

## The fixed substrate

A small evaluation harness is frozen and must not be touched. `data_gen.py` simulates the ANM: it
draws a random DAG (Erdos-Renyi or scale-free, edges only from lower to higher node index so the
truth is a DAG), assigns each parent-child pair a random nonlinear scalar function (GP via random
Fourier features, a one-hidden-layer MLP, a low-degree polynomial, or a steep sigmoid), draws
additive noise (exponential / Laplace / uniform / Gaussian, centered), and generates each `X_j` in
topological order as `e_j` plus the summed parent contributions. `metrics.py` scores the *directed*
edge set: an edge is present when `|B[i,j]| > 0.01`; a true positive needs both skeleton and
direction right. `run_eval.py` calls `simulate_nonlinear_anm`, then `run_causal_discovery(X)`, then
`compute_metrics`. None of these is editable.

## The editable interface

Exactly one function is editable — `run_causal_discovery` in `bench/custom_algorithm.py` (lines
3–14 of the template). The contract is rigid and load-bearing for every rung:

- **Input:** `X` of shape `(n_samples, n_variables)` — the observational data only. No graph type,
  no node count beyond `X.shape`, no noise family, no function family is passed in. The method must
  be agnostic to all of those (the scenarios deliberately vary them).
- **Output:** an adjacency matrix `B` of shape `(n_variables, n_variables)` with the **causal-learn
  convention** `B[i, j] != 0` meaning `j -> i` (row = child, column = parent). The harness reads a
  nonzero entry as a directed edge; magnitudes above `0.01` count, so a 0/1 matrix is fine.
- **Reproducibility:** the seed is read from the `SEED` environment variable inside the function
  (`int(os.environ.get("SEED", "42"))`), not passed as an argument.

The starting point is the scaffold default: **return no edges** — an empty graph.

```python
import numpy as np

# =====================================================================
# EDITABLE: implement run_causal_discovery below
# =====================================================================
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    n = X.shape[1]
    return np.zeros((n, n))
# =====================================================================
```

Each rung replaces exactly this function body and nothing else; the available libraries are the
harness's `causal-learn`, plus `numpy`, `scipy`, and `scikit-learn`.

## Evaluation settings

Three scenarios, each over three seeds {42, 123, 456}, chosen to vary every axis at once so a method
cannot key on one combination:

| Label          | Graph type          | Nodes | Samples | Noise       | Nonlinearity |
|----------------|---------------------|-------|---------|-------------|--------------|
| SF20-GP        | Scale-Free (BA)     | 20    | 2000    | Exponential | GP           |
| ER20-Gauss     | Erdos-Renyi         | 20    | 2000    | Gaussian    | Mixed        |
| ER12-LowSample | Erdos-Renyi         | 12    | 150     | Laplace     | Mixed        |

ER20-Gauss is the delicate case (ANM identifiability with Gaussian noise leans entirely on the
nonlinearity); ER12-LowSample is the small-`n` stress test. Four metrics per scenario: **F1**
(primary, higher is better), **SHD** (lower is better), **precision** and **recall** (higher is
better), all on the directed edge set.
