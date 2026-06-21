## Research question

Recover the **directed acyclic graph** from purely observational data drawn from a nonlinear
additive-noise model (ANM), `X_j = f_j(parents(j)) + e_j`, with `f_j` nonlinear and the `e_j`
mutually independent. The single thing being designed is `run_causal_discovery(X)`: it takes the
data matrix and returns the oriented edge set — both skeleton and every arrow's direction.

The hard part is direction, not association. For a linear-Gaussian SEM, forward and backward models
are observationally identical, so second-order methods bottom out at the *Markov equivalence class*.
Here the mechanisms are *nonlinear*, and nonlinear additive noise makes the DAG identifiable:
regressing an effect on its cause can leave a residual independent of the cause, while the reverse
direction generally does not.

## Prior art / Background / Baselines

- **PC.** Tests conditional independencies and orients only v-structures, returning a CPDAG.
  *Gap:* it leaves the equivalence class — many edges undirected — and is direction-blind whenever
the distribution is summarized only by conditional independence information.
- **GES.** Greedy score-based search over DAGs with a penalized likelihood that is constant across an
  equivalence class. *Gap:* the same equivalence-class ceiling; the score cannot break the
  forward/backward Gaussian symmetry, and the search is combinatorial.
- **LiNGAM / ICA-LiNGAM.** Uses non-Gaussian noise in a linear SEM to make the mixing matrix
  identifiable, which yields directions. *Gap:* relies on a non-convex ICA optimization with local
  minima and permutation/scale ambiguities, and it is linear, so it mis-scores on nonlinear
  mechanisms.

## Fixed substrate / Code framework

The evaluation harness is frozen. `data_gen.py` simulates the ANM: it draws a random DAG (Erdős-Rényi
or scale-free, edges only from lower to higher node index), assigns each parent-child pair a random
nonlinear scalar function (GP via random Fourier features, a one-hidden-layer MLP, a low-degree
polynomial, or a steep sigmoid), draws additive noise (exponential / Laplace / uniform / Gaussian,
centered), and generates each `X_j` in topological order. `metrics.py` scores the *directed* edge
set: an edge is present when `|B[i,j]| > 0.01`; a true positive needs both skeleton and direction
right. `run_eval.py` calls `simulate_nonlinear_anm`, then `run_causal_discovery(X)`, then
`compute_metrics`. None of these is editable.

## Editable interface

Exactly one function is editable — `run_causal_discovery` in `bench/custom_algorithm.py`. The
contract is rigid:

- **Input:** `X` of shape `(n_samples, n_variables)` — observational data only. No graph type, node
count beyond `X.shape`, noise family, or function family is passed in.
- **Output:** an adjacency matrix `B` of shape `(n_variables, n_variables)` with the **causal-learn
convention** `B[i, j] != 0` meaning `j -> i` (row = child, column = parent). A 0/1 matrix is fine.
- **Reproducibility:** the seed is read from the `SEED` environment variable inside the function
(`int(os.environ.get("SEED", "42"))`), not passed as an argument.

The starting point is the scaffold default: **return no edges**.

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

Each rung replaces exactly this function body; available libraries are the harness's `causal-learn`,
plus `numpy`, `scipy`, and `scikit-learn`.

## Evaluation settings

Three scenarios, each over seeds {42, 123, 456}, chosen to vary every axis at once:

| Label          | Graph type          | Nodes | Samples | Noise       | Nonlinearity |
|----------------|---------------------|-------|---------|-------------|--------------|
| SF20-GP        | Scale-Free (BA)     | 20    | 2000    | Exponential | GP           |
| ER20-Gauss     | Erdős-Rényi         | 20    | 2000    | Gaussian    | Mixed        |
| ER12-LowSample | Erdős-Rényi         | 12    | 150     | Laplace     | Mixed        |

ER20-Gauss is the delicate case (ANM identifiability with Gaussian noise leans entirely on the
nonlinearity); ER12-LowSample is the small-`n` stress test. Metrics per scenario: **F1** (primary,
higher is better), **SHD** (lower is better), **precision**, and **recall** (higher is better), all
on the directed edge set.
