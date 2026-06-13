# The PC algorithm, distilled

The PC (Peter–Clark) algorithm recovers, from purely observational data, the Markov
equivalence class of the causal DAG behind a set of variables, represented as a CPDAG
(completed partially directed acyclic graph). It is **constraint-based**: it asks the data
only conditional-independence (CI) questions and assembles the graph from the answers. Its
defining moves are (1) thinning a complete graph into the skeleton with CI tests of
increasing conditioning-set size, conditioning *only on subsets of current neighbors*; (2)
orienting unshielded colliders from the recorded separating sets; (3) maximally propagating
the remaining orientations with Meek's rules.

## Problem it solves

Given integer-coded discrete observational data (no interventions, no known variable
ordering, causal sufficiency assumed — no unmeasured common causes), recover the identifiable
causal structure. Observational independences identify only the **Markov equivalence class**,
so the output is a CPDAG: an edge wherever two variables are directly connected, an arrowhead
wherever every DAG in the class agrees on the direction, an undirected edge where they
disagree.

## Foundations

- **Markov condition**: `P(V) = ∏_{W∈V} P(W | Pa(W))`; equivalently every variable is
  independent of its non-descendants given its parents. Every independence the graph asserts is
  real.
- **Faithfulness**: the converse — *all and only* the distribution's independences are those
  the graph entails. Together they give a two-way bridge: d-separation in `G` ⟺ conditional
  independence in `P`.
- **d-separation**: `X ⟂_d Y | W` iff every path between them is blocked — a non-collider in
  `W` blocks; a collider blocks unless it (or a descendant) is in `W`.
- **Identifiability (Verma–Pearl)**: two DAGs entail the same independences iff they have the
  same skeleton and the same unshielded colliders. So the CPDAG is the right target.

## Key ideas and why

1. **Thin, don't build.** A non-adjacent pair has a separating set (faithfulness:
   non-adjacent ⟹ ∃ S with `X ⟂ Y | S`). Start complete and delete edges; an absent edge is
   evidenced by a found separator.
2. **Condition only on adjacency subsets.** If `X`, `Y` are non-adjacent with `Y` a
   non-descendant of `X`, then `X ⟂ Y | Pa(X)`, and `Pa(X) ⊆ Adj(X)`. A removal-only search
   keeps the working graph a supergraph of the truth, so the needed separator is always among
   subsets of the *current* adjacencies. With the usual local-degree parameter `k`, the standard
   loose count is `2·C(p,2)·Σ_{i=0}^{k-1} C(p-1,i)`, summarized by the upper-bound scale
   `p²(p-1)^{k-1}/(k-1)!` — polynomial in `p` for sparse graphs, vs. the exponential
   `2^{p-2}`-per-pair exhaustive search.
3. **Increase conditioning order 0, 1, 2, …** Low-order discrete CI tests are statistically
   reliable (well-populated contingency tables); high-order tests are not (cells grow as
   `∏ Cat`, so most are empty). Removing easy edges first shrinks neighborhoods and keeps
   later tests low-order.
4. **Colliders from sepsets, for free.** For an unshielded triple `X — Y — Z` (`X`, `Z`
   non-adjacent), orient `X → Y ← Z` **iff `Y` is absent from the stored separating information
   for `X`, `Z`**. If `Y` is a non-collider, the length-two path remains active unless a
   separator includes `Y`; if `Y` is a collider, including `Y` opens that path and no other
   interior node on the length-two path can close it. Causal-learn's `uc_sepset` implements
   this as `all(y not in S for S in cg.sepset[x, z])`.
5. **Meek propagation.** Close under rules that avoid creating an unrecorded unshielded
   collider or a cycle, yielding the CPDAG (sound and complete):
   - **R1**: `A → B`, `B — C`, `A`/`C` non-adjacent ⟹ `B → C` (else a new collider at `B`).
   - **R2**: `A → B → C`, `A — C` ⟹ `A → C` (else cycle `A→B→C→A`).
   - **R3**: `A — B`, `A — C`, `A — D`, `B → D`, `C → D`, `B`/`C` non-adjacent ⟹ `A → D`.
     Orienting `D → A` would force `B → A` and `C → A` by R2, creating a new unshielded
     collider `B → A ← C`.
   - (R4 belongs to the background-knowledge extension; causal-learn's default `meek()` path
     closes under R1-R3.)
6. **Order-independent skeleton (PC-stable).** With sample data, deleting an edge mid-level
   changes other pairs' adjacency sets, making the output depend on variable order. Fix:
   snapshot each node's adjacency set `a(X)` at the start of each level, draw all conditioning
   subsets from the frozen `a(X)`, record removals during the level, and let the next level's
   snapshot absorb those removals. Oracle output unchanged; the sample skeleton is
   order-independent, though collider and later orientation choices can still depend on conflict
   policy.
7. **Prioritize existing collider orientations.** Under sampling error two collider decisions can
   conflict over a shared edge. With `uc_sepset(..., priority=2)`, orient `X → Y ← Z` only if
   neither `Y → X` nor `Y → Z` is already directed, preventing overwrite cascades and spurious
   bidirected edges.

## Discrete CI test (chi-squared)

For `X ⟂ Y | S`, stratify by each joint configuration `k` of `S`. Expected cell count under
conditional independence:

```
E(x_{ijk}) = x_{i+k} · x_{+jk} / x_{++k}        (per stratum k; for S = ∅, E(x_{ij}) = x_{i+} x_{+j} / N)
```

Pearson statistic and degrees of freedom:

```
X² = Σ_k Σ_{i,j} (O_{ijk} − E_{ijk})² / E_{ijk}
df = Σ_k (Cat(X) − 1 − z_rows_k) · (Cat(Y) − 1 − z_cols_k)     # z = #all-zero rows/cols in stratum k
p  = chi2.sf(X², df)      (df = 0 ⟹ p = 1).   Delete edge X — Y when p > α.
```

`α` is a tuning parameter (the search runs many tests, so it is not a true significance
level); smaller `α` ⟹ sparser graph. Default `α = 0.05`. The likelihood-ratio `G² = 2 Σ O
ln(O/E)` is an asymptotically equivalent alternative.

## Algorithm

```
1. C ← complete undirected graph on V
2. Skeleton (PC-stable): for n = 0, 1, 2, …
     snapshot a(X) = Adj(C, X) for all X
     for each adjacent ordered pair (X, Y) with |a(X)\{Y}| ≥ n:
       for each S ⊆ a(X)\{Y}, |S| = n:
         if X ⟂ Y | S (p > α):  mark X—Y for deletion; record S among Sepset(X,Y), Sepset(Y,X)
     remove all edges marked at this level before taking the next snapshot
   until every adjacent ordered pair has |a(X)\{Y}| < n for the next level
3. Colliders: for each unshielded triple X — Y — Z (X,Z non-adjacent):
     if Y is absent from every stored sepset entry for X,Z and neither Y→X nor Y→Z already oriented: orient X → Y ← Z
4. Propagation: repeatedly apply Meek R1, R2, R3 until no edge orients ⟹ CPDAG
```

## Working code

Built on the `causallearn` discrete CI oracle and graph primitives; returns the CPDAG as a
`GeneralGraph`. The three phases are the three calls.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from causallearn.utils.cit import CIT


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data.
    Output: estimated CPDAG as a causallearn GeneralGraph.
    """
    alpha = 0.05
    indep_test = CIT(X, "chisq")          # discrete chi-squared CI test

    # 1. Skeleton: thin the complete graph with chi-squared CI tests of increasing
    #    conditioning-set size; stable=True freezes adjacency sets per level so the
    #    skeleton is order-independent.
    cg_1 = SkeletonDiscovery.skeleton_discovery(
        X, alpha, indep_test, stable=True,
        background_knowledge=None, verbose=False,
        show_progress=False, node_names=None,
    )

    # 2. Orient unshielded colliders X -> Y <- Z only when Y is absent
    #    from every stored sepset entry for X, Z. priority=2 preserves an
    #    existing opposite orientation instead of overwriting it.
    cg_2 = UCSepset.uc_sepset(cg_1, 2, background_knowledge=None)

    # 3. Maximally orient remaining edges with Meek rules (R1-R3) -> CPDAG.
    cg = Meek.meek(cg_2, background_knowledge=None)

    return cg.G
```

## Properties and limits

- **Correctness**: given faithful (oracle) CI information, the output is the CPDAG of the true
  DAG (PC and PC-stable agree in the oracle).
- **Complexity**: bounded by the maximal degree `k`; feasible on sparse graphs with up to
  hundreds of variables; exponential in `k`.
- **Fragility**: an early wrong edge deletion can ramify (a shrunk neighborhood leaves a
  different false edge un-testable); the collider phase is the least stable, since one
  mis-decided collider propagates through the Meek closure. PC-stable and conservative collider
  orientation contain the worst of this, but on small samples or near-unfaithful distributions
  a score-based repair pass can still help.
