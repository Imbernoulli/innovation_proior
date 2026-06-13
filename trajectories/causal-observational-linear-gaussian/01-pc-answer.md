**Problem.** Recover the CPDAG of a linear Gaussian SEM from observational data, asking the data
only conditional-independence questions. Observational independences identify only the Markov
equivalence class, so the output is a CPDAG: an edge wherever two variables are directly connected,
an arrowhead wherever every DAG in the class agrees, an undirected edge where they disagree.

**Key idea.** The constraint-based PC pipeline in three phases. (1) *Skeleton*: start from the
complete graph and thin it with CI tests of increasing conditioning-set size, conditioning only on
subsets of current neighbors (a removal-only search keeps the working graph a supergraph of the
truth, so the needed separator — `Pa(X)` for a non-adjacent non-descendant pair — is always among
current adjacencies). (2) *Colliders*: for each unshielded triple `X — Y — Z`, orient `X → Y ← Z`
iff `Y` is absent from the stored separating set for `X, Z`. (3) *Propagation*: close under Meek's
rules R1-R3 to the maximally oriented CPDAG.

**Why this fill.** This task is *linear Gaussian and continuous*, so the CI test is **Fisher's z**
on partial correlations (exact for joint Gaussians: `X ⟂ Y | S` iff `ρ_{XY·S} = 0`), instantiated
as `CIT(X, "fisherz")` — not the discrete chi-squared of the textbook PC. `stable=True` freezes
adjacency sets per conditioning level so the skeleton is order-independent; `uc_sepset(..., 2)`
orients colliders conservatively (priority 2 preserves an existing opposite orientation rather than
overwriting it); `Meek.meek` runs the R1-R3 closure. The three phases are the three library calls.

**Hyperparameters.** `alpha = 0.05` (CI acceptance threshold — a sparsity knob, not a true
significance level, since the search runs thousands of tests); `fisherz` CI test; `stable=True`;
collider priority `2`; no background knowledge.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.utils.PCUtils import SkeletonDiscovery, Meek, UCSepset
    from causallearn.utils.cit import CIT

    alpha = 0.05
    indep_test = CIT(X, "fisherz")

    # Step 1: skeleton discovery via conditional independence tests (stable PC)
    cg_1 = SkeletonDiscovery.skeleton_discovery(
        X, alpha, indep_test, stable=True,
        background_knowledge=None, verbose=False,
        show_progress=False, node_names=None,
    )

    # Step 2: orient unshielded colliders using UC-sepset rule (priority=2)
    cg_2 = UCSepset.uc_sepset(cg_1, 2, background_knowledge=None)

    # Step 3: complete orientation with Meek rules
    cg = Meek.meek(cg_2, background_knowledge=None)

    return cg.G
```
