**Problem.** GRaSP rescued the medium-density Alarm (SHD 3.3) but stayed poor and unstable on the
densest networks (Hailfinder 55, Win95pts 51.5, one seed unfinished), with adjacency recall sagging —
its heuristic ordering search wanders a rugged score landscape on big graphs. Try the opposite
instrument: test the skeleton directly instead of scoring whole structures.

**Key idea (PC — constraint-based).** Recover the equivalence class with conditional-independence tests,
not a score. **Skeleton:** start complete, thin by chi-squared CI tests of increasing conditioning-set
size, conditioning only on current neighbors (a non-adjacent pair is separated by the parents of an
endpoint, and parents are neighbors), low-order first so tests stay cheap and statistically sane; freeze
adjacency sets within each round (`stable`) for order-independence. **Colliders:** for each unshielded
triple `X — Y — Z`, orient `X → Y ← Z` iff `Y` is absent from the recorded separator of `X, Z` — a
lookup, applied conservatively so it does not overwrite an existing orientation. **Propagate:** Meek
rules — avoid new unrecorded colliders and cycles — until nothing more orients. The result is the CPDAG.

**Why try it over GRaSP.** It never scores a structure, so it sidesteps the rugged score landscape;
low-order CI thinning can prune a large skeleton cheaply where GRaSP's ordering search stalls. The cost
is a different fragility: each decision is a hard yes/no on one CI test, so early errors cascade, and the
discrete chi-squared gets unreliable when conditioning-set cells are sparse — so it should trade higher
adjacency precision for lower recall on the densest, near-unfaithful networks.

**Hyperparameters.** CI test `"chisq"`; `alpha = 0.05` (a sparsity knob, not a true significance level);
stable skeleton (`stable=True`); conservative collider orientation (`uc_sepset` priority `2`); Meek
propagation. No background knowledge.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.utils.PCUtils import SkeletonDiscovery, Meek, UCSepset
    from causallearn.utils.cit import CIT

    alpha = 0.05
    indep_test = CIT(X, "chisq")

    # Step 1: skeleton discovery via chi-squared CI tests (stable PC)
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
