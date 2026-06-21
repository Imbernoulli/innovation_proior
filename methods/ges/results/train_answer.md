We are given a matrix of iid records over discrete variables and asked to recover the directed acyclic graph that generated them. The first thing to accept is that observational data cannot distinguish every DAG. Two DAGs that encode the same conditional-independence constraints fit every dataset equally well, so the honest target is the Markov equivalence class: the skeleton plus the v-structures, represented compactly as a completed partially directed acyclic graph (CPDAG). The challenge is to learn that class efficiently while carrying a large-sample correctness guarantee. Existing approaches each miss part of the requirement. Constraint-based methods such as PC build the graph from conditional-independence tests, but every decision is a hard accept/reject on a single hypothesis, and one wrong verdict cascades through the orientation rules. On discrete data the chi-squared tests become unreliable exactly where conditioning sets are large and cell counts thin. Score-based hill-climbing over individual DAGs avoids that cascade by optimizing one global objective, but it searches the larger, redundant space of DAGs, wasting moves on covered-edge reversals that change the DAG without changing the class, and it terminates at a local optimum with no guarantee of recovering the true structure even with infinite data.

The right combination is a local, greedy search that operates directly on equivalence classes and is justified by the score itself. The Bayesian Dirichlet equivalent uniform (BDeu) score for discrete data is decomposable into per-node family terms and, because it is derived under the likelihood-equivalence assumption, score-equivalent: any two DAGs in the same equivalence class receive the same score. These properties localize the global notion of consistency. In the large-sample limit, adding an edge raises the BDeu score if and only if the child is dependent on the new parent given its current parents, and removing an edge raises the score if and only if the corresponding independence holds. That local consistency lets each greedy step make the correct edge-level decision in the limit. To turn those local decisions into a global guarantee, the search must move through equivalence classes in the right way: additions climb to a class that contains the true distribution, and deletions climb back down to the sparsest such class, which is exactly the true equivalence class. Meek's conjecture, proved constructively, shows that any independence-map relation between DAGs can be realized by a sequence of covered-edge reversals and single edge additions, which is the structural fact that makes the backward phase terminate at the perfect map.

The method is Greedy Equivalence Search (GES). It maintains a completed PDAG, the canonical representative of an equivalence class, and proceeds in two phases. In the forward phase, it considers every class reachable by adding a single edge to some member DAG of the current class, scores each candidate by the local change in the BDeu score, and applies the best improving move until no insertion helps. The resulting local maximum is guaranteed to contain the true distribution under the composition axiom and local consistency. In the backward phase, it considers every class reachable by deleting a single edge, again scores locally, and applies the best improving deletion until none remains. The deletion phase is guaranteed to descend to the true equivalence class because the score's Occam preference for fewer parameters prevents the search from leaving the set of classes containing the true distribution, and Meek's characterization guarantees that any sparser true container is reachable by a valid delete-neighbor move.

The local operators make this efficient without enumerating class members. Insert(X, Y, T) adds the edge X -> Y for non-adjacent X and Y and orients a chosen subset T of Y's undirected neighbors as parents of Y, creating any new v-structures that the edge requires. It is valid when NA_{Y,X} ∪ T is a clique and every semi-directed path from Y to X meets that clique, which together ensure a consistent extension exists and no cycle is created. Delete(X, Y, H) removes the edge between adjacent X and Y and orients a chosen subset H of Y's neighbors as children of both endpoints; it is valid when NA_{Y,X} \ H is a clique. Because the score is decomposable and score-equivalent, each candidate is scored by evaluating only the family score of Y on a witness DAG implied by the current completed PDAG. After an accepted move the PDAG is reconverted to completed form by extracting a consistent extension via the Dor–Tarsi algorithm and relabeling compelled versus reversible edges; that conversion runs only on accepted moves, never for every scored candidate.

The family score is the uniform-Dirichlet multinomial marginal likelihood. With r_i states of node X_i, q_i parent configurations, counts N_ijk and N_ij, and equivalent sample size N', the contribution is the log-Gamma form of BDeu plus a decomposable structure-prior term. The default equivalent sample size is 1 and the structure prior is 1, with cardinalities inferred directly from the data. Larger score means better penalized fit, and GES compares score differences.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GESUtils import (
    precompute_graph_info, Combinatorial, find_subset_include,
    check_clique_fast, insert_vc2_fast,
    insert_changed_score_fast, delete_changed_score_fast,
    insert, delete, score_g,
)
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BDeu


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """GES with the BDeu score. Two-phase greedy search over equivalence classes;
    returns the estimated CPDAG as a GeneralGraph."""
    N = X.shape[1]
    maxP = N

    score_func = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    parameters = None

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(N)]
    G = GeneralGraph(nodes)
    score = score_g(X, G, score_func, parameters)
    G = dag2cpdag(pdag2dag(G))
    cache = {}

    while True:
        best_gain, best = -np.inf, None
        nbrs, adj, pa, semi = precompute_graph_info(G, N)
        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == 0
                    and G.graph[j, i] == 0
                    and i != j
                    and len(pa[j]) <= maxP
                ):
                    NA = nbrs[j] & adj[i]
                    subsets = Combinatorial(sorted(nbrs[j] - adj[i]))
                    flag = np.zeros(len(subsets))
                    for k in range(len(subsets)):
                        if flag[k] >= 2:
                            continue
                        T = set(subsets[k])
                        if check_clique_fast(G, NA | T):
                            if flag[k] == 0:
                                valid_path = insert_vc2_fast(j, i, NA | T, semi)
                            else:
                                valid_path = 1
                            if valid_path:
                                flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 1
                                gain, desc, cache = insert_changed_score_fast(
                                    X, i, j, subsets[k], NA, pa[j], cache, score_func, parameters)
                                if gain > best_gain:
                                    best_gain, best = gain, desc
                        else:
                            flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
        if best is None or best_gain <= 0:
            break
        G = insert(G, best[0], best[1], best[2])
        G = dag2cpdag(pdag2dag(G))
        score += best_gain

    while True:
        best_gain, best = -np.inf, None
        nbrs, adj, pa, semi = precompute_graph_info(G, N)
        for i in range(N):
            for j in range(N):
                if (j in nbrs[i]) or (i in pa[j]):
                    NA = nbrs[j] & adj[i]
                    subsets = Combinatorial(sorted(NA))
                    ok = np.ones(len(subsets))
                    for k in range(len(subsets)):
                        H = set(subsets[k])
                        if ok[k] == 1:
                            if check_clique_fast(G, NA - H):
                                ok[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
                            else:
                                continue
                        gain, desc, cache = delete_changed_score_fast(
                            X, i, j, subsets[k], NA, pa[j], cache, score_func, parameters)
                        if gain > best_gain:
                            best_gain, best = gain, desc
        if best is None or best_gain <= 0:
            break
        G = delete(G, best[0], best[1], best[2])
        G = dag2cpdag(pdag2dag(G))
        score += best_gain

    return G
```
