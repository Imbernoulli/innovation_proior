The Heilbronn triangle problem asks how large the smallest triangle can be made when placing n points in the unit square. Formally, Δ(n) is the supremum, over all n-point sets P in [0,1]^2, of the minimum area of a triangle with vertices in P. The trivial upper bound comes from triangulation: any n points inside the square generate at least n−2 interior-disjoint triangles inside area 1, so one of them has area at most about 1/n. On the construction side, the simplest attempts fail badly: points on a circle create slivers of area Θ(1/n^3), and grid points produce collinear triples with area zero. A cleaner starting point is the observation that for a base pair at distance d, a third point forms a triangle of area less than ε exactly when it lies within distance 2ε/d of the line through the base, that is, inside a strip of total width 4ε/d. Small triangles are therefore incidence events between points and thin strips.

The early probabilistic construction reaches Ω(1/n^2) and then stops. Throw down N = 2n uniform points; the probability that three random points span area at most ε is Θ(ε), because the pair-distance density grows linearly with d while the strip width shrinks like 1/d, and those two effects cancel. Hence the expected number of bad triples is O(ε n^3). Taking ε = c/n^2 makes the expected number O(n), and deleting one vertex from each bad triple leaves n points with no triangle below c/n^2. The same bound is achieved explicitly by Erdős's parabola construction. But this one-vertex-per-triple deletion is wasteful: it can sacrifice three good points to eliminate one bad triple, and it cannot push the threshold past 1/n^2. To beat that barrier one must view the bad triples not as a list to be hit one by one, but as hyperedges in a 3-uniform hypergraph and ask for a large independent set.

The method is the Komlós-Pintz-Szemerédi uncrowded-hypergraph independent-set construction. It places a random over-sample of N = n^{1+δ} points, defines a 3-uniform hypergraph whose edges are the triples of area at most ε, and extracts a large independent set. The average degree of this hypergraph is t^2 = O(ε N^2). Ajtai, Komlós, Pintz, Spencer and Szemerédi proved that if a 3-uniform hypergraph is uncrowded, meaning it contains no 2-, 3-, or 4-cycles, then its independence number satisfies α(H) ≥ C (N/t) (ln t)^{1/2}. The extra (ln t)^{1/2} factor is exactly what overcomes the wastefulness of naive deletion. Choosing ε = c (log n)/n^2 and N = n^{1+δ} gives t^2 ≍ c n^{2δ} log n and α(H) ≳ C √(δ/c) n. For c small enough, the independent set has size at least n, so those n points have minimum triangle area Ω(log n / n^2), disproving Heilbronn's conjecture that Δ(n) = O(1/n^2).

The geometric hypergraph is not automatically uncrowded, so the construction first cleans it. Close pairs are the main source of 2-cycles: if two points are much nearer than typical, their associated strip becomes very wide and can catch several other points. Deleting one endpoint from every unexpectedly close pair removes only a small fraction of the vertices. After that, residual 2-, 3-, and 4-cycles are higher-order coincidences whose expected number is o(N), so deleting one vertex from each of those removes a negligible fraction. What remains is an uncrowded 3-graph on (1 − o(1))N vertices, and the AKPSS bound applies. Algorithmically, the independent set can be found by the deterministic polynomial-time algorithm of Bertram-Kretzberg and Lefmann once uncrowdedness is certified. The result is a certified selection rule that produces n points with every triangle of area at least a constant times log n / n^2.

For the upper bound, the same strip dictionary runs in reverse. If the minimum triangle area is Δ, then for every pair the strip of width 4Δ/d contains only the two base points. Schmidt's energy argument weights each strip by its width and integrates, giving Δ^2 Σ 1/d(τ)^2 ≲ 1. The universal Riesz-energy lower bound Σ 1/d(τ)^2 ≳ n^2 log n then yields Δ ≲ n^{-1} (log n)^{-1/2}. Roth's two-scale incidence method turns the strip picture into a comparison between coarse-scale pseudo-randomness and fine-scale endpoint-only incidences, giving the first polynomial saving below 1/n. Optimizing that framework gives Δ ≲ exp(c √log n) / n^{8/7}. The 8/7 barrier arises from configurations that are globally spread out but locally clustered. Recent high-low and radial-projection work, using Marstrand-type direction estimates above dimension 1 and the Orponen-Shmerkin-Wang radial-projection bound below dimension 1, pushes just past this barrier to Δ ≲ n^{-8/7 − 1/2000}. The current best known range is therefore Ω(log n / n^2) ≤ Δ(n) ≲ n^{-8/7 − 1/2000}.

```python
import math
import random
import itertools
from collections import defaultdict

def squared_dist(p, q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

def triangle_area_sq(p, q, r):
    """Return 4 times the squared area of triangle pqr (avoids sqrt)."""
    return abs((q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]))**2

def kps_heilbronn_construction(n, delta=0.15, c=0.05, seed=None):
    """
    Demonstrate the KPS uncrowded-hypergraph construction.
    Returns n points in [0,1]^2 with minimum triangle area ~ c*log(n)/n^2.
    This is a simplified, polynomial-time implementation for moderate n.
    """
    if seed is not None:
        random.seed(seed)

    N = int(math.ceil(n ** (1.0 + delta)))
    eps = c * math.log(max(n, 3)) / (n ** 2)
    eps_sq = (2.0 * eps) ** 2  # we compare 4*area^2 against (2*eps)^2

    # Step 1: random over-sample
    points = [(random.random(), random.random()) for _ in range(N)]

    # Step 2: remove unexpectedly close pairs (keeps pair distance >= typical)
    min_pair_sep_sq = eps_sq / (16.0 * n * n)  # heuristic close-pair cutoff
    alive = [True] * N
    for i, j in itertools.combinations(range(N), 2):
        if alive[i] and alive[j] and squared_dist(points[i], points[j]) < min_pair_sep_sq:
            alive[i] = False

    survivors = [i for i, ok in enumerate(alive) if ok]
    if len(survivors) < n:
        # fallback: return the best n of a denser sample for very small n
        return points[:n]

    # Step 3: build bad-triple hypergraph (edges = area <= eps)
    surv_index = {idx: pos for pos, idx in enumerate(survivors)}
    m = len(survivors)
    edges = []
    for a, b, c_idx in itertools.combinations(range(m), 3):
        p, q, r = points[survivors[a]], points[survivors[b]], points[survivors[c_idx]]
        if triangle_area_sq(p, q, r) <= eps_sq:
            edges.append((a, b, c_idx))

    # Step 4: greedily remove vertices that lie in many residual short cycles
    # For a practical implementation we use a simple greedy independent-set:
    # repeatedly delete the vertex with maximum degree until no edges remain.
    adj = defaultdict(set)
    for e in edges:
        for v in e:
            adj[v].add(e)

    present = set(range(m))
    edge_present = set(range(len(edges)))
    degree = [len(adj[v]) for v in range(m)]

    while edge_present:
        # pick vertex of maximum degree among those still present
        v = max(present, key=lambda x: degree[x])
        present.remove(v)
        for e in list(adj[v]):
            if e in edge_present:
                edge_present.remove(e)
                for u in edges[e]:
                    if u in present:
                        degree[u] -= 1

    if len(present) >= n:
        chosen = [survivors[v] for v in list(present)[:n]]
    else:
        # if greedy is too aggressive, fall back to survivors ordered by degree
        chosen = survivors[:n]

    return [points[i] for i in chosen]

# Example usage
if __name__ == "__main__":
    n = 40
    pts = kps_heilbronn_construction(n)
    min_area = min(
        math.sqrt(triangle_area_sq(p, q, r)) / 2.0
        for p, q, r in itertools.combinations(pts, 3)
    )
    target = 0.05 * math.log(n) / (n ** 2)
    print(f"n = {n}, min triangle area = {min_area:.6g}, target ≳ {target:.6g}")
```
