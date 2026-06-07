# The Heilbronn triangle problem: principled bounds

**Problem.** Place n points in the unit square to maximize the area of the *smallest* triangle they span:

    Δ(n) = sup_{|P|=n, P⊂[0,1]²}  min_{distinct p,q,r∈P}  area(p,q,r).

Heilbronn conjectured Δ(n) = O(1/n²). The logarithmic construction refutes that conjectured order, while the incidence/projection argument forces a universal upper bound just beyond the 8/7 barrier.

**Strip dictionary.** For a pair x,y at distance d, area(x,y,z) < ε iff z lies within distance 2ε/d of the line through x,y. If T_xy(w) denotes the strip of total width w, this is T_xy(4ε/d). Constructions keep these strips empty of third points; upper bounds exploit the fact that a large Δ would force many wide strips containing only their base endpoints.

## Lower Bound

**Δ(n) = Ω(1/n²) by basic alteration.** For three uniform points, P(area ≤ ε) = Θ(ε); the O(ε) side follows by conditioning on the base length x, where the strip probability is O(ε/x) and the pair-distance density is O(x). Drop N=2n uniform points. Then E[#bad triples] = O(εN³). Taking ε=c/n² with c small gives expected bad-triple count below n. Delete one vertex from each bad triple, leaving n points with no triangle below c/n² with positive probability. Erdős's parabola construction gives the same order explicitly.

**Δ(n) = Ω(log n / n²) by the Komlós-Pintz-Szemerédi uncrowded-independent-set construction.** Let H be the 3-uniform hypergraph whose vertices are a random N-point set and whose edges are triples of area at most ε. The edge count is O(εN³), so the average degree is t² = O(εN²). Ajtai-Komlós-Pintz-Spencer-Szemerédi prove that an uncrowded 3-graph has

    α(H) ≥ C (N/t)(ln t)^{1/2}.

Take N=n^{1+δ} and ε=c(log n)/n². Then t²≈c n^{2δ}log n, so t≈√c n^δ(log n)^{1/2}, and

    (N/t)(ln t)^{1/2} ≈ √(δ/c) n.

Choosing c small gives an independent set of at least n vertices. The random-geometric hypergraph is made uncrowded by deleting one endpoint from each exceptionally close pair and one vertex from each residual 2-, 3-, and 4-cycle. This gives Δ(n)=Ω(log n/n²).

## Upper Bound

If Δ is the minimum area, each pair τ determines a third-point-free strip T_τ(4Δ/d(τ)).

- **Schmidt energy:** Weight T_τ by w_τ=4Δ/d(τ). Schmidt's estimate gives Σ_τ w_τ1_{T_τ}(x)≲1 for every x, hence the weighted incidence integral is O(1). Expanding by strips gives Δ²Σ_τ d(τ)^{-2}≲1. Since every n-point set has Riesz energy Σ_τ d(τ)^{-2}≳n²log n, Δ≲n^{-1}(log n)^{-1/2}.
- **Roth two-scale incidence:** Count smoothed incidences I(w;P,L) between points and lines from short pairs. A coarse scale has I(w_i)≳w_i|P||L|, while the fine scale has only endpoint incidences unless a small triangle already exists. Quasi-orthogonality of strip differences (Selberg/Bessel) keeps normalized incidences stable between scales. Roth obtained μ=1−√(4/5)≈0.1056, then μ=(9−√65)/8≈0.1172 in Δ≲n^{-1−μ}.
- **KPS upper optimization:** Optimizing Roth's framework gives Δ≤exp(c√log n)/n^{8/7}.
- **Incidence-to-area bridge:** If lines L are spanned by pairs of length at most u and B(w;P,L)≥κ with enough actual incidences, then endpoint-only strips force

    Δ/|log Δ| ≲ max{u,u^{2/3}w^{1/3}} |P|^{-1/3}|L|^{-1/3}κ^{-2/3} + u|P|^{-1}κ^{-1}.

  This is the quantitative step that turns a lower bound for normalized incidences into a smaller forced triangle.
- **High-low/projection refinement:** With B(w)=I(w)/(w|P||L|),

    |B(w)-B(w/10)| ≲ ((M_P(w×w)M_L(w×1))/(|P||L|) · w^{-3})^{1/2}.

  The error exposes the 8/7 obstruction as global spread with local point clustering. Marstrand-type direction estimates handle clusters of dimension above 1; the Orponen-Shmerkin-Wang radial-projection estimate handles below-dimension-1 clusters when no thin tube captures a fixed positive fraction, and the complementary thin-tube case is treated as a rectangle. This gives Δ≤n^{-8/7−1/2000}; for homogeneous sets, Δ≤n^{-7/6+ε} for every fixed ε>0.

Bounds reached here:

    Ω(log n / n²) ≤ Δ(n) ≤ n^{-8/7−1/2000}.

## Code

This executable block certifies finite trials by exact independent-set search. In the asymptotic construction, the same `independent_subset` slot is filled by the uncrowded-hypergraph selector.

```python
import numpy as np
from itertools import combinations, permutations

def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1])
                     - (q[1] - p[1]) * (r[0] - p[0]))

def small_triples(pts, eps):
    return [(i, j, k) for i, j, k in combinations(range(len(pts)), 3)
            if tri_area(pts[i], pts[j], pts[k]) <= eps]

def construct_by_deletion(n, c=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    N = 2 * n
    pts = rng.random((N, 2))
    eps = c / n**2
    alive = np.ones(N, dtype=bool)
    for i, j, k in small_triples(pts, eps):
        if alive[i] and alive[j] and alive[k]:
            alive[k] = False
    keep = np.flatnonzero(alive)
    if len(keep) < n:
        raise RuntimeError("this random trial did not certify n surviving points")
    return pts[keep[:n]], eps

def _forms_cycle(edge_sets):
    def has_distinct_cycle_witness(ordered):
        choices = []
        for a in range(len(ordered)):
            inter = ordered[a] & ordered[(a + 1) % len(ordered)]
            if not inter:
                return False
            choices.append(tuple(inter))

        def search(pos, used):
            if pos == len(ordered):
                return True
            for v in choices[pos]:
                if v not in used and search(pos + 1, used | {v}):
                    return True
            return False

        return search(0, set())

    return any(has_distinct_cycle_witness(order)
               for order in permutations(edge_sets))

def remove_crowding_obstructions(num_vertices, triples):
    live = set(range(num_vertices))
    edges = [tuple(e) for e in triples]
    while True:
        edge_sets = [set(e) for e in edges]
        drop = None
        for length in (2, 3, 4):
            for idxs in combinations(range(len(edge_sets)), length):
                if _forms_cycle([edge_sets[i] for i in idxs]):
                    drop = next(iter(edge_sets[idxs[0]]))
                    break
            if drop is not None:
                break
        if drop is None:
            return sorted(live), edges
        live.discard(drop)
        edges = [e for e in edges if drop not in e]

def independent_subset(vertices, triples, target):
    vertices = list(vertices)
    edges = [frozenset(e) for e in triples]
    incident = {v: [e for e in edges if v in e] for v in vertices}
    chosen, best = set(), []

    def dfs(pos):
        nonlocal best
        if len(chosen) > len(best):
            best = list(chosen)
        if len(chosen) >= target:
            return True
        if pos == len(vertices):
            return False
        if len(chosen) + len(vertices) - pos <= len(best):
            return False

        v = vertices[pos]
        chosen.add(v)
        if not any(e <= chosen for e in incident.get(v, ())):
            if dfs(pos + 1):
                return True
        chosen.remove(v)
        return dfs(pos + 1)

    dfs(0)
    return best[:target]

def construct(n, delta=0.5, c=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    N = int(np.ceil(n ** (1 + delta)))
    pts = rng.random((N, 2))
    eps = c * np.log(max(n, 3)) / n**2
    triples = small_triples(pts, eps)
    vertices, triples = remove_crowding_obstructions(N, triples)
    keep = independent_subset(vertices, triples, n)
    if len(keep) < n:
        raise RuntimeError("this finite trial did not certify n independent vertices")
    return pts[keep], eps
```
