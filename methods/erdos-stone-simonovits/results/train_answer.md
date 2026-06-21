The extremal number ex(n, H) asks for the most edges an n-vertex graph can have without containing a fixed forbidden subgraph H. At first this seems to depend on the fine structure of H: its cycles, degrees, clique number, and the way its vertices are connected. Turan's theorem gives an exact answer when H itself is a clique, but for a general forbidden graph it is not obvious what the leading term should be. One might hope the largest clique in H controls the answer, but that fails because an odd cycle has clique number 2 and chromatic number 3, and forbidding it produces the same first-order density as forbidding a triangle. Local counting arguments also fail to pin down the n^2 coefficient. What matters is not the internal shape of H but the smallest number of colors needed to paint it.

The Erdős–Stone–Simonovits theorem gives the universal answer. For a fixed graph H with chromatic number chi(H) = r + 1 at least 2, the extremal number satisfies

ex(n, H) = (1 - 1/r + o(1)) * n^2 / 2,

which can also be written as (1 - 1/(chi(H) - 1) + o(1)) * n^2 / 2. When H is bipartite the theorem says only ex(n, H) = o(n^2), leaving the precise subquadratic order to graph-specific methods.

The theorem works in two matching directions. The lower bound comes from the Turán graph T_r(n), the balanced complete r-partite graph on n vertices. Because T_r(n) is r-colorable, it cannot contain any graph that needs r + 1 colors, and it has (1 - 1/r + o(1)) * n^2 / 2 edges. So every forbidden graph with chromatic number r + 1 has at least that extremal density. The upper bound is the deeper half. The Erdős–Stone blow-up theorem says that any graph whose density exceeds the Turán density by a fixed positive amount must contain a complete (r + 1)-partite blow-up K_{r+1}(t) for every fixed t once n is large enough. Every fixed graph H with chi(H) = r + 1 embeds into such a blow-up: color H properly with r + 1 colors, place each color class in its own part of the blow-up, and fill in all missing cross-edges. Therefore once the density barrier is crossed, H itself is forced. The detailed shape of H only influences the bounded blow-up size needed and possible lower-order terms; the n^2 coefficient is governed entirely by chromatic number.

```python
import itertools
import math
from typing import Iterable, List, Set, Tuple


def chromatic_number(edges: Set[Tuple[int, int]], n: int) -> int:
    """Brute-force chromatic number of a graph with vertices 0..n-1."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Try k colors from 1 upward.
    for k in range(1, n + 1):
        colors = [-1] * n

        def backtrack(v: int) -> bool:
            if v == n:
                return True
            used = {colors[u] for u in adj[v] if colors[u] != -1}
            for c in range(k):
                if c not in used:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False

        if backtrack(0):
            return k
    return n


def turan_edges(n: int, r: int) -> int:
    """Number of edges in the balanced complete r-partite graph T_r(n)."""
    if r <= 0:
        return 0
    # Split n vertices into r parts as evenly as possible.
    q, rem = divmod(n, r)
    # rem parts of size q+1 and r-rem parts of size q.
    sizes = [q + 1] * rem + [q] * (r - rem)
    total = n * (n - 1) // 2
    same_part = sum(s * (s - 1) // 2 for s in sizes)
    return total - same_part


def ess_extremal_estimate(n: int, chi_h: int) -> float:
    """Asymptotic estimate of ex(n, H) from the Erdős-Stone-Simonovits theorem.

    chi_h is the chromatic number of H. Returns the leading n^2 term only.
    """
    if chi_h <= 2:
        return 0.0  # o(n^2); no universal leading coefficient.
    r = chi_h - 1
    return (1.0 - 1.0 / r) * n * n / 2.0


def embeds_in_blowup(edges: Set[Tuple[int, int]], n: int,
                     parts: int, part_size: int) -> bool:
    """Check whether H (given by edges on vertices 0..n-1) embeds in K_parts(part_size).

    This is a small brute-force test useful for sanity-checking the theorem on
    tiny examples.
    """
    # Enumerate all assignments of vertices to distinct slots in the blow-up.
    vertices = list(range(n))
    slots = [(p, i) for p in range(parts) for i in range(part_size)]
    # We need an injective map f : V(H) -> slots with no two vertices in the same part
    # adjacent in H (cross-edges are complete in the blow-up).
    if len(vertices) > len(slots):
        return False

    for placement in itertools.permutations(slots, len(vertices)):
        ok = True
        for u, v in edges:
            pu, _ = placement[u]
            pv, _ = placement[v]
            if pu == pv:
                ok = False
                break
        if ok:
            return True
    return False


# Example: the theorem for the 5-cycle C5, which has chromatic number 3.
if __name__ == "__main__":
    n = 5
    c5_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)}
    chi = chromatic_number(c5_edges, n)
    print(f"chi(C5) = {chi}")                     # 3
    print(f"ESS leading term for n=100: {ess_extremal_estimate(100, chi)}")
    print(f"Turan edges T_2(100) = {turan_edges(100, 2)}")
    # C5 embeds in K_3(2): 3 parts of size 2 = 6 slots, enough for 5 vertices.
    print(f"C5 embeds in K_3(2): {embeds_in_blowup(c5_edges, n, 3, 2)}")
```
