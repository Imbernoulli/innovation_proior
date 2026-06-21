A perfect matching asks for a global pairing of all of a graph's vertices into disjoint edges, and the trouble is that local compatibility does not compose. A graph can carry many plausible partial pairings and still strand its last few vertices, an odd number of vertices kills the question outright, and the genuine obstruction in the general case is parity after deletion: removing a set of vertices can leave too many odd components, each of which must export an edge to be covered. Petersen's regular-graph factorizations, with their alternating color changes along closed polygons, give structure but only for special degree patterns and only as operations on already-decomposed graphs. What I want instead is a single criterion that keeps the whole pairing problem in view at once, without ever committing to a particular partial matching, without a case analysis over the ways a greedy search can get stuck, that nonetheless separates genuinely different pairings and respects the parity obstruction. The natural place to look is the classical algebra of skew-symmetric arrays, whose facts — odd-order skew determinants vanish, even-order ones are perfect squares, and the square root expands over index pairings — are true before any graph is inserted and seem to be describing matchings already.

I propose encoding the graph in what I call the Tutte matrix. Number the vertices $V = \{1,\dots,n\}$. If $n$ is odd, no pairing can cover every vertex and we stop. For even $n$, assign an independent indeterminate $x_{ij}$ to each edge $\{i,j\}$ with $i<j$, and build the skew-symmetric matrix $T(G)$ by
$$T_{ij} = \begin{cases} x_{ij} & i<j,\ \{i,j\}\in E \\ -x_{ji} & i>j,\ \{i,j\}\in E \\ 0 & \text{otherwise.}\end{cases}$$
The diagonal and every non-edge are zero, and the lower triangle is the negated transpose of the upper, so $T(G)$ is skew-symmetric by construction. The choice of which orientation gets the positive sign is free: flipping it merely negates some variables, which cannot change whether the resulting determinant is identically zero, so nothing of substance rides on the convention. What matters is the zeros — they are the filter that makes illegal pairs vanish.

The reason this works is the Pfaffian. For an even skew-symmetric matrix the determinant is the square of the Pfaffian, and the Pfaffian is precisely the signed sum over all ways of partitioning the indices into unordered pairs,
$$\mathrm{Pf}(T) = \sum_{\text{pairings}} \varepsilon \prod (\text{paired entries}), \qquad \varepsilon\in\{+1,-1\},$$
with no extra numeric factor out front — each pairing contributes exactly one product, signed by the parity of its listed sequence. This is exactly the combinatorial shape of a perfect matching before the graph has decided which pairs are legal, which is why I do not first ask which matching to choose; I let the Pfaffian try every pairing and let the matrix decide. A raw determinant would not do, because its permutation expansion carries cycles and directions whose signs belong to directed arrangements rather than to unordered pairs; the Pfaffian is the object whose native terms are already pairings. Now inspect a surviving term. It selects pairs covering all $n$ vertices; if any selected pair is a non-edge, that entry is $0$ and the whole product dies, and if every selected pair is an edge then those pairs are pairwise disjoint and cover everything — that is a perfect matching. So the surviving monomials are not merely related to perfect matchings, they *are* the perfect matchings, each written as the product of its edge variables.

The one real worry is cancellation: if two matchings appeared with opposite signs, a numeric sum could erase them. This is exactly why the variables must be independent. Two distinct perfect matchings differ in at least one edge, so their products are distinct monomials in the polynomial ring, and distinct monomials over the integers cannot cancel regardless of sign. Hence $\mathrm{Pf}(T(G))$ is the zero polynomial precisely when $G$ has no perfect matching. Passing to the determinant adds no new combinatorics: $\det(T(G)) = \mathrm{Pf}(T(G))^2$, and squaring may erase signs but over an integral polynomial ring it preserves the boundary between zero and nonzero. Folding in the odd-order case, the same symbolic determinant decides every finite simple graph:
$$G \text{ has a perfect matching} \iff \mathrm{Pf}(T(G))\not\equiv 0 \iff \det(T(G))\not\equiv 0 \iff T(G) \text{ has full generic rank.}$$

What makes this algorithmic, without ever expanding the determinant symbolically, is that rank over a field is computable by elimination once the variables hold concrete values. Substitute each $x_{ij}$ independently from a large set $S$ in a field. If there is no perfect matching the determinant polynomial is identically zero, so every substitution yields a singular matrix. If there is one, the determinant is a nonzero polynomial of degree $n$, and by polynomial identity testing a random substitution fails to give full rank with probability at most $n/|S|$; independent repetition multiplies that false-negative risk down. The test is therefore one-sided and asymmetric. Full rank at a single random point is decisive, because a nonzero evaluation proves the polynomial is not identically zero and a matching must exist. Singularity after one trial is not decisive, since a nonzero polynomial can still vanish at a particular point, especially over a small field — so repeated singular trials are reported as nonexistence only with one-sided error. The implementation below mirrors exactly this decision consequence and nothing more: it returns False immediately for odd $n$, instantiates the signed skew matrix at random values over a prime field, computes the modular rank by Gaussian elimination, returns True the moment a trial is full rank, and returns False if every trial is singular. It deliberately does not construct a witness matching or implement the later Rabin–Vazirani maximum-matching search; the conceptual core is the whole point — encode all vertex pairings in one Pfaffian, use the graph's zeros to kill illegal pairings, and use independent edge variables to keep legal pairings from cancelling.

```python
"""Randomized Tutte-matrix perfect-matching test over a prime field.

This is a compact reference implementation of the decision consequence:
instantiate the symbolic skew-symmetric matrix at random field values and
test whether any trial has full rank.
"""

from __future__ import annotations

import random
from collections.abc import Iterable


DEFAULT_PRIME = 2_147_483_647


def _rank_mod(matrix: list[list[int]], prime: int) -> int:
    a = [row[:] for row in matrix]
    n_rows = len(a)
    n_cols = len(a[0]) if a else 0
    rank = 0

    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if a[row][col] % prime:
                pivot = row
                break
        if pivot is None:
            continue

        a[rank], a[pivot] = a[pivot], a[rank]
        inv = pow(a[rank][col] % prime, prime - 2, prime)
        a[rank] = [(value * inv) % prime for value in a[rank]]

        for row in range(n_rows):
            if row == rank:
                continue
            factor = a[row][col] % prime
            if factor:
                a[row] = [
                    (x - factor * y) % prime
                    for x, y in zip(a[row], a[rank], strict=True)
                ]

        rank += 1
        if rank == n_rows:
            break

    return rank


def tutte_matrix(
    n_vertices: int,
    edges: Iterable[tuple[int, int]],
    *,
    prime: int = DEFAULT_PRIME,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Return one random field instantiation of the Tutte matrix.

    Vertices are numbered `0..n_vertices-1`; loops are ignored because they
    cannot participate in a perfect matching.
    """

    if rng is None:
        rng = random.Random()

    matrix = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
    for u, v in edges:
        if u == v:
            continue
        if not (0 <= u < n_vertices and 0 <= v < n_vertices):
            raise ValueError("edge endpoint outside vertex range")
        i, j = sorted((u, v))
        value = rng.randrange(1, prime)
        matrix[i][j] = value
        matrix[j][i] = (-value) % prime
    return matrix


def has_perfect_matching(
    n_vertices: int,
    edges: Iterable[tuple[int, int]],
    *,
    trials: int = 8,
    prime: int = DEFAULT_PRIME,
    seed: int | None = None,
) -> bool:
    """Return True on a full-rank trial, otherwise False with one-sided error."""

    if n_vertices % 2:
        return False

    edge_list = list(edges)
    rng = random.Random(seed)
    for _ in range(trials):
        matrix = tutte_matrix(n_vertices, edge_list, prime=prime, rng=rng)
        if _rank_mod(matrix, prime) == n_vertices:
            return True
    return False


if __name__ == "__main__":
    assert has_perfect_matching(4, [(0, 1), (1, 2), (2, 3)], seed=1)
    assert not has_perfect_matching(3, [(0, 1), (1, 2), (2, 0)], seed=1)
    assert not has_perfect_matching(4, [(0, 1), (1, 2)], seed=1)
```
