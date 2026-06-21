I frame the problem as the zero-error capacity of a noisy channel. The signal values are vertices of a graph, and two values are joined by an edge whenever the receiver might confuse them. A single-use code is simply an independent set in this confusability graph, so its largest size is the independence number `alpha(G)`. When messages are allowed to use `k` positions, the confusability graph becomes the `k`th strong power `G^k`, and the asymptotic zero-error capacity is `Theta(G) = lim alpha(G^k)^(1/k)`. This limit is easy to write but hard to evaluate, because independence numbers are already difficult for one graph and the strong product compounds the difficulty across an exponentially growing vertex set.

The canonical method I propose for taming this limit is the Lovasz theta function. The guiding idea is to certify that every independent set is small by embedding the vertices into a Hilbert space with a geometric constraint. I assign a unit vector `u_v` to each vertex `v`, requiring that `u_v` and `u_w` be orthogonal whenever `v` and `w` are non-adjacent. Any independent set then becomes a set of mutually orthogonal unit vectors. I introduce a unit handle vector `c`. For any unit vector, the sum of squared projections onto a mutually orthogonal family is at most one. Therefore, if every vertex vector has squared projection at least `1/t` onto `c`, no independent set can contain more than `t` vertices.

This one-shot certificate becomes useful for the capacity problem only if it respects the strong product. Tensor products give exactly the needed multiplicativity. If `U` and `V` are orthogonal representations of `G` and `H`, then the tensor products `u_i ⊗ v_j` form an orthogonal representation of the strong product `G ⊠ H`, and inner products multiply. The optimized value `theta(G)`, which I define as the minimum over representations and handles of `max_v 1 / (c^T u_v)^2`, satisfies `theta(G ⊠ H) = theta(G) theta(H)`. Consequently `alpha(G^k) <= theta(G^k) = theta(G)^k`, and taking `k`th roots gives `Theta(G) <= theta(G)`. The same object that bounds a single independent set therefore bounds every block length.

For the five-cycle `C5`, the geometry is tight. I place five equal umbrella ribs around a common handle so that non-adjacent ribs are orthogonal. The handle projection is `5^{-1/4}`, so the inverse squared projection is `sqrt(5)`. A two-use code of size five, such as `(0,0), (1,2), (2,4), (3,1), (4,3)`, shows `alpha(C5^2) >= 5` and therefore `Theta(C5) >= sqrt(5)`. Matching this with the theta upper bound yields `Theta(C5) = theta(C5) = sqrt(5)`, the first capacity value that could not be obtained from one-shot clique or fractional-packing arguments.

The same value admits an equivalent semidefinite program that is often easier to compute and reason about. In the primal form, `theta(G)` is the maximum of `<J, X>` subject to `Tr(X) = 1`, `X_ij = 0` on every edge of `G`, and `X` being positive semidefinite. A stable set gives a feasible rank-one matrix, which explains why the optimum upper bounds `alpha(G)`. The dual side supplies the product equality through the tensor structure. In the complementary coloring convention the theta function sits in the sandwich `omega(H) <= theta(complement(H)) <= chi(H)`, linking independence, coloring, and the same convex relaxation.

Because the theta function is computable in polynomial time through semidefinite programming, it gives a tractable upper bound on an otherwise intractable asymptotic quantity. It also illustrates a broader pattern I find valuable: when a combinatorial limit is hard because of products, I look for a convex or geometric invariant that is multiplicative under those products. The Lovasz theta function is the archetypal such invariant for graph Shannon capacity.

```python
import itertools
import numpy as np


def lovasz_theta_c5_umbrella():
    """Construct the umbrella representation that gives theta(C5) <= sqrt(5)."""
    n = 5
    # Handle projection chosen so that non-adjacent ribs can be orthogonal.
    c = 5.0 ** (-0.25)
    s = np.sqrt(1.0 - c * c)
    angles = 2.0 * np.pi * np.arange(n) / n
    vectors = np.column_stack([
        s * np.cos(angles),
        s * np.sin(angles),
        np.full(n, c)
    ])
    handle = np.array([0.0, 0.0, 1.0])

    # Verify unit lengths.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    # Verify orthogonality for every non-adjacent pair.
    for i in range(n):
        j = (i + 2) % n
        assert np.isclose(np.dot(vectors[i], vectors[j]), 0.0)

    # The Lovasz theta upper bound from this representation.
    worst_proj_sq = max(np.dot(v, handle) ** 2 for v in vectors)
    theta_bound = 1.0 / worst_proj_sq
    return theta_bound


def c5_strong_power_adjacency(k):
    """Adjacency matrix of the kth strong power of C5."""
    n = 5 ** k
    tuples = np.array([[(i // 5 ** m) % 5 for m in range(k)] for i in range(n)])

    # Coordinate pairs that are equal or adjacent in C5.
    ok = np.zeros((5, 5), dtype=bool)
    for a in range(5):
        for b in range(5):
            if a == b or (a - b) % 5 in (1, 4):
                ok[a, b] = True

    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(ok[tuples[i], tuples[j]]):
                adj[i, j] = adj[j, i] = True
    return adj


def max_independent_set(adj):
    """Return size and one maximum independent set (exact for small graphs)."""
    n = adj.shape[0]
    # Start from a greedy lower bound.
    best = []
    remaining = set(range(n))
    while remaining:
        v = remaining.pop()
        best.append(v)
        remaining -= {u for u in range(n) if adj[v, u]}

    # Exhaustively search for anything larger.
    for size in range(len(best) + 1, n + 1):
        found = False
        for subset in itertools.combinations(range(n), size):
            if all(not adj[i, j] for i, j in itertools.combinations(subset, 2)):
                best = list(subset)
                found = True
                break
        if not found:
            break
    return len(best), best


if __name__ == "__main__":
    print("=== Lovasz theta of C5 from the umbrella representation ===")
    theta_bound = lovasz_theta_c5_umbrella()
    print(f"theta(C5) <= {theta_bound}")
    print(f"sqrt(5)   = {np.sqrt(5.0)}")
    print()

    print("=== Brute-force independence numbers ===")
    for k in (1, 2):
        adj = c5_strong_power_adjacency(k)
        alpha, code = max_independent_set(adj)
        print(f"alpha(C5^{k}) = {alpha}")
        if k == 2:
            print(f"Example maximum independent set: {code}")
            print(f"Capacity lower bound: alpha(C5^2)^(1/2) = {alpha ** 0.5}")
```

The script constructs the umbrella representation, checks the orthogonality conditions, and brute-forces the independence numbers of `C5` and `C5^2` to confirm that the geometric bound matches the combinatorial capacity. Running it reproduces the `sqrt(5)` value that makes the pentagon the classic demonstration of the method.
