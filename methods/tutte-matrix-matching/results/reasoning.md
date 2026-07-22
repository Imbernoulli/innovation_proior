I want the perfect-matching question to stop being a search through possible edge sets. First I separate the trivial parity case: if the graph has odd order, no pairing can cover every vertex, so there is nothing to compute there. The real work is even order. A matching is only a partition of the vertices into disjoint pairs, so I would like an algebraic expression whose native terms are already such pairings — then "does a matching exist" becomes "is this expression nonzero," and I never enumerate edge sets by hand.

A determinant is the obvious first thing to reach for, because its terms range over all the ways of pairing rows with columns. So suppose I just build a symbolic adjacency matrix: put a variable in position `(i, j)` when `{i, j}` is an edge, zero otherwise, and take the determinant. The determinant expands over permutations `sigma` as a signed sum of products of entries `a_{i, sigma(i)}`. A product survives only if every `a_{i, sigma(i)}` is a nonzero entry, i.e. every `(i, sigma(i))` is an edge. That is encouraging — but a permutation is not a pairing. Its cycle structure can have cycles of length three, four, and longer, and a length-three cycle is a closed triangle in the graph, not a matching edge. So the surviving terms of a plain adjacency determinant correspond to disjoint-cycle covers of the graph, which is a strictly larger family than perfect matchings. I should check whether that larger family corrupts the test.

Take two disjoint triangles on vertices `{0, 1, 2}` and `{3, 4, 5}`. The whole graph has six vertices but no perfect matching, because each triangle is an odd component and a pairing inside it must leave one vertex uncovered. If I write the symmetric variable adjacency matrix and expand its determinant symbolically, I get

```text
det(symmetric) = 4 * x01 * x02 * x12 * x34 * x35 * x45,
```

which is not the zero polynomial. So the symmetric determinant reports a false positive on a graph with no matching: the two surviving terms are the two 3-cycles, one per triangle, and together they cover all six vertices as a cycle cover even though no pairing does. A plain determinant therefore tests the wrong thing. I need to kill the odd cycles.

This is where skew-symmetry earns its place. If I make the matrix skew-symmetric — entry `x_ij` above the diagonal and `-x_ij` below — then odd-length permutation cycles cancel against their reversals, because reversing an odd cycle flips an odd number of the negated entries and so flips the overall sign while leaving the monomial fixed. The same two-triangles matrix, built skew-symmetric, gives

```text
det(skew) = 0,
```

which is the correct verdict. The contrast is exactly the diagnostic I wanted: same edges, same variables, only the sign convention below the diagonal differs, and that flip is what suppresses the spurious odd-cycle terms. The odd-order parity case falls out of the same mechanism — every odd-order skew-symmetric determinant is identically zero, which is consistent with there being no matching on an odd vertex set, so I do not even need a separate argument for it.

The skew-symmetric structure has more to give than just killing odd cycles in a determinant. For an even skew-symmetric matrix the determinant is the square of a Pfaffian, and the Pfaffian is itself the signed sum over ways to partition the indices into unordered pairs — with no extra numeric factor in front, each pairing contributing one product with sign `+1` or `-1`. That is the object whose terms are literally pairings, which is what I was after from the start. So I should work with the Pfaffian directly and let the determinant be its square at the end.

Now I build the filter. I number the vertices and place `x_ij` in position `(i, j)` with `i < j` only when `{i, j}` is an edge, `-x_ij` in the transposed position, and `0` on the diagonal and on every nonedge. Choosing the opposite orientation for some edges would only flip the signs of those variables, which cannot change whether the resulting polynomial is identically zero, so the orientation convention is free. The zeros are the real content: an entry of zero forces every Pfaffian term that uses that pair to vanish.

Trace a Pfaffian term. It picks a set of pairs covering all vertices. If one chosen pair is a nonedge, its entry is `0` and the whole product dies. If every chosen pair is an edge, those pairs are disjoint and cover all vertices, so they form a perfect matching, and the term is the product of that matching's edge variables (up to sign). So the surviving terms are in bijection with perfect matchings.

The one thing that could still ruin this is cancellation: if two matchings contributed with opposite signs and the same monomial, the Pfaffian could be zero even when matchings exist. Independence of the variables is supposed to prevent that, but I should see it happen rather than trust it. Take the 4-cycle on vertices `1, 2, 3, 4` with edges `12, 23, 34, 14`. It has exactly two perfect matchings, `{12, 34}` and `{23, 14}`. Expanding the `4 x 4` Pfaffian `a12 a34 - a13 a24 + a14 a23` on this matrix (with `a13 = a24 = 0`, since `13` and `24` are nonedges) gives

```text
Pf = x12 * x34 + x14 * x23.
```

Two terms, one per matching, distinct monomials, no cancellation — exactly as the independence argument predicts, and now I have seen it. The point is general: two different perfect matchings differ in at least one edge, so their products are different monomials in distinct variables, and distinct monomials cannot cancel in a polynomial ring over the integers regardless of their signs. So the Pfaffian vanishes identically if and only if the graph has no perfect matching. I can double-check the determinant relation on the same example: squaring,

```text
Pf^2 = x12^2 x34^2 + 2 x12 x14 x23 x34 + x14^2 x23^2,
```
