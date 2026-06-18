I start with the capacity definition and immediately run into the wrong kind of object. `Theta(G)` is built from
`alpha(G^k)`, so a lower bound is just a matter of finding explicit independent sets in graph powers. That is why
the pentagon gives evidence quickly: one use gives only two safe symbols, while two uses give five safe words and
therefore a `sqrt(5)` lower bound. But an upper bound cannot come from checking a few powers. I need a certificate
whose algebra follows the strong product.

The certificate should make an independent set visibly small. If I assign unit vectors to vertices so that
non-adjacent vertices are orthogonal, then any independent set becomes a mutually orthogonal family. Now I can use
a unit handle vector as a measuring device. The squared projections of a unit vector onto a mutually orthogonal
unit family sum to at most one. So if every vertex vector has squared projection at least `1/t` onto the handle,
no independent set can have more than `t` vertices.

This suggests an optimization problem: choose the representation and the handle so that the worst inverse squared
projection is as small as possible. That value is already an upper bound on `alpha(G)`, but the capacity problem
still asks about every strong power. I have to check the product behavior before the one-use certificate is useful.

Tensor products give the first direction. If I represent `G` and `H`, then the tensor products of their vertex
vectors represent the strong product, and inner products multiply. Handle projections multiply too. That gives
`theta(G * H) <= theta(G) theta(H)`, which is enough for the capacity upper bound. The matrix formulation gives
the reverse inequality, so the optimized value actually satisfies
`theta(G * H) = theta(G) theta(H)`.

Now the one-use certificate controls every block length:
`alpha(G^k) <= theta(G^k) = theta(G)^k`, so `Theta(G) <= theta(G)`. This is the missing upper-bound mechanism:
the same object that bounds one independent set also survives the limit over graph powers.

For the pentagon, the geometry is exactly tight. I open five equal unit ribs around a common handle until
non-adjacent ribs are orthogonal. The handle projection is `5^(-1/4)`, so the inverse squared projection is
`sqrt(5)`. Tensoring this representation gives `alpha(C5^k) <= 5^(k/2)`. The earlier two-use code gives the
opposite inequality at the level of capacity, so `Theta(C5) = sqrt(5)`.

The matrix version explains why this is not only a drawing. The same value is the optimum of a positive
semidefinite program: maximize the sum of the entries of `X`, set `Tr(X)=1`, set `X_ij=0` on edges of the
confusability graph, and require `X` to be positive semidefinite. A stable set gives a feasible matrix, while the
dual side supplies the product equality. In the complementary coloring language this is the sandwich
`omega(H) <= theta(complement(H)) <= chi(H)`.
