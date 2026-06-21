# Weil conjectures: the cohomological route

The distinctive solution idea is to reinterpret the point counts of a variety over a finite field as traces of Frobenius on cohomology. For `X/F_q`, the numbers

`N_r = |X(F_{q^r})|`

are packaged into

`Z(X,t) = exp(sum_{r >= 1} N_r t^r/r)`.

The decisive move is that `F_{q^r}`-points are fixed points of `Frob^r`. Grothendieck's `l`-adic etale cohomology gives finite-dimensional spaces `H^i(X_bar,Q_l)` on which Frobenius acts, and the Grothendieck-Lefschetz trace formula gives

`N_r = sum_i (-1)^i Tr(Frob^r | H^i(X_bar,Q_l))`.

Linear algebra then converts the generating function into

`Z(X,t) = product_i det(1 - t Frob | H^i(X_bar,Q_l))^((-1)^(i+1))`.

This explains rationality: the zeta function is built from finitely many characteristic polynomials. It also explains the functional equation: Poincare duality pairs cohomological degrees `i` and `2d-i`, forcing a spectral symmetry. The Betti-number statement comes from comparison with ordinary cohomology in characteristic-zero lift situations.

The deepest part is the Riemann-hypothesis bound. After the trace formula, the problem is no longer "estimate every finite-field point count directly." It becomes "control the eigenvalues of Frobenius." Deligne's theory of weights proves that Frobenius eigenvalues in degree `i` have absolute value `q^(i/2)`. That spectral purity is exactly the cancellation statement encoded by the Weil conjectures.

So the route is:

1. Count finite-field points as Frobenius fixed points.
2. Count fixed points by a Lefschetz trace formula in `l`-adic cohomology.
3. Express the zeta function as an alternating product of Frobenius characteristic polynomials.
4. Use duality for the functional equation.
5. Use Deligne's weights to prove the Riemann-hypothesis eigenvalue bound.

The conceptual transformation is the main innovation: arithmetic counting is turned into geometry through cohomology, then into a spectral problem through Frobenius eigenvalues.
