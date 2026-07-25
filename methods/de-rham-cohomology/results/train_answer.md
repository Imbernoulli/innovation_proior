The problem is to turn the local language of calculus into a detector of global shape. On a small patch of a manifold everything looks like Euclidean space, so the usual tests report that a curl-free vector field should be a gradient, a divergence-free field should be a curl, and more generally a closed differential form should have a primitive. These local tests are blind to holes: a loop that winds around a missing point is invisible inside any single coordinate chart, yet the loop cannot be shrunk to a point and the hoped-for global potential does not exist. Potential theory, Stokes' theorem on its own, and singular homology each capture part of the story, but none of them directly records the failure of a smooth differential equation to have a smooth solution. What is needed is a calculus-level invariant that keeps the differential-form data and quotients out the local clutter.

The right construction is de Rham cohomology. Let M be a smooth manifold and let Omega^k(M) be the space of smooth k-forms. The exterior derivative d raises degree by one and satisfies d^2 = 0, so every exact form d eta is automatically closed. The k-th de Rham cohomology group is the quotient H_dR^k(M) = ker(d: Omega^k -> Omega^(k+1)) / im(d: Omega^(k-1) -> Omega^k). It is precisely the space of closed k-forms modulo those that are globally differentials of lower-degree forms. The Poincare lemma removes all positive-degree local information: on any chart diffeomorphic to R^n every closed form of positive degree is exact. Therefore H_dR^k measures only global obstructions. Integration over smooth k-cycles gives a well-defined pairing, and de Rham's theorem says this pairing identifies H_dR^k(M) with real singular cohomology H^k(M; R). In short, de Rham cohomology is the space of periods of closed forms, with exact forms discarded because their periods are zero.

This quotient is the entire deliverable, and it is worth setting down exactly as the theorem that closes the construction. Let $M$ be a smooth manifold, let $\Omega^k(M)$ be the smooth $k$-forms, and let $d:\Omega^k(M)\to\Omega^{k+1}(M)$ be the exterior derivative, so that

$$Z^k(M)=\ker\bigl(d:\Omega^k(M)\to\Omega^{k+1}(M)\bigr), \qquad B^k(M)=\operatorname{im}\bigl(d:\Omega^{k-1}(M)\to\Omega^k(M)\bigr)$$

satisfy $B^k(M)\subset Z^k(M)$, since $d^2=0$, and define

$$H^k_{dR}(M) = Z^k(M)/B^k(M).$$

For every smooth $k$-cycle $c$, integration $I([\omega])([c]) = \int_c \omega$ is well defined on $H^k_{dR}(M)\times H_k(M;\mathbb{R})$, because Stokes' theorem sends it to zero whenever $\omega$ is exact or $c$ is a boundary. The resulting homomorphism

$$I : H^k_{dR}(M) \xrightarrow{\ \cong\ } H^k(M;\mathbb{R})$$

is an isomorphism onto real singular cohomology, proved by matching the two theories on convex charts, where both are trivial in positive degree, and propagating that match across a good cover with the Mayer-Vietoris sequences and the five lemma. When $M$ is oriented of dimension $n$ and admits a finite good cover, the same integration pairing

$$([\alpha],[\beta]) \mapsto \int_M \alpha\wedge\beta$$

identifies $H^{n-k}_{dR}(M)$ with the dual of the compactly-supported group $H_c^k(M)$, specializing on a compact oriented $M$ to Poincare duality between $H^k_{dR}(M)$ and $H^{n-k}_{dR}(M)$. That is the finished object: the real vector space of periods of closed forms, exact forms discarded because they integrate to zero on every cycle, equal on the nose to the topologist's singular cohomology with real coefficients.
