Consider a compact smooth manifold X and an elliptic differential or pseudodifferential operator P acting between complex vector bundles over X. Elliptic regularity tells us that, once the spaces of smooth sections are completed in suitable Sobolev norms, P becomes a Fredholm operator. Its analytic index

ind(P) = dim ker(P) - dim coker(P)

is therefore a well-defined integer. The difficulty is that this integer is built from the solutions of a partial differential equation, and one would like to compute it without ever solving the equation. Direct spectral analysis is not enough: the kernel and cokernel can change drastically under compact or lower-order perturbations, even though their difference stays constant. Classical formulas such as the Euler characteristic for the de Rham complex or Hirzebruch-Riemann-Roch for the Dolbeault complex give the right kind of answer in special cases, but they do not cover arbitrary elliptic operators on arbitrary compact smooth manifolds. Heat-kernel methods are powerful, yet for a general elliptic operator the small-time asymptotics involve too many derivatives of the total symbol; they become truly clean only for Dirac-type operators where Clifford multiplication forces the cancellations that produce characteristic forms. What is missing is a single topological invariant that captures the Fredholm index using only the principal symbol and the topology of X.

The Atiyah-Singer Index Theorem supplies exactly that invariant. The theorem says that the analytic Fredholm index of P equals the topological index of the principal symbol of P. In more detail, the principal symbol of P is a bundle isomorphism over the cotangent bundle away from the zero section, so it defines a compactly supported K-theory class [sigma(P)] in K^0_c(T^*X). The analytic index is the homomorphism that sends this class to dim ker(P) - dim coker(P). The topological index is the K-theoretic pushforward of [sigma(P)] to a point, which lands in K^0(pt) = Z. The theorem asserts that these two homomorphisms are equal:

a-ind([sigma(P)]) = t-ind([sigma(P)]).

Passing from K-theory to cohomology via the Chern character and the Thom isomorphism gives the more familiar integral formula

ind(P) = < Todd(TX tensor C) cup phi^{-1}(ch [sigma(P)]), [X] >.

The placement of the Thom isomorphism depends on orientation conventions, but the invariant content is fixed: the analytic defect of the elliptic PDE is a topological invariant of the symbol bundle over the cotangent bundle and the manifold that carries it.

This is the right formulation because every structural property of the Fredholm index is mirrored by a corresponding construction in K-theory. Lower-order terms are compact perturbations, so the analytic index factors through the principal symbol class. The symbol itself is exactly the data K-theory was invented to study: two vector bundles glued by an isomorphism outside a compact set. The pushforward to a point is forced by functoriality under diffeomorphisms, additivity under direct sums, homotopy invariance, excision, compatibility with Thom products, and normalization on the model class over a point. Once both the analytic and topological index maps are shown to satisfy these properties, uniqueness forces them to be the same. Special cases such as Gauss-Bonnet-Chern for the de Rham complex, Hirzebruch-Riemann-Roch for Dolbeault operators, and the signature theorem for the signature operator all emerge as particular computations of the same universal map.

Stated in full, with every symbol pinned down, this is the deliverable. Let $X$ be a compact smooth manifold and $P : C^\infty(X; E^0) \to C^\infty(X; E^1)$ an elliptic differential or pseudodifferential operator between complex vector bundles. Its principal symbol $\sigma(P)$ is invertible off the zero section of $T^*X$, so it determines a compactly supported K-theory class $[\sigma(P)] \in K^0_c(T^*X)$: two bundles pulled back from $X$ together with an isomorphism outside a compact set. Two homomorphisms out of this group compute the same integer. The analytic index

$$\mathrm{a\text{-}ind} : K^0_c(T^*X) \to \mathbb{Z}, \qquad \mathrm{a\text{-}ind}([\sigma(P)]) = \dim\ker(P) - \dim\mathrm{coker}(P),$$

is defined by choosing any elliptic operator representing the class and taking its Fredholm index; it is well-defined because homotopic symbols give Fredholm-homotopic operators and compact perturbations do not move the index. The topological index

$$\mathrm{t\text{-}ind} : K^0_c(T^*X) \to K^0(\mathrm{pt}) = \mathbb{Z}$$

is the K-theoretic pushforward of the same class to a point, built from an embedding of $X$ into a sphere, the Thom isomorphism, and Bott periodicity. The theorem is the identity of these two maps,

$$\mathrm{a\text{-}ind} = \mathrm{t\text{-}ind} \;:\; K^0_c(T^*X) \to \mathbb{Z},$$

which holds because both sides are additive, homotopy-invariant, functorial under diffeomorphism, compatible with excision, compatible with the Thom-isomorphism products used in the pushforward, and normalized to $1$ on the point model — properties that pin down such a map uniquely, so agreement on the generating construction forces agreement everywhere. Passed through the Chern character and the Thom isomorphism $\varphi$, the same identity becomes the explicit cohomological formula

$$\operatorname{ind}(P) \;=\; \Big\langle\, \mathrm{Todd}(TX \otimes \mathbb{C}) \,\cup\, \varphi^{-1}\big(\mathrm{ch}\,[\sigma(P)]\big),\; [X] \,\Big\rangle,$$

with everything on the right built only from the symbol class and the tangent bundle of $X$ — no reference to $\ker(P)$ or $\mathrm{coker}(P)$ survives. On a compact Riemann surface of genus $g$, with $P$ the Dolbeault operator $\bar\partial_L$ twisted by a holomorphic line bundle $L$ of degree $d$, this formula collapses to $\mathrm{ch}(L) \cup \mathrm{Todd}(TX) = 1 + \big(c_1(L) + \tfrac{1}{2} c_1(TX)\big)$, whose pairing with $[X]$ is $d + \tfrac{1}{2}(2 - 2g) = d - g + 1$ — exactly the classical Riemann-Roch number $\dim H^0(X;L) - \dim H^1(X;L)$. That agreement, checked against a known analytic answer rather than merely type-correct, is the formula's calibration. The theorem itself is the pair of homomorphisms $\mathrm{a\text{-}ind}$ and $\mathrm{t\text{-}ind}$, proved equal on all of $K^0_c(T^*X)$, with the cohomological identity above as its explicit, computable face.
