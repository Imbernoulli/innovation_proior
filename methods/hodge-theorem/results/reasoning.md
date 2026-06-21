I start with the quotient that de Rham gives me. A `k`-dimensional cohomology class is not a single object on the manifold; it is every closed `k`-form modulo the ones of the form `d eta`. That is the right topological answer, but it leaves me with a representative problem. If I want topology to become something I can see analytically on the manifold itself, I need a way to choose one form out of the whole affine family

`alpha + d Omega^{k-1}(M)`.

Choosing by coordinates is hopeless, and choosing by periods only moves the class into the dual of homology. I want a representative that the geometry of the manifold itself chooses.

A Riemannian metric gives me exactly the extra structure that a quotient lacks: an inner product. Once forms have an `L^2` inner product, the phrase "best representative" can mean something. In a finite-dimensional Euclidean space, if I have an affine subspace and I want the vector of least norm, I take the vector perpendicular to the directions along which I am allowed to move. Here the allowed motion inside a de Rham class is `d eta`. So the least-norm representative, if it exists, should be orthogonal to every exact form:

`<alpha, d eta> = 0` for all `eta`.

On a closed manifold, integration by parts rewrites this as

`<d^* alpha, eta> = 0` for all `eta`,

so the condition is `d^* alpha = 0`. But the representative must also stay in the cohomology class, so it must be closed: `d alpha = 0`. The analytic condition is therefore not just "closed"; it is closed and co-closed.

Now that pair of equations has the right shape to be one elliptic equation. Define

`Delta = d d^* + d^* d`.

Then

`<Delta alpha, alpha> = ||d alpha||^2 + ||d^* alpha||^2`.

So on a closed Riemannian manifold, `Delta alpha = 0` is exactly the same as `d alpha = 0` and `d^* alpha = 0`. The least-norm condition inside the topological class has become the Laplacian equation on differential forms. Harmonic functions were only degree zero; harmonic forms are the same analytic idea in every degree.

The first wall is existence. Orthogonality tells me what the minimizer must satisfy, but the space of smooth forms is infinite-dimensional, and the exact forms need not behave like a closed finite-dimensional subspace for free. If I simply say "project onto the orthogonal complement of exact forms," I have skipped the real analytic issue. I need compactness and ellipticity to make the projection legitimate.

The Laplacian is self-adjoint and elliptic. Compactness of the manifold is what makes the elliptic theory behave like a controlled infinite-dimensional analogue of linear algebra: the kernel of `Delta` is finite-dimensional, the image is closed in the right smooth/Sobolev sense, and a Green operator solves `Delta u = v` on the orthogonal complement of the harmonic forms. Elliptic regularity then brings weak solutions back to smooth forms. This is the analytic engine; without it, the quotient-to-projection idea is only a formal picture.

So I should expect an identity of operators

`I = H + Delta G = H + G Delta`,

where `H` is orthogonal projection onto `ker Delta` and `G` is the Green operator, zero on harmonic forms and inverse to `Delta` on their orthogonal complement. If I expand `Delta`, every smooth `k`-form `omega` becomes

`omega = H omega + d d^* G omega + d^* d G omega`.

This is already the whole decomposition:

`Omega^k(M) = Harm^k(M) oplus im d oplus im d^*`.

The three summands are orthogonal. A harmonic form is orthogonal to exact forms because `d^* h=0`, and it is orthogonal to co-exact forms because `d h=0`. Exact and co-exact forms are orthogonal because `<d a, d^* b> = <d^2 a, b> = 0`. The decomposition is not a recombination of topology and analysis after the fact; it is the point where the metric turns the topological ambiguity into perpendicular analytic directions.

Now I test whether this decomposition really gives one representative per de Rham class. Take a closed form `omega`. Decompose it:

`omega = h + d a + d^* b`,

with `h` harmonic. Since `omega` and `h` and `d a` are closed, the remaining term `d^* b` must also be closed:

`d d^* b = 0`.

Then

`||d^* b||^2 = <d^* b, d^* b> = <d d^* b, b> = 0`.

So the co-exact part vanishes. The closed form reduces to

`omega = h + d a`.

Thus `omega` and `h` define the same de Rham class. Every class has at least one harmonic representative.

For uniqueness, suppose a harmonic form is exact, say `h = d a`. Since harmonic means co-closed,

`||h||^2 = <h, d a> = <d^* h, a> = 0`.

So `h=0`. Therefore two harmonic representatives in the same class cannot differ by a nonzero exact form. The map

`Harm^k(M) -> H^k_dR(M),   h |-> [h]`

is both onto and one-to-one.

The minimization interpretation now falls out without adding a new theorem. If `h` is the harmonic representative of a class, every other representative has the form `h + d a`. The harmonic part is orthogonal to every exact form, so

`||h + d a||^2 = ||h||^2 + ||d a||^2`.

The norm is smallest at the harmonic representative, and equality forces the exact perturbation to vanish as a form. A topological class is represented by the unique smooth form of least `L^2` energy in that class.

This also explains why the assumptions are not cosmetic. If the manifold has a boundary, integration by parts produces boundary terms unless boundary conditions are imposed. Then "harmonic" by `Delta omega=0` can fail to mean simply closed and co-closed, and the cohomology carried by harmonic forms changes unless one selects absolute or relative boundary conditions. If the manifold is noncompact, the Green-operator and closed-range statements need decay, completeness, or spectral assumptions. The clean canonical statement belongs to compact manifolds without boundary.

So the final object is not an arbitrary smooth representative, nor a basis of cycles, nor a recombination of de Rham theory with the scalar Laplacian. The metric builds an energy landscape on each de Rham class. The Laplacian detects the critical points of that energy under exact variations, elliptic compactness makes the projection exist smoothly, and the Hodge decomposition proves that the critical point is present and unique in every class. The topology of the compact manifold can be read as the finite-dimensional space of harmonic forms:

`H^k_dR(M) ~= Harm^k(M)`.
