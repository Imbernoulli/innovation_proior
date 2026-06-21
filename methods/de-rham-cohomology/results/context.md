## Research question

How can smooth calculus on a manifold detect global topology? Locally, a manifold looks like `R^n`, and local calculus often says that a closed differential condition should come from a potential. Globally, a loop can wind around a missing point, a surface can enclose a void, and a coordinate potential can fail to exist even though all local tests pass.

The goal is to turn this failure into an invariant. A satisfactory construction should start from the calculus of differential forms, use the exterior derivative as the common language for gradient, curl, and divergence, respect change of coordinates, and explain why a form that is locally a derivative may still carry global information when integrated over cycles.

## Background

Differential forms are the objects naturally integrated over oriented manifolds of matching dimension: `1`-forms over curves, `2`-forms over surfaces, and `k`-forms over `k`-dimensional chains. The wedge product is alternating, so repeated directions vanish, and the exterior derivative `d` raises degree by one. In local coordinates, if `alpha = sum_I alpha_I dx_I`, then `d alpha = sum_{I,j} (partial alpha_I / partial x_j) dx_j wedge dx_I`.

The exterior derivative satisfies two decisive identities. First, it obeys the graded product rule `d(alpha wedge beta) = d alpha wedge beta + (-1)^k alpha wedge d beta` when `alpha` has degree `k`. Second, `d^2 = 0`, ultimately because mixed partial derivatives commute while wedge factors change sign. This makes every derivative of a form automatically satisfy a closedness condition.

Stokes' theorem is the bridge from calculus to topology. For a compact oriented manifold-with-boundary `X` and an `(n-1)`-form `omega`, it says `int_X d omega = int_boundary X omega`. In the half-space proof, the theorem reduces to Fubini plus the one-dimensional fundamental theorem of calculus; on manifolds, partitions of unity move the local statement into coordinate charts and sum the pieces.

The local obstruction disappears on Euclidean space. The Poincare lemma says that on `R^n`, and on any manifold diffeomorphic to `R^n`, every closed form of positive degree is exact. Thus closedness alone is not a local invariant of a hole; locally, closed forms have primitives.

The punctured plane gives the basic global warning. On `R^2 - {(0,0)}`, the form `(x dy - y dx)/(x^2 + y^2)` is closed. Around the unit circle it integrates to `2*pi`, so it cannot be the differential of a globally defined function: by Stokes, an exact form integrates to zero on a closed cycle.

## Baselines

- **Local vector calculus.** The identities `curl grad = 0` and `div curl = 0` are special cases of `d^2 = 0`. Gap: these identities say derivatives satisfy compatibility equations, but they do not separate local compatibility from global obstruction.

- **Potential functions.** If a `1`-form is `df`, then its integral along a path depends only on endpoints, and its integral around any closed loop is zero. Gap: on a space with a hole, a closed form can look like `df` in every small chart while no single-valued global `f` exists.

- **Poincare lemma on contractible charts.** Contractible coordinate patches have no positive-degree cohomology: closed forms are exact there. Gap: this confirms that the interesting obstruction is invisible in one chart and must be assembled globally.

- **Stokes' theorem alone.** Stokes explains why exact forms vanish on cycles and why closed forms have integrals unchanged under replacing a cycle by a homologous one. Gap: without quotienting out the exact forms, one still keeps calculus artifacts that all cycles fail to see.

- **Singular homology.** Chains, cycles, and boundaries already encode holes topologically. Gap: this language does not by itself explain why smooth differential equations of the form `d eta = omega` detect the same holes.

## Evaluation settings

The artifact is a theorem and proof, not an experiment. The natural test cases are smooth manifolds, contractible open subsets of `R^n`, the circle, the punctured plane, compact oriented manifolds, and smooth singular cycles.

Success means proving that exact forms are closed, that local closed forms of positive degree are exact on Euclidean charts, that integration over cycles is unchanged by adding exact forms or boundaries, and that the resulting calculus invariant agrees with singular cohomology with real coefficients under the standard comparison hypotheses.

The proof should also account for the oriented integration pairing. On an oriented `n`-manifold, compactly supported `k`-classes pair with `(n-k)`-classes by `([alpha],[beta]) -> int_M alpha wedge beta`; in the compact oriented case this specializes to the usual duality between complementary degrees.

## Code framework

No executable implementation is needed. The field-appropriate scaffold is a proof sequence:

1. Define the graded vector spaces of forms and the exterior derivative.
2. Use `d^2 = 0` to form a complex and distinguish closed forms from exact forms.
3. Use Stokes' theorem to prove that exact forms vanish on closed cycles and that closed forms have cycle integrals depending only on homology classes.
4. Use the Poincare lemma or the homotopy operator on `R x X` to show that local closed positive-degree forms have primitives.
5. Assemble local calculations with covers and Mayer-Vietoris, using the five lemma, to compare the calculus construction with singular cohomology.
