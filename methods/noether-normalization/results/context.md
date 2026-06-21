## Research question

An affine algebraic object can be presented as a finitely generated algebra

`A = k[x_1,...,x_n]/I`.

That presentation is concrete but not automatically controlled. The question is whether every such algebra can be re-coordinatized so that part of it is a plain polynomial algebra and all remaining complexity is finite over that polynomial part.

The target form is to find algebraically independent elements `y_1,...,y_d in A` such that

`k[y_1,...,y_d] subset A`

and `A` is finite as a module over this subring.

## Background

For a finitely generated `k`-algebra, a polynomial subring `k[y_1,...,y_d]` is the coordinate ring of affine space `A^d_k`. If `A` is finite over that subring, then every element of `A` is integral over it: each remaining coordinate satisfies a monic polynomial whose coefficients live in `k[y_1,...,y_d]`.

Geometrically, for `X = Spec A`, the inclusion `k[y_1,...,y_d] -> A` gives a morphism

`X -> A^d_k`.

Module-finiteness says this morphism is finite. In the reduced or variety case, this means the projection has finite fibers in the scheme-theoretic sense and packages the original object as a finite cover of affine space. The coordinate ring may still contain singularities, nilpotents, or several components, but the map to the simple base is controlled by finite algebra.

## Baselines

- **Direct equation handling.** One can manipulate the ideal `I` directly to understand the structure of `A`.

- **Naive projection.** One can project from ambient affine space using some of the original coordinates.

- **Dimension counting alone.** The Krull dimension indicates how many independent parameters should remain.

- **Elimination alone.** Eliminating variables can produce equations among chosen coordinates, describing an image in a lower-dimensional space.

## Evaluation settings

The standard setting is a nonzero finitely generated algebra `A` over a field `k`. The cleanest geometric picture is for an affine variety over an infinite or algebraically closed field. The algebraic theorem itself is broader: it asserts the existence of a polynomial subring over which `A` is finite.

Success means finding `d = dim A` algebraically independent parameters `y_1,...,y_d` and proving that `A` is module-finite over `k[y_1,...,y_d]`. The resulting finite morphism `Spec A -> A^d_k` should preserve the original object while making it analyzable through a simpler base and finite fibers.
