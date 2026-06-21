## Research question

A manifold has holes in many dimensions, and ordinary homology lists them degree by degree. A closed oriented `n`-manifold has a special global structure: locally it is always `R^n`, and its orientation lets all local `n`-dimensional pieces carry compatible signs. The question is whether that global top-dimensional coherence forces a symmetry between homology in degree `k` and cohomology in degree `n-k`, and if so, what algebraic operation produces that symmetry.

## Background

The local invariant behind orientation is `H_n(M, M - {x}; Z)`. For an `n`-manifold this group is infinite cyclic at each point, and a local orientation is a choice of generator. A global orientation is a locally consistent choice of those generators. On a closed oriented manifold, these compatible local choices assemble into a fundamental class `[M] in H_n(M; R)` with the property that its image near every point is the chosen local generator.

Cohomology already behaves like a system of tests: a class in `H^k(M; R)` evaluates on `k`-cycles, and the universal coefficient theorem makes this literal over fields. Cup product combines tests: if `alpha in H^k` and `beta in H^l`, then `alpha cup beta in H^{k+l}`. Top-degree cohomology on an oriented compact manifold can be evaluated on the fundamental class, so a product in degree `n` gives a number.

Cap product is the complementary operation to cup product. At the chain level it contracts a chain by a cochain:

```
C_p(X; R) x C^q(X; R) -> C_{p-q}(X; R).
```

The boundary identity for cap product makes it descend to homology and cohomology. Thus a top-dimensional cycle together with a degree-`k` cohomology class naturally produces a homology class in degree `n-k`.

For smooth manifolds over `R`, differential forms give the same testing language concretely. Closed `k`-forms represent de Rham classes; wedge product sends degrees `k` and `n-k` to degree `n`; integration over an oriented compact manifold evaluates the result. Stokes' theorem explains why exact terms vanish when there is no boundary.

## Baselines

- **Betti-number symmetry from dual cell decompositions.** A triangulated closed manifold often admits a dual cell structure: a `k`-cell meets one `(n-k)`-cell transversely. This makes the expected dimension reversal visually clear and depends on choosing compatible cell structures.

- **Universal coefficient testing in the same degree.** Over a field, `H^k(M; F)` is the linear dual of `H_k(M; F)`. This says that cohomology detects `k`-cycles via evaluation on chains in the same degree.

- **Cup product and top-degree evaluation.** The pairing `<alpha cup beta, [M]>` is meaningful once a fundamental class exists and `deg alpha + deg beta = n`.

- **De Rham integration.** On an oriented smooth manifold, `int_M omega wedge eta` is the analytic expression of complementary-degree testing, with compactness ensuring finite integrals and Stokes' theorem handling boundary terms.

## Evaluation settings

The theorem should be checked in settings where the answer is independently visible: spheres, tori, closed orientable surfaces, products of circles, and finite cell decompositions with obvious dual cells. It should also handle the stress cases that explain the hypotheses: nonorientable manifolds where signs cannot be chosen globally, compact manifolds with boundary where relative groups enter, and noncompact manifolds where ordinary top homology or ordinary integration no longer gives the right target.

The success criterion is exactness, not an estimate: the degree-reversing map must be an isomorphism, and the resulting complementary pairing must be nonsingular over field coefficients. In the smooth real setting, the integration pairing must depend only on cohomology classes, not on the chosen forms.
