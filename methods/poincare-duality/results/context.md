## Research question

A manifold has holes in many dimensions, but ordinary homology lists them degree by degree with no built-in reason that degree `k` should know anything about degree `n-k`. A closed oriented `n`-manifold looks different from a general space: locally it is always `R^n`, and globally its orientation lets all local `n`-dimensional pieces carry compatible signs. The question is whether that global top-dimensional coherence forces a symmetry between a hole and a complementary-dimensional way of detecting it.

The desired theorem should do more than match Betti numbers. It should produce a canonical map, explain what datum supplies the shift by `n`, and turn the geometric intuition of transverse intersection into an algebraic statement. If a `k`-dimensional class is real, there should be a complementary test that sees it; if every complementary test misses it, the class should have been zero.

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

- **Betti-number symmetry from dual cell decompositions.** A triangulated closed manifold often admits a dual cell structure: a `k`-cell meets one `(n-k)`-cell transversely. This makes the expected dimension reversal visually clear. The limitation is that it depends on choosing cell structures and first suggests only a rank symmetry, not a canonical operation on homology and cohomology for arbitrary manifolds.

- **Universal coefficient testing in the same degree.** Over a field, `H^k(M; F)` is the linear dual of `H_k(M; F)`. This says that cohomology detects `k`-cycles, but it does not explain why a closed oriented `n`-manifold should identify those tests with objects of complementary dimension.

- **Cup product and top-degree evaluation.** The pairing `<alpha cup beta, [M]>` is meaningful once a fundamental class exists and `deg alpha + deg beta = n`. The limitation is that the product alone does not show nondegeneracy; a space can have a cup product without every nonzero class having a complementary partner.

- **De Rham integration.** On an oriented smooth manifold, `int_M omega wedge eta` is the analytic expression of complementary-degree testing. The limitation is that the clean ordinary pairing needs compactness; on noncompact manifolds the compact-support version is the stable form, and on manifolds with boundary Stokes' theorem leaves boundary terms unless relative conditions are imposed.

## Evaluation settings

The theorem should be checked in settings where the answer is independently visible: spheres, tori, closed orientable surfaces, products of circles, and finite cell decompositions with obvious dual cells. It should also handle the stress cases that explain the hypotheses: nonorientable manifolds where signs cannot be chosen globally, compact manifolds with boundary where relative groups enter, and noncompact manifolds where ordinary top homology or ordinary integration no longer gives the right target.

The success criterion is exactness, not an estimate: the degree-reversing map must be an isomorphism, and the resulting complementary pairing must be nonsingular over field coefficients. In the smooth real setting, the integration pairing must depend only on cohomology classes, not on the chosen forms.

## Proof artifact

The final artifact should state the cap-product theorem for a closed `R`-orientable `n`-manifold:

```
D_M : H^k(M; R) -> H_{n-k}(M; R),    D_M(alpha) = [M] cap alpha.
```

It should then extract the perfect pairing form over a field:

```
H^k(M; F) x H^{n-k}(M; F) -> F,
(alpha, beta) |-> <alpha cup beta, [M]>.
```

For smooth real manifolds, it should also record the de Rham version:

```
H^k_dR(M) x H^{n-k}_dR(M) -> R,
([omega], [eta]) |-> int_M omega wedge eta.
```

The proof should make clear where each hypothesis is used: orientation gives `[M]`; dimension `n` gives the complementary degree shift; compactness removes compact-support qualifications; absence of boundary removes boundary correction terms.
