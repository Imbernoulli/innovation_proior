# Poincare Duality

Let `M` be a closed `R`-orientable `n`-manifold, and let `[M] in H_n(M; R)` be its fundamental class. The Poincare duality map is

```
D_M : H^k(M; R) -> H_{n-k}(M; R),
D_M(alpha) = [M] cap alpha.
```

For every `k`, this map is an isomorphism. Orientation is the datum that produces the globally coherent fundamental class. Closedness matters because the class lives in ordinary top homology; with boundary it is relative, and without compactness the clean integration statement must use compact support.

## Perfect Pairing Form

If `F` is a field, then the cup product pairing

```
H^k(M; F) x H^{n-k}(M; F) -> F,
(alpha, beta) |-> <alpha cup beta, [M]>
```

is nonsingular. Equivalently, every nonzero `k`-cohomology class has a complementary `(n-k)`-class whose product integrates/evaluates nontrivially over the whole manifold, and conversely.

For integral coefficients, the same statement gives a perfect intersection pairing after quotienting homology by torsion. The torsion linking refinement is a further statement and is not part of this artifact.

## Smooth Real Form

For a closed oriented smooth `n`-manifold,

```
H^k_dR(M) x H^{n-k}_dR(M) -> R,
([omega], [eta]) |-> int_M omega wedge eta
```

is a perfect pairing of finite-dimensional vector spaces.

## Proof Artifact

Define cap product on chains by contracting a `p`-chain with a `q`-cochain and retaining degree `p-q`. The boundary identity

```
partial(c cap phi) = (+/-)(partial c cap phi - c cap delta phi)
```

implies that a cycle capped with a cocycle gives a cycle, and that changing either input by a boundary or coboundary changes the result by a boundary. Hence cap product descends to

```
H_p(M; R) x H^q(M; R) -> H_{p-q}(M; R).
```

An orientation is a locally consistent choice of generators of the local groups `H_n(M, M - {x}; R)`. For a closed oriented manifold these local choices assemble into `[M] in H_n(M; R)`. Capping by this class gives `D_M(alpha) = [M] cap alpha`.

The isomorphism is local and then global. On a coordinate ball the assertion reduces to the one nonzero top local class and ordinary degree-zero cohomology. For a finite good cover, the ordinary and relative Mayer-Vietoris sequences compare the two pieces, their intersection, and their union. Naturality of cap product makes the duality maps commute with these exact sequences up to the standard signs. The five lemma propagates the local isomorphism across unions of good-cover sets, proving that `D_M` is an isomorphism for the whole closed manifold.

The cup-product pairing is the same isomorphism read through evaluation. The cap-cup identity gives

```
<beta, [M] cap alpha> = <alpha cup beta, [M]>
```

with the usual grading convention. Since `D_M` is an isomorphism and, over a field, cohomology is the linear dual of homology in the same degree, this pairing is nonsingular in both variables.

For differential forms, the same theorem becomes integration. If `omega` is closed of degree `k` and `eta` is closed of degree `n-k`, then `omega wedge eta` is a closed top-degree form. Replacing `omega` by `omega + dtheta` changes the integral by an integral of an exact top-degree form; by Stokes' theorem this is zero on a compact manifold with empty boundary. The same argument applies to changing `eta` by an exact form. Thus integration descends to cohomology, and the de Rham pairing is the real-coefficient form of the same duality.
