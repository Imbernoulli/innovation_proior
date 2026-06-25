I start with the integer that analysis hands me. An elliptic operator on a compact manifold is not just a differential expression; once I complete the spaces, it becomes Fredholm. That means there are only finitely many smooth solutions of `Pu=0`, only finitely many obstructions to solving `Pu=f`, and the difference

```text
dim ker(P) - dim coker(P)
```

does not jump under small Fredholm deformations. This already feels odd in the right way. A kernel is made of solutions to a PDE, and a cokernel is made of missed right-hand sides, but the difference is stable under perturbations that can radically change the individual solution spaces. I should not try to compute the two dimensions separately. The stable object is the difference.

The next thing to isolate is what part of the operator can control a stable Fredholm invariant. Lower-order terms are too flexible. Elliptic theory tells me they are compact relative to the highest-order behavior, and Fredholm index ignores compact perturbations. So the analytic integer is already pushing me away from the full local formula for `P` and toward its leading term. In local coordinates a differential operator of order `m` has a top part that looks like

```text
a_{i_1...i_m}(x) partial_{i_1} ... partial_{i_m},
```

and that top part transforms invariantly as a function on cotangent vectors. For each nonzero covector `xi in T_x^*X`, the symbol is a linear map

```text
sigma(P)(x, xi) : E^0_x -> E^1_x.
```

Ellipticity says precisely that this linear map is invertible whenever `xi != 0`. That is the first clean replacement for the PDE: over the cotangent bundle I have two pulled-back vector bundles, and away from the zero section I have an isomorphism between them.

If I keep the symbol as a pointwise formula, I have not yet gained enough. I need the right equivalence relation. Homotopies of elliptic symbols should not change the index, because they lift to continuous paths of Fredholm operators. Direct sums of operators add indices. Stabilizing by adding a trivial invertible piece should add nothing. These are exactly K-theory moves. A compactly supported K-class on `T^*X` is represented by two vector bundles and an isomorphism outside a compact set. The elliptic symbol is not just an input to a topological construction; it already has the form of a K-theory class.

So the analytic side becomes a homomorphism

```text
a-ind : K^0_c(T^*X) -> Z,
```

where `a-ind([sigma(P)])` is defined by choosing an elliptic operator with that symbol and taking its Fredholm index. I have to be careful: this is well-defined only because elliptic pseudodifferential operators are flexible enough that symbol classes can be realized, and because changing the representative without changing the symbol class changes the operator through index-preserving moves. That flexibility is why the pseudodifferential calculus matters. Differential operators alone are too rigid for the topology to be visible as a full K-theory group.

Now I need a topological homomorphism from the same group to `Z`. The symbol lives on `T^*X`, not on `X`, but K-theory has exactly the mechanism for pushing compactly supported classes forward. Embed, use Thom isomorphism and Bott periodicity, reduce to a class over a Euclidean space or a sphere, and finally collapse to a point. In the most compressed K-theoretic language, the topological index is the pushforward

```text
t-ind : K^0_c(T^*X) -> K^0(pt) = Z.
```

This is at least the right type of object. It is built from the same data the operator gives me, the symbol class, and it lands in the integers. If I want a cohomological formula, I can apply the Chern character and Thom isomorphism. Then the pushforward becomes integration against characteristic classes, giving a candidate form of the answer

```text
ind(P) = < Todd(TX tensor C) cup phi^{-1}(ch sigma(P)), [X] >,
```

up to the conventional placement of the Thom isomorphism and the chosen orientation data.

Before I trust this, I should pin the factors down on a case where I already know both sides as numbers. The honest worry is that I have written a formula with the right shape but the wrong characteristic class or a wrong sign, and that kind of error is invisible at the level of "this is well-typed." A Riemann surface is the smallest place where the cohomological formula has a nontrivial term and the analytic index is classically known, so let me grind it there.

Take `X` a compact Riemann surface of genus `g`, and let `L` be a holomorphic line bundle of degree `d`. The relevant elliptic operator is the Dolbeault operator `\bar\partial_L : C^\infty(X; L) -> C^\infty(X; L tensor \bar{T}^*X)`, whose kernel is `H^0(X; L)` and whose cokernel is `H^1(X; L)`. So the analytic index is

```text
ind = dim H^0(X; L) - dim H^1(X; L),
```

and classical Riemann-Roch already tells me this equals `d - g + 1`. That is the number the cohomological formula has to reproduce.

Now I compute the right-hand side directly. On a complex curve `dim_C X = 1`, so only the top two graded pieces survive integration. For a line bundle, `ch(L) = 1 + c_1(L)`. For the Todd class of the holomorphic tangent line bundle, `Td(TX) = 1 + (1/2) c_1(TX) + ...`, and on a curve everything above degree `2` integrates to zero. Multiplying,

```text
ch(L) cup Td(TX) = (1 + c_1(L)) (1 + (1/2) c_1(TX)) = 1 + c_1(L) + (1/2) c_1(TX) + (deg 4 junk).
```

The degree-`2` part is `c_1(L) + (1/2) c_1(TX)`. Pairing with `[X]`: the first term gives `< c_1(L), [X] > = deg(L) = d`. The second gives `(1/2) < c_1(TX), [X] > = (1/2) chi(X)`, and the Euler characteristic of a genus-`g` surface is `2 - 2g`, so that is `(1/2)(2 - 2g) = 1 - g`. Adding,

```text
< ch(L) cup Td(TX), [X] > = d + (1 - g) = d - g + 1.
```

That is exactly `dim H^0 - dim H^1`. The two sides agree, and they agree as integers for every `d` and `g`, which is a much sharper test than agreement of types. The arithmetic also tells me which Todd class is doing the work in this complex-geometric model: it is the Todd class of the holomorphic tangent bundle, paired against `ch(L)`, with no separate Thom factor surviving because `X` already carries the complex orientation. So when I write the general formula with `Todd(TX tensor C)` and an explicit `phi^{-1}`, those extra ingredients are the price of handling a real manifold without a given complex structure; on a complex manifold they collapse back to this calculation. Good — the formula is not just well-typed, it is calibrated on a known case.

It is worth a second small sanity check at the trivial end. Take `L` trivial, so `d = 0`: the formula gives `1 - g`. On the other side `H^0` of the trivial bundle is the constants, dimension `1`, and `H^1` is `g`-dimensional by Hodge theory on a genus-`g` surface, so `1 - g`. Consistent again, and it confirms the sign of the `g` term rather than leaving it to convention.

Having a calibrated formula, I return to the structural question: why should the analytic and topological homomorphisms be equal as maps on all of `K^0_c(T^*X)`, not just on the few classes I can compute by hand. Computing each individual index is hopeless in general, so I want to characterize the maps instead. List the properties any reasonable index map should have: functorial under diffeomorphisms, additive under direct sum, invariant under homotopy, compatible with excision, multiplicative for the product constructions used in the Thom picture, and normalized on the model class over a point. Suppose for the moment that these properties pin down a homomorphism `K^0_c(T^*X) -> Z` uniquely. Then I do not have to match the two maps class by class; I only have to check that each separately obeys the list.

The topological index is built to satisfy them, so that half is bookkeeping. The analytic index is the side I should actually inspect. Homotopy invariance holds because it is a Fredholm index and the index is locally constant on Fredholm operators. Compact-perturbation invariance holds for the same reason, which is also what let me drop the lower-order terms at the start. Finite-dimensional kernels and cokernels come from elliptic regularity. Additivity under direct sums is immediate from `ker` and `coker` of a block-diagonal operator. The harder entries are excision and the product/multiplicative behavior, and those are exactly the steps that need the full pseudodifferential calculus rather than just differential operators — which is the same flexibility I already had to invoke to make `a-ind` well-defined on symbol classes. So the analytic index does pass the same checks, granted the uniqueness assumption. That assumption is the real content I am leaning on; the original axiomatic proof is precisely the work of showing the property list determines the map, and I am treating that as the load-bearing input rather than something I can re-derive in a paragraph. With it,

```text
a-ind = t-ind : K^0_c(T^*X) -> Z,
```

and evaluating on the symbol class of `P` recovers the index of `P`.

This also explains why K-theory was the right ambient setting and not cohomology. The symbol is linear algebra in families: a bundle map, invertible off the zero section. K-theory is the topology of exactly that data, so the analytic and topological maps live on the same group with no translation. Characteristic classes are a powerful translation layer — they are what made the Riemann surface check computable — but they are downstream of the symbol class, not the first object the operator hands me.

There is another route that keeps the PDE more visible, and it is worth seeing how far it goes on its own. If `P` is elliptic, then `P^*P` and `PP^*` are nonnegative elliptic operators. Their heat operators are trace class for positive time, and the difference

```text
Tr exp(-t P^*P) - Tr exp(-t PP^*)
```

is independent of `t`: differentiating in `t` and using `P^*P` and `PP^*` having the same nonzero spectrum kills the derivative. As `t -> infinity`, every positive eigenvalue is suppressed and only the zero eigenspaces remain, so the expression becomes `dim ker(P) - dim ker(P^*)`, the Fredholm index. As `t -> 0`, the heat kernels localize near the diagonal, and the same global integer is rewritten as the integral of a local density built from the symbol and its derivatives. That much is general and exact.

But this route does not, by itself, finish the general theorem, and I want to be honest about where it stalls. For an arbitrary elliptic `P`, the small-`t` heat coefficient is a genuine but messy local expression involving many derivatives of the total symbol; there is no reason it should organize into a characteristic form, and it generally does not in any recognizable closed shape. The collapse to `A-hat` times a Chern character is special to Dirac-type operators, where the Clifford algebra forces the would-be local terms to cancel. So the heat method cleanly proves the Dirac case and exhibits the local density, but it does not on its own show that the answer depends only on the symbol class for general `P` — that symbol-class dependence is what the K-theory argument supplies. The two proofs are complementary: heat kernels make the integer literally an integral of local curvature in the cases where the algebra cooperates, and K-theory explains why a symbol-class invariant exists at all.

With both views in hand the classical examples stop looking like separate theorems. For the de Rham complex, `d + d^*` packages the cohomology into a Fredholm problem and the index is the Euler characteristic; the cohomological formula is Gauss-Bonnet-Chern, and on `S^2` it returns `chi = 2`, matching the genus-`0`, `d = 0` count above through `1 - g = 1` per orientation pairing. For the Dolbeault complex it returns Hirzebruch-Riemann-Roch, which is the very calculation I just verified. For the signature operator it returns the signature theorem. Each is the same mechanism evaluated on a different symbol: the analytic integer sees only the stable symbol class, and K-theory knows how to push that class to a point.

So the theorem, stated in invariant form. Given a compact smooth manifold `X` and an elliptic operator `P : C^\infty(X; E^0) -> C^\infty(X; E^1)`, the principal symbol defines a class `[sigma(P)] in K^0_c(T^*X)`. The analytic construction sends this class to the Fredholm index of `P`; the topological construction sends it to the K-theory pushforward to a point; these two homomorphisms are equal. In cohomological notation, after the Chern character and the Thom isomorphism, the same integer is the pairing of the transformed symbol class and the Todd class of the complexified tangent bundle with the fundamental class — the formula I calibrated on a Riemann surface, where it reduced correctly to `d - g + 1`. The finite-dimensional defect of an elliptic PDE is topological data carried by the symbol bundle and the manifold that carries it.
