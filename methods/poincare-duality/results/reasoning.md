I start with the simplest obstruction: homology gives me a list of vector spaces or modules, one for each dimension, and a general space has no reason to identify one entry in that list with another. A circle has `H_0` and `H_1`, a wedge can have whatever ranks I build into it, and cohomology in degree `k` only tells me how to test `k`-cycles. If I want a `k`-dimensional hole to have a complementary detector, I need extra structure that is not present in an arbitrary space.

An `n`-manifold has exactly that kind of extra structure locally. Around every point it looks like `R^n`, and the local homology group `H_n(M, M - {x})` is a copy of the coefficient ring in the orientable case. A choice of generator at one point is a sign choice; moving it through a coordinate ball transports it to nearby points. If I can make these choices consistently around every loop, then the manifold is oriented. So orientation is not decorative. It is the data that lets all little local top-dimensional pieces point the same way.

If those local choices are coherent and the manifold is closed, I should be able to add all the local pieces into one global top-dimensional class. That is the fundamental class `[M]`. It is not merely a class in the highest degree; it is the class whose image near each point is the chosen local generator. This is the first place the closed hypothesis is doing work. If the manifold has a boundary, the top class naturally lives relative to the boundary. If the manifold is noncompact, an ordinary finite cycle cannot cover the whole manifold. For the clean statement I want, `[M]` has to live in ordinary `H_n(M; R)`.

Now I need to turn `[M]` into complementary tests. Cup product alone almost gets there. If `alpha` has degree `k` and `beta` has degree `n-k`, then `alpha cup beta` has degree `n`, and I can evaluate it on `[M]`. This gives a number. But I should be careful: a number-valued pairing is the end of the story, not the mechanism. I still need to see why every nonzero class has a partner, and I need a map that actually shifts degree from `k` to `n-k`.

The operation with the right shape is cap product. A `p`-chain contains `p` directions. A `q`-cochain can test `q` of those directions and leave the remaining `p-q` directions behind. On a singular simplex this is literally a front-face evaluation followed by keeping the back face. So cap product has the degree bookkeeping I need:

```
H_p(M; R) x H^q(M; R) -> H_{p-q}(M; R).
```

If I feed it the one global `n`-class, it becomes

```
alpha |-> [M] cap alpha in H_{n-k}(M; R).
```

This is already the right kind of object. A degree-`k` cohomology test becomes a complementary `(n-k)`-dimensional homology class. The manifold is using its top-dimensional class as a converter between tests and holes.

But why should this converter be perfect? I can test it locally first. In a coordinate ball, there is only one real top-dimensional local class and the cohomology is trivial except in degree zero. The statement reduces to the obvious fact that the local top class pairs with constants and nothing mysterious exists in other degrees. Locally the theorem is forced.

The problem is gluing. If I prove the statement on two open pieces and on their intersection, the Mayer-Vietoris sequences compare the pieces with the union. Cap product is natural enough to make the maps between the sequences commute up to the conventional signs. Exactness then lets the five lemma carry the isomorphism from the pieces and their overlap to the union. A finite good cover reduces the manifold to coordinate balls. The proof is not a recombination of tricks; it is local certainty propagated through exact sequences. The local generator is coherent, cap product respects restriction, and exactness leaves no room for a hidden kernel or cokernel.

So the degree-shifting map should be stated directly:

```
D_M : H^k(M; R) -> H_{n-k}(M; R),
D_M(alpha) = [M] cap alpha.
```

For a closed `R`-orientable `n`-manifold, `D_M` is an isomorphism in every degree. This is the algebraic core. The same sentence also explains the geometry: the manifold's global orientation class turns a `k`-dimensional measuring device into an `(n-k)`-dimensional object, and every complementary object arises uniquely this way.

Now I want the "holes pair with tests" form. Over a field `F`, cohomology is the linear dual of homology in the same degree. If `D_M` identifies `H^k(M; F)` with `H_{n-k}(M; F)`, then dualizing tells me that `H^k` and `H^{n-k}` should pair perfectly. The cup product formula gives exactly the pairing:

```
(alpha, beta) |-> <alpha cup beta, [M]>.
```

The cap-cup identity explains why this is not an extra structure. Evaluating `beta` on `[M] cap alpha` is the same as evaluating `alpha cup beta` on `[M]`, up to the standard order convention. Thus nonsingularity of the cup pairing is just the isomorphism `D_M` read through ordinary evaluation. If a class `alpha` has zero product against every complementary `beta`, then `[M] cap alpha` evaluates to zero against every same-degree cohomology test, hence is zero over a field; since `D_M` is an isomorphism, `alpha` was zero. The same argument works in the other variable. That is the perfect pairing.

The differential-form version makes the same structure visible without singular chains. A closed `k`-form `omega` and a closed `(n-k)`-form `eta` wedge to a closed top-degree form. On an oriented compact manifold I can integrate:

```
([omega], [eta]) |-> int_M omega wedge eta.
```
