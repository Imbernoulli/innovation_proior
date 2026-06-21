# Atiyah-Singer Index Theorem

## Theorem

Let `X` be a compact smooth manifold, and let

```text
P : C^\infty(X; E^0) -> C^\infty(X; E^1)
```

be an elliptic differential or pseudodifferential operator between complex vector bundles. Its principal symbol is invertible away from the zero section of `T^*X`, so it defines a compactly supported K-theory class

```text
[sigma(P)] in K^0_c(T^*X).
```

The analytic index is the Fredholm index

```text
a-ind([sigma(P)]) = ind(P) = dim ker(P) - dim coker(P).
```

The topological index is the K-theory pushforward of the same symbol class to a point,

```text
t-ind([sigma(P)]) in K^0(pt) = Z.
```

The theorem asserts

```text
ind(P) = a-ind([sigma(P)]) = t-ind([sigma(P)]).
```

Equivalently, in cohomological form, the integer is obtained by applying the Chern character to the symbol class, using the Thom isomorphism to return to `X`, multiplying by the Todd class of the complexified tangent bundle, and pairing with the fundamental class:

```text
ind(P) = < Todd(TX tensor C) cup phi^{-1}(ch [sigma(P)]), [X] >.
```

The exact notation depends on the orientation and Thom-isomorphism conventions, but the invariant content is fixed: the analytic Fredholm index is determined by the K-theory class of the elliptic symbol and the topology of the manifold.

## Proof Artifact

1. Ellipticity gives a Fredholm problem.

   Complete the spaces of smooth sections in Sobolev norms. Elliptic regularity and parametrix theory imply that `P` has closed range and finite-dimensional kernel and cokernel. Thus

   ```text
   ind(P) = dim ker(P) - dim coker(P)
   ```

   is defined. The same regularity identifies the Sobolev kernels with smooth kernels.

2. The index only depends on the symbol class.

   Lower-order changes are compact relative to the principal elliptic part, and compact perturbations do not change Fredholm index. Continuous deformations through elliptic operators produce continuous paths through Fredholm operators, and the Fredholm index is locally constant. Therefore the analytic index depends only on the homotopy class of the principal symbol.

   Since `sigma(P)` is an isomorphism off the zero section, it is exactly the data of a compactly supported K-theory class on `T^*X`: two pulled-back bundles and an isomorphism outside a compact set. This yields a well-defined homomorphism

   ```text
   a-ind : K^0_c(T^*X) -> Z.
   ```

3. K-theory supplies the matching topological map.

   The compactly supported class `[sigma(P)]` can be pushed forward in K-theory. The construction uses the Thom isomorphism and Bott periodicity, or equivalently an embedding of `X` into a sphere followed by symbol-level pushforward and reduction to the model class over a point. This defines

   ```text
   t-ind : K^0_c(T^*X) -> K^0(pt) = Z.
   ```

4. The two maps satisfy the same characterizing properties.

   Both assignments are additive, homotopy invariant, functorial under diffeomorphism, compatible with excision, compatible with the product/Thom operations used in the pushforward, and normalized on the point model. The original K-theoretic proof establishes that these properties characterize the index map uniquely. The analytic index satisfies them by Fredholm stability, elliptic regularity, and pseudodifferential symbol calculus; the topological index satisfies them by construction.

5. Hence the maps are equal.

   Uniqueness forces

   ```text
   a-ind = t-ind : K^0_c(T^*X) -> Z.
   ```

   Evaluating on the symbol class of `P` gives

   ```text
   dim ker(P) - dim coker(P) = t-ind([sigma(P)]).
   ```

This is the core insight: a finite-dimensional analytic defect of an elliptic PDE is secretly a topological invariant of the symbol bundle over the cotangent bundle and the manifold that carries it.

