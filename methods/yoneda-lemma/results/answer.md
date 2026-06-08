# The Yoneda lemma

The Yoneda lemma says that a natural transformation out of a representable functor is completely determined by one value: the image of the identity arrow at the representing object. For a locally small category `C`, an object `A`, and a functor `F : C → Set`, write `H^A = C(A, −)`. Then

```
Nat(H^A, F) ≅ F(A)
```

by evaluation at `id_A`. This is the mechanism behind the Yoneda embedding: the assignment `A ↦ C(A, −)` is a full and faithful functor `C^op → Set^C`.

## Statement

Let `C` be a locally small category, let `A ∈ C`, and let `F : C → Set`. The function

```
ev : Nat(C(A, −), F) → F(A),     ev(α) = α_A(id_A)
```

is a bijection. Its inverse sends `x ∈ F(A)` to the natural transformation `Ψ(x) : C(A, −) ⇒ F` defined by

```
Ψ(x)_d(f) = F(f)(x),     for f : A → d.
```

The bijection is natural in `F` and natural in the object variable with the contravariance of `A ↦ C(A, −)`.

## Proof

Let `α : C(A, −) ⇒ F`. For any arrow `f : A → d`, naturality of `α` for `f` gives the square

```
C(A, A) --α_A--> F(A)
  f∘− |           | F(f)
C(A, d) --α_d--> F(d).
```

Evaluating this square at `id_A` gives

```
α_d(f) = F(f)(α_A(id_A)).        (∗)
```

Thus `α` is determined by the single element `α_A(id_A) ∈ F(A)`.

Conversely, for `x ∈ F(A)`, define `Ψ(x)_d(f) = F(f)(x)`. This family is natural: if `g : d → e` and `f : A → d`, then

```
Ψ(x)_e(g∘f) = F(g∘f)(x)
```

and

```
F(g)(Ψ(x)_d(f)) = F(g)(F(f)(x)).
```

These are equal because `F(g∘f) = F(g)∘F(f)`, so the naturality square for `Ψ(x)` commutes for every `g : d → e`.

The two inverse checks are immediate from the same formulas:

```
ev(Ψ(x)) = Ψ(x)_A(id_A) = F(id_A)(x) = x,
```

and, for every `α`,

```
Ψ(ev(α))_d(f) = F(f)(α_A(id_A)) = α_d(f)
```

by `(∗)`. Hence `ev` and `Ψ` are inverse bijections.

Naturality in `F`: for `β : F ⇒ G`,

```
ev^G(β·α) = (β·α)_A(id_A) = β_A(α_A(id_A)) = β_A(ev^F(α)).
```

Naturality in the object: for `f : A → A'`, precomposition gives `f^* = (−∘f) : C(A', −) ⇒ C(A, −)`. For `α : C(A, −) ⇒ F`,

```
ev^{A'}(α·f^*) = (α·f^*)_{A'}(id_{A'})
               = α_{A'}(id_{A'}∘f)
               = α_{A'}(f)
               = F(f)(α_A(id_A))
               = F(f)(ev^A(α)).
```

## Yoneda Embedding

Taking `F = H^{A'} = C(A', −)` gives

```
Nat(C(A, −), C(A', −)) ≅ C(A', A).
```

For `h : A' → A`, define `H^h : H^A ⇒ H^{A'}` by precomposition:

```
(H^h)_d(p) = p∘h,     for p : A → d.
```

This is a natural transformation: for `g : d → e` and `p : A → d`, the two paths in the naturality square give

```
C(A', g)((H^h)_d(p)) = g∘(p∘h)
```

and

```
(H^h)_e(C(A, g)(p)) = (g∘p)∘h,
```

which are equal by associativity.

Evaluation recovers `h`:

```
(H^h)_A(id_A) = id_A∘h = h.
```

So `h ↦ H^h` is the inverse of the Yoneda bijection. Since a morphism `A → A'` in `C^op` is exactly a morphism `A' → A` in `C`, the assignment

```
A ↦ C(A, −),     h : A' → A ↦ H^h : H^A ⇒ H^{A'}
```

is a full and faithful functor `C^op → Set^C`. Dually, `A ↦ C(−, A)` is a full and faithful functor `C → Set^{C^op}`.

## Uniqueness

If `H^A ≅ H^{A'}`, then each chosen natural isomorphism `φ : H^A ⇒ H^{A'}` corresponds to a unique arrow `h : A' → A`, and its inverse `ψ : H^{A'} ⇒ H^A` corresponds to a unique arrow `k : A → A'`. Since `ψ∘φ = id_{H^A}` and `φ∘ψ = id_{H^{A'}}`, faithfulness gives `h∘k = id_A` and `k∘h = id_{A'}`. Thus the hom-functor determines the representing object up to isomorphism, and a specified identification of representable functors determines a unique compatible isomorphism of objects.

Equivalently, if a functor `X : C → Set` is represented by `(A, u ∈ X(A))` and also by `(A', u' ∈ X(A'))`, then there is a unique isomorphism compatible with the representations: the arrow `h : A' → A` determined by the induced isomorphism `H^A ≅ H^{A'}` satisfies `u = X(h)(u')`, and its inverse carries `u` back to `u'`.
