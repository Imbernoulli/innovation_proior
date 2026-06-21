# Gelfand Representation

## Banach-algebra form

Let `A` be a complex unital commutative Banach algebra. Its character space is

`Delta(A) = { phi:A->C : phi is a nonzero multiplicative linear functional }`,

with the weak-* topology inherited from `A^*`. For each `a in A`, define the Gelfand transform

`hat a: Delta(A)->C`, `hat a(phi)=phi(a)`.

Then `Delta(A)` is compact Hausdorff, `a -> hat a` is a unital algebra homomorphism `A -> C(Delta(A))`, maximal ideals of `A` are exactly kernels of characters, and

`sigma(a) = hat a(Delta(A))`,

so

`||hat a||_infty = r(a) = lim_n ||a^n||^(1/n)`.

The kernel of the transform is the radical of `A`, the intersection of all maximal ideals.

## Proof

If `phi` is a character, then `ker phi` is maximal because `A/ker phi` is isomorphic to `C`. Conversely, let `M` be a maximal ideal. It is closed: if `1` were in its closure, some `m in M` would satisfy `||1-m||<1`, making `m` invertible by the Neumann series and forcing `M=A`. Thus `A/M` is a complex Banach division algebra. By the Gelfand-Mazur theorem, `A/M` is isomorphic to `C`, and the quotient map gives a character with kernel `M`.

Each character has norm one. If `|lambda|>||a||`, then `lambda 1-a` is invertible by the Neumann series, so `lambda-phi(a)=phi(lambda 1-a)` is nonzero. Hence `|phi(a)|<=||a||`, and `||phi||<=1`; also `phi(1)=1`, so `||phi||=1`. The character space is therefore contained in the weak-* compact unit ball of `A^*`. The multiplicativity and unit equations are weak-* closed pointwise equations, so `Delta(A)` is compact Hausdorff.

For each `a`, the map `phi -> phi(a)` is weak-* continuous, hence `hat a in C(Delta(A))`. Linearity, multiplication, and preservation of the unit follow directly from the defining equations for characters.

To identify the spectrum, first suppose `lambda 1-a` is invertible. Then no character can vanish on it, so `lambda != phi(a)` for every `phi`. Conversely, if `lambda 1-a` is not invertible, the ideal it generates is proper and is contained in a maximal ideal `M=ker phi`; then `0=phi(lambda 1-a)=lambda-phi(a)`. Thus `lambda in sigma(a)` exactly when `lambda=phi(a)` for some character `phi`.

The spectral-radius formula in a Banach algebra gives

`r(a)=max{|lambda| : lambda in sigma(a)}`.

Using the spectrum identity above,

`r(a)=sup_{phi in Delta(A)} |phi(a)|=||hat a||_infty`.

Finally, `hat a=0` exactly when every character vanishes on `a`, equivalently when `a` lies in every maximal ideal. This is the radical.

## Commutative `C^*`-algebra form

Let `A` be a unital commutative `C^*`-algebra. Then the Gelfand transform is an isometric `*`-isomorphism

`A ~= C(Delta(A))`.

Equivalently, every unital commutative `C^*`-algebra is exactly the algebra of continuous complex-valued functions on its compact character space.

## Proof

The Banach-algebra construction already gives a unital homomorphism `A -> C(Delta(A))`. In a commutative `C^*`-algebra every element is normal, and normal elements satisfy `||a||=r(a)`. Therefore

`||hat a||_infty=r(a)=||a||`,

so the transform is isometric and injective. Its range is consequently closed.

If `h=h^*`, then `sigma(h) subset R`; since `phi(h) in sigma(h)`, every character takes real values on self-adjoint elements. Writing

`a = (a+a^*)/2 + i (a-a^*)/(2i)`,

with both summands self-adjoint, gives

`hat(a^*) = overline{hat a}`.

Thus the range is a closed self-adjoint subalgebra of `C(Delta(A))`. It contains constants, and it separates points because distinct characters differ on some element of `A`. By the complex Stone-Weierstrass theorem, the range is all of `C(Delta(A))`. Hence the transform is an isometric `*`-isomorphism.

The algebraic multiplication of `A` has therefore recovered the point space: points are maximal ideals or characters, and the topology is the one seen by all transformed elements.
