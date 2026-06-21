# Riemann Mapping Theorem

## Theorem

Let `Omega` be a simply connected domain in `C` with `Omega != C`, and fix
`a in Omega`. Then there is a unique biholomorphic map

```math
f: \Omega \to D=\{w\in C: |w|<1\}
```

such that

```math
f(a)=0,\qquad f'(a)>0.
```

Thus every proper simply connected planar domain has a canonical conformal coordinate after one
basepoint and one tangent direction are fixed.

## Proof

Use four standard facts from one-variable complex analysis: simply connected zero-free holomorphic
functions admit holomorphic logarithms and square roots; Montel's theorem gives compactness for
locally bounded holomorphic families; Hurwitz's theorem preserves univalence in nonconstant compact
limits; Schwarz's lemma identifies disk automorphisms fixing `0`.

Let `S` be the family of all one-to-one holomorphic maps `g: Omega -> D` satisfying
`g(a)=0` and `g'(a)>0`.

First `S` is nonempty. Choose `b notin Omega`. Since `z-b` is zero-free on `Omega` and `Omega` is
simply connected, there is a holomorphic square root `h(z)=sqrt(z-b)`. The map `h` is injective, and
`h(Omega)` cannot contain both `w` and `-w`, because squaring would give the same point of `Omega`.
Since `h(Omega)` contains a disk around `h(a)`, it omits a disk around `-h(a)`. Hence

```math
q(z)={1\over h(z)+h(a)}
```

is bounded and injective on `Omega`. Subtract `q(a)`, divide by a large enough constant to put the
image in `D`, and rotate so the derivative at `a` is positive. This gives an element of `S`.

Set

```math
M=\sup_{g\in S} g'(a).
```

The number `M` is finite by Cauchy's estimate on a small closed disk centered at `a` and contained
in `Omega`, since every `g in S` is bounded by `1`. Take `g_n in S` with `g_n'(a) -> M`. By Montel's
theorem, a subsequence converges uniformly on compact subsets to a holomorphic map `f: Omega -> C`.
Derivative convergence follows from Cauchy's integral formula, so

```math
f(a)=0,\qquad f'(a)=M>0.
```

The limit is nonconstant. By Hurwitz's theorem, the nonconstant compact limit of univalent maps is
univalent. Also `|f|<1` on `Omega`; if `|f|` reached `1` at an interior point, the maximum modulus
principle would make `f` constant. Therefore `f in S` and the extremal derivative is attained.

It remains to prove that `f(Omega)=D`. Suppose not, and choose
`c in D \ f(Omega)`. Since `f(a)=0`, we have `c != 0`. Let

```math
B_c(w)={w-c\over 1-\overline c\,w},
```

the disk automorphism sending `c` to `0`. The function `B_c(f(z))` is zero-free on `Omega`, so simple
connectivity gives a holomorphic square root

```math
s(z)^2=B_c(f(z)).
```

The map `s` is injective: equality of `s` implies equality of `s^2`, hence equality of `B_c(f)`,
hence equality of `f`, hence equality of points. Let `alpha=s(a)`. Since `f(a)=0`,

```math
\alpha^2=B_c(0)=-c,\qquad |\alpha|=\sqrt{|c|}.
```

Move `alpha` to `0` by the disk automorphism

```math
\Phi_\alpha(w)={w-\alpha\over 1-\overline\alpha\,w}
```

and rotate so that the derivative at `a` is positive. The map
`F=e^{i\theta}\Phi_\alpha(s)` is again in `S`.

Differentiate `s^2=B_c(f)` at `a`. Since `B_c'(0)=1-|c|^2`,

```math
2\alpha s'(a)=(1-|c|^2)f'(a).
```

Thus

```math
|F'(a)|
={|s'(a)|\over 1-|\alpha|^2}
={f'(a)(1-|c|^2)\over 2\sqrt{|c|}(1-|c|)}
={f'(a)\over 2}\left(\sqrt{|c|}+{1\over\sqrt{|c|}}\right)
> f'(a)=M.
```

This contradicts the definition of `M`. Hence no such omitted `c` exists, so `f` maps `Omega` onto
`D`. Since `f` is also injective, it is biholomorphic.

For uniqueness, let `f` and `g` be two normalized biholomorphisms. Then
`g o f^{-1}: D -> D` is a disk automorphism fixing `0`. By Schwarz's lemma it is a rotation
`w -> e^{i\theta}w`. The derivative normalization `f'(a)>0` and `g'(a)>0` forces
`e^{i\theta}=1`, so `f=g`.

## Harmonic form

The same coordinate can be read through the Dirichlet/Green-function picture. If `f(a)=0`, then
`log|f(z)|` is harmonic on `Omega \ {a}`, has the singularity `log|z-a|+O(1)` at `a`, and has
boundary value `0` in the disk normalization. Solving this Dirichlet problem supplies the modulus;
simple connectivity supplies the harmonic conjugate; exponentiation supplies the holomorphic map.
The normal-family proof above is the compactness version of the same idea: the extremal derivative
normalization forces the constructed holomorphic coordinate to use the entire disk.
