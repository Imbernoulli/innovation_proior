# Cauchy Integral Formula

## Theorem

Let `Omega subset C` be a domain, let `gamma` be a positively oriented piecewise `C^1` simple closed contour whose interior and image lie in `Omega`, and let `f` be holomorphic on `Omega`. If `a` lies inside `gamma`, then

```math
f(a) = {1 \over 2\pi i}\int_\gamma {f(z) \over z-a}\,dz.
```

Moreover `f` has derivatives of every order at `a`, and for every integer `n >= 0`,

```math
f^{(n)}(a) = {n! \over 2\pi i}\int_\gamma {f(z) \over (z-a)^{n+1}}\,dz.
```

Equivalently, the boundary values of a holomorphic function on any contour enclosing `a` determine the value and all derivatives at `a`.

## Proof

Define

```math
g(z)=
\begin{cases}
{f(z)-f(a) \over z-a}, & z\ne a,\\
f'(a), & z=a.
\end{cases}
```

Then `g` is continuous at `a` and holomorphic on `Omega \ {a}`. To see that its contour integral is zero, remove a small positively oriented circle `C_r` centered at `a`, lying inside `gamma`. On the punctured region between `gamma` and `C_r`, the function `g` is holomorphic, so Cauchy's theorem gives

```math
\int_\gamma g(z)\,dz-\int_{C_r}g(z)\,dz=0.
```

The integral over `C_r` tends to zero as `r -> 0`, because `g` is bounded near `a` and the length of `C_r` is `2*pi*r`. Hence

```math
\int_\gamma g(z)\,dz=0.
```

For `z \ne a`,

```math
{f(z) \over z-a} = {f(a) \over z-a} + g(z).
```

Integrating around `gamma`,

```math
\int_\gamma {f(z) \over z-a}\,dz
=
f(a)\int_\gamma {dz \over z-a}
+
\int_\gamma g(z)\,dz.
```

The last integral is zero. Since `gamma` winds once positively around `a`,

```math
\int_\gamma {dz \over z-a}=2\pi i.
```

Hence

```math
\int_\gamma {f(z) \over z-a}\,dz = 2\pi i\,f(a),
```

which proves the value formula.

This formula is independent of the particular enclosing contour: if `gamma_0` and `gamma_1` are homologous in `Omega \ {a}` and both wind once around `a`, then the difference of their integrals is the integral of `f(z)/(z-a)` over the boundary of a region where that integrand is holomorphic. Cauchy's theorem makes the difference zero. Thus a large boundary contour may be shrunk to a small circle around `a` without changing the integral.

For derivatives, first prove the case `n=1`. Let `b` be near `a` and still inside `gamma`. Applying the value formula at both points gives

```math
f(b)-f(a)
=
{1 \over 2\pi i}
\int_\gamma f(z)
\left({1 \over z-b}-{1 \over z-a}\right)\,dz.
```

Since

```math
{1 \over z-b}-{1 \over z-a}
=
{b-a \over (z-b)(z-a)},
```

division by `b-a` yields

```math
{f(b)-f(a) \over b-a}
=
{1 \over 2\pi i}
\int_\gamma {f(z) \over (z-b)(z-a)}\,dz.
```

As `b -> a`, the integrand converges uniformly on `gamma` to `f(z)/(z-a)^2`, because `gamma` has positive distance from `a`. Therefore

```math
f'(a) =
{1 \over 2\pi i}
\int_\gamma {f(z) \over (z-a)^2}\,dz.
```

Now repeat the same differentiation under the integral sign. The contour remains a positive distance from `a`, so the convergence is uniform on the contour at each step. Since

```math
{d \over da}(z-a)^{-(n+1)}=(n+1)(z-a)^{-(n+2)},
```

an induction from the value formula gives

```math
f^{(n)}(a)
=
{n! \over 2\pi i}
\int_\gamma {f(z) \over (z-a)^{n+1}}\,dz
```

for all `n >= 0`.

The proof also shows the central rigidity. After subtracting `f(a)`, the apparent pole at `a` disappears, so Cauchy's theorem kills the remainder. The only surviving contribution is the singular kernel `1/(z-a)`, whose loop integral is `2*pi*i`. A local differentiability condition has become a global reconstruction principle: the boundary contour determines the interior value and all derivatives.
