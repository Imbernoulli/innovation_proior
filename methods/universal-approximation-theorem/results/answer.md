# Universal Approximation Theorem

## Sigmoidal Version

Let `I_n = [0,1]^n`, and let `C(I_n)` carry the uniform norm. Let `sigma: R -> R` be continuous and sigmoidal with distinct finite limits `a_-` at `-infinity` and `a_+` at `+infinity`. Then the finite sums

```text
G(x) = sum_{j=1}^N alpha_j sigma(w_j . x + theta_j)
```

with `alpha_j, theta_j in R` and `w_j in R^n` are dense in `C(I_n)`. Thus, for every `f in C(I_n)` and every `epsilon > 0`, some finite hidden layer satisfies

```text
sup_{x in I_n} |f(x) - G(x)| < epsilon.
```

This is an existence result. It gives no training algorithm, no practical width bound, and no guarantee that an optimizer will find the approximating parameters.

## Proof Skeleton

Cybenko's proof uses the normalized convention `a_- = 0` and `a_+ = 1`. The distinct finite-limit case reduces to it by defining `tau(t) = (sigma(t) - a_-)/(a_+ - a_-)`: constants are in the span by taking `w = 0`, so `span(tau)` and `span(sigma)` have the same closure.

Call an activation `rho` discriminatory if the only finite signed regular Borel measure `mu` on `I_n` satisfying

```text
int_{I_n} rho(w . x + theta) dmu(x) = 0
```

for all `w` and `theta` is `mu = 0`.

If `rho` is discriminatory, its ridge span is dense. Otherwise Hahn-Banach gives a nonzero bounded linear functional vanishing on the closed span, and Riesz represents that functional as integration against a nonzero measure `mu`. That measure would annihilate every ridge function, contradicting discrimination.

Every continuous normalized sigmoidal activation is discriminatory. If `mu` annihilates every `tau` ridge function, then for every `w`, `theta`, `phi`, and large `lambda`, the function

```text
x -> tau(lambda(w . x + theta) + phi)
```

also has zero integral against `mu`. Letting `lambda -> infinity` and using bounded convergence gives

```text
mu({w . x + theta > 0}) + tau(phi) mu({w . x + theta = 0}) = 0.
```

Choosing two values of `phi` with different `tau(phi)` values forces both the hyperplane mass and the open-half-space mass to be zero.

Fix `w` and push `mu` forward by `T_w(x) = w . x`; call the signed measure on `R` by `nu_w`. Half-lines and points have zero `nu_w`-mass, so intervals have zero mass, and the monotone-class theorem gives `nu_w = 0`. Hence

```text
int_{I_n} exp(i s w . x) dmu(x) = 0
```

for every real `s`. Since `s w` ranges over all frequencies in `R^n`, the Fourier transform of `mu` is identically zero. Uniqueness of Fourier transforms of finite compactly supported measures gives `mu = 0`.

## Decision Regions

For a finite measurable partition of `I_n`, a continuous sigmoidal single-hidden-layer network can approximate the associated decision function outside a set of arbitrarily small Lebesgue measure. The proof uses Lusin's theorem to replace the measurable decision function by a continuous function on a large-measure set, then applies the uniform approximation result.

For a closed decision region `D`, use the distance function to build a continuous ramp that is `1` on `D` and `0` outside an `epsilon`-neighborhood of `D`. If the network approximates that ramp within less than `1/2`, thresholding its output classifies all points in `D` and all points outside the `epsilon`-neighborhood correctly; only the boundary band is undecided.

## Activation Frontier

For locally bounded piecewise continuous activations whose discontinuities have negligible closure, the thresholded span

```text
span{ sigma(w . x + theta) : w in R^n, theta in R }
```

is dense in `C(K)` for every compact `K` contained in `R^n` if and only if `sigma` is not an algebraic polynomial almost everywhere.

Polynomial activations fail because affine substitution and finite summation stay inside a finite-degree polynomial space. Non-polynomial activations succeed by reducing multivariate approximation to ridge functions and then proving one-dimensional density for shifted and scaled copies of the activation, with smoothing and distribution theory handling the nonsmooth case.

The threshold is essential. Without shifts, even a non-polynomial activation such as `sin(w x)` spans only odd functions on a symmetric interval and cannot approximate `cos(x)`.
