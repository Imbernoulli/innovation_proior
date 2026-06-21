# Matrix Concentration Via Lieb Cumulants

For independent random self-adjoint matrices `X_1, ..., X_n`, define

```text
M_X(theta) = E exp(theta X),
Xi_X(theta) = log E exp(theta X).
```

The scalar Laplace step lifts to matrices as

```text
P{lambda_max(Y) >= t}
  <= inf_{theta > 0} exp(-theta t) E tr exp(theta Y).
```

The noncommutative step is Lieb's theorem. For fixed self-adjoint `H`, the map

```text
A -> tr exp(H + log A)
```

is concave on the positive definite cone. Therefore, with `Y = exp(X)`,

```text
E tr exp(H + X)
  <= tr exp(H + log E exp(X)).
```

Iterating this one-summand inequality over independent summands gives matrix cgf subadditivity:

```text
E tr exp(sum_k theta X_k)
  <= tr exp(sum_k log E exp(theta X_k)).
```

Combining the two pieces yields the master bound

```text
P{lambda_max(sum_k X_k) >= t}
  <= inf_{theta > 0}
     exp(-theta t) tr exp(sum_k log E exp(theta X_k)).
```

If `E exp(theta X_k) <= exp(g(theta) A_k)` and `rho = lambda_max(sum_k A_k)`, then

```text
P{lambda_max(sum_k X_k) >= t}
  <= d inf_{theta > 0} exp(-theta t + g(theta) rho).
```

This is the essential improvement over Golden-Thompson peeling: the scale is `lambda_max(sum_k A_k)`, not `sum_k lambda_max(A_k)`.

Important specializations:

```text
Gaussian/Rademacher:
sigma^2 = ||sum_k A_k^2||
P{lambda_max(sum_k xi_k A_k) >= t}
  <= d exp(-t^2/(2 sigma^2)).

Rectangular Gaussian/Rademacher:
sigma^2 = max{||sum_k B_k B_k^*||, ||sum_k B_k^* B_k||}
P{||sum_k xi_k B_k|| >= t}
  <= (d_1 + d_2) exp(-t^2/(2 sigma^2)).

Bounded centered Bernstein:
E X_k = 0, lambda_max(X_k) <= R,
sigma^2 = ||sum_k E X_k^2||
P{lambda_max(sum_k X_k) >= t}
  <= d exp(-(t^2/2)/(sigma^2 + Rt/3)).
```

For positive semidefinite summands with `0 <= lambda_min(X_k)` and `lambda_max(X_k) <= R`, the same master calculus with the chord bound on `exp(theta x)` gives matrix Chernoff upper and lower tails in terms of eigenvalues of `sum_k E X_k`.
