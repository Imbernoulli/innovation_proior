# Empirical Risk Minimization

Given i.i.d. training data

```text
S = ((x_1,y_1), ..., (x_n,y_n)),
```

a hypothesis class `F`, and a loss `L`, define true risk and empirical risk by

```text
R(f) = E[L(f(X), Y)]
R_emp(f) = (1/n) sum_{i=1}^n L(f(x_i), y_i).
```

The empirical risk minimizer is

```text
f_hat in argmin_{f in F} R_emp(f).
```

This rule is a sound learning principle exactly when the class is controlled strongly enough that empirical and true risks converge uniformly:

```text
P( sup_{f in F} |R(f) - R_emp(f)| > epsilon ) -> 0
```

for every `epsilon > 0` as `n -> infinity`.

Under this condition, if `f_F` is the best member of `F` under true risk, then

```text
R(f_hat) - R(f_F) <= 2 sup_{f in F} |R(f) - R_emp(f)|.
```

For binary classification, finite VC dimension gives a distribution-free way to make the uniform gap small. In the Vapnik risk-minimization statement, for VC dimension `h >= 1` and confidence parameter `delta in (0,1)`, the simultaneous bound can be written with

```text
c0(n, h, delta) = sqrt((h (log(2n/h) + 1) - log(delta)) / n).
```

With probability at least `1 - delta`, every `f in F` satisfies the true-risk upper bound

```text
R(f) <= R_emp(f) + c0(n, h, delta),
```

so the same bound applies to the empirical minimizer selected after seeing the data.

The method is therefore not "choose the lowest training error" in isolation. It is "choose the lowest empirical risk inside a class whose capacity makes the class-wide empirical-to-true risk gap small."
