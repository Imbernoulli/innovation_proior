# Algorithmic Stability

Let `A` be a symmetric learning algorithm, `S = (z_1, ..., z_m)` an i.i.d. training sample, and `ell(A(S), z)` a loss bounded by `0 <= ell(A(S), z) <= M`. Define

```text
R(A,S) = E_z ell(A(S), z)
R_emp(A,S) = (1/m) sum_i ell(A(S), z_i)
R_loo(A,S) = (1/m) sum_i ell(A(S \ i), z_i)
```

where `S \ i` removes the `i`th example.

The algorithm has uniform stability `beta` with respect to `ell` if

```text
for every sample S, every index i, and every test point z,
|ell(A(S), z) - ell(A(S \ i), z)| <= beta.
```

This deletion condition also controls replacement: if `S^i` replaces one point, then

```text
|ell(A(S), z) - ell(A(S^i), z)| <= 2 beta.
```

The generalization theorem gives the following separate high-probability bounds:

```text
With probability at least 1 - delta,

R(A,S) <= R_emp(A,S)
          + 2 beta
          + (4 m beta + M) sqrt(log(1/delta) / (2m)).
```

The leave-one-out form is:

```text
R(A,S) <= R_loo(A,S)
          + beta
          + (4 m beta + M) sqrt(log(1/delta) / (2m)).
```

Thus the proof obligation for a learning method is reduced to proving a small `beta`. If `beta = O(1/m)`, the stability terms vanish with sample size and the empirical or leave-one-out estimate becomes a valid risk certificate.

For regularized learning in an RKHS, this obligation can be discharged directly. If `k(x,x) <= kappa^2`, the loss is `sigma`-admissible, and

```text
A(S) = argmin_g (1/m) sum_i ell(g, z_i) + lambda ||g||_k^2,
```

then the algorithm has uniform stability

```text
beta <= sigma^2 kappa^2 / (2 lambda m).
```

This is the stability certificate; applying the high-probability theorem still requires the separate bounded-loss condition above. The method is therefore: analyze the algorithm's one-sample loss sensitivity, prove a uniform-stability rate, and plug that rate into the generalization bound.
