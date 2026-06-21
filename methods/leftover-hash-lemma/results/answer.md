# The Leftover Hash Lemma

Let `X` be a random variable with min-entropy at least `k` over a finite domain. Let `H` be chosen uniformly and independently from a 2-universal family of functions into `{0,1}^m`, so for all distinct `x, x'`,

```text
Pr_H[H(x) = H(x')] <= 2^-m.
```

Let `U_m` be uniform over `{0,1}^m` and independent of `H`; all logarithms are base 2. For `0 < epsilon <= 1`, if

```text
m <= k - 2 log(1/epsilon),
```

then

```text
Delta((H, H(X)), (H, U_m)) <= epsilon.
```

Thus `Ext(x, h) = h(x)` is a strong extractor: the chosen hash function may be public, but the hashed value is still statistically close to uniform.

The proof is a collision argument. Put `D` for the number of functions in the family, `K = 2^k`, and `M = 2^m`. For two independent copies, a collision in `(H, H(X))` requires the same seed and then either `X = X'`, bounded by min-entropy, or `X != X'` but `H(X) = H(X')`, bounded by 2-universality. Thus

```text
CP(H, H(X)) <= (1/D)(1/K + 1/M) <= (1 + epsilon^2)/(D M),
```

while `CP(H, U_m) = 1/(D M)`. The excess collision probability gives an L2 bound at most `epsilon / sqrt(D M)`. Cauchy-Schwarz converts this to L1 over `D M` outcomes, and statistical distance is half of L1, giving `Delta <= epsilon/2 <= epsilon`.
