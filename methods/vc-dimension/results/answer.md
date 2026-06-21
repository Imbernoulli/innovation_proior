# VC Dimension and Uniform Convergence

For a class `S` of measurable events, a sample `X_l=(x_1,...,x_l)`, and empirical frequency `nu_A^(l)`, define the simultaneous deviation

```text
pi_l = sup_{A in S} |nu_A^(l) - P(A)|.
```

On a finite sample, the effective size of `S` is not its parameter count or cardinality. It is the number of distinct labelings it induces:

```text
Delta^S(X_l) = |{(1_A(x_1),...,1_A(x_l)) : A in S}|,
m^S(l) = max_{X_l} Delta^S(X_l).
```

A finite set is shattered when every binary labeling of it is induced by some event in `S`. The modern VC dimension is the largest shattered size:

```text
VC(S) = max {d : some d-point set is shattered by S}.
```

Vapnik and Chervonenkis prove the distribution-free uniform convergence bound

```text
P(pi_l > epsilon) <= 4 m^S(2l) exp(-epsilon^2 l / 8),
for l >= 2 / epsilon^2.
```

Their combinatorial lemma uses the recurrence

```text
Phi(n,r) = Phi(n,r-1) + Phi(n-1,r-1),
Phi(0,r)=1, Phi(n,0)=1.
```

If full shattering first fails at size `n`, then no sample can reach the `Phi(n,r)` threshold, and

```text
m^S(r) <= r^n + 1.
```

Thus finite capacity makes `m^S` polynomial, and the exponential factor forces `pi_l -> 0` in probability for every distribution. The same tail bound is summable, so the convergence is almost sure. A sufficient sample size from the original theorem is

```text
l >= (16 / epsilon^2) * (n log(16 n / epsilon^2) - log(eta / 4)).
```

In modern sharpened Sauer form, if `VC(S)=d` and `l >= d >= 1`, then

```text
m^S(l) <= sum_{k=0}^d binom(l,k) <= (e l / d)^d.
```

For `d=0`, the class induces only one labeling on every nonempty sample, so `m^S(l)=1`.

So the controlling quantity is the largest set on which the class can realize all labelings. This is the key distinction: statistical learnability is measured by finite-sample shattering capacity, not by the number of parameters, the continuum size of the class, or whether a rule can fit one observed dataset.

For a fixed distribution, the necessary and sufficient condition uses entropy

```text
H^S(l) = E log_2 Delta^S(X_l).
```

Uniform convergence in probability over `S` holds exactly when

```text
H^S(l) / l -> 0.
```

Examples: rays on the line have `m^S(l)=l+1`, recovering the Glivenko-Cantelli theorem; affine halfspaces in `R^d` have VC dimension `d+1`; all subsets of an interval shatter every finite set and do not satisfy the uniform law.
