# Central Limit Theorem via Characteristic Functions

Let `X_1, X_2, ...` be iid real random variables with `E X_1 = mu` and `Var(X_1) = sigma^2`, where `0 < sigma^2 < infinity`. Then

`(X_1 + ... + X_n - n mu) / (sigma sqrt(n)) => N(0,1)`.

The proof centers and rescales, so assume `E X_1 = 0` and `Var(X_1) = 1`. Let `phi(t) = E exp(itX_1)`. A finite second moment gives the local expansion

`phi(t) = 1 - t^2/2 + o(t^2)` as `t -> 0`.

The characteristic function of `S_n / sqrt(n)` is

`[phi(t / sqrt(n))]^n`.

Using the local expansion,

`phi(t / sqrt(n)) = 1 - t^2/(2n) + o(1/n)`,

so

`[phi(t / sqrt(n))]^n -> exp(-t^2/2)`.

Since `exp(-t^2/2)` is the characteristic function of `N(0,1)`, Levy's continuity theorem gives convergence in distribution.

## Lindeberg-Feller Form

The same method proves the non-identically distributed triangular-array theorem. Let `X_{n,1}, ..., X_{n,k_n}` be independent, centered variables with variances `sigma_{n,j}^2` and

`sum_j sigma_{n,j}^2 -> 1`.

Assume that for every `epsilon > 0`,

`sum_j E[X_{n,j}^2 1{|X_{n,j}| > epsilon}] -> 0`.

Then

`sum_j X_{n,j} => N(0,1)`.

For each fixed `t`, compare the true factor `E exp(itX_{n,j})` with `1 - t^2 sigma_{n,j}^2/2`. On the event `|X_{n,j}| <= epsilon`, Taylor's theorem controls the remainder by a small multiple of `X_{n,j}^2`. On the complement, the Lindeberg condition controls the total tail variance. Therefore

`sum_j |E exp(itX_{n,j}) - (1 - t^2 sigma_{n,j}^2/2)| -> 0`.

Independence turns sums into products of characteristic functions, so the true product has the same limit as

`prod_j (1 - t^2 sigma_{n,j}^2/2) -> exp(-t^2/2)`.

Levy's theorem again gives convergence to the standard normal law.

Lyapunov's higher-moment condition is a sufficient condition for Lindeberg's condition. If, after unit variance normalization, `sum_j E|X_{n,j}|^{2+delta} -> 0` for some `delta > 0`, then the Lindeberg tail sum is bounded by `epsilon^{-delta} sum_j E|X_{n,j}|^{2+delta}`, so it vanishes.

Historically, de Moivre's binomial approximation identifies the square-root scale and the bell-shaped limit in a special model. Lindeberg's 1922 work identifies the variance-tail criterion needed for unequal summands. Characteristic functions give the compact modern proof because they convert independent sums into products and isolate the universal quadratic variance term.
