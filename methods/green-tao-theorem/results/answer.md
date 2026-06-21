# Green-Tao theorem by transference

For every integer `k >= 3`, the primes contain infinitely many arithmetic progressions of length `k`. More strongly, every subset of the primes with positive relative upper density inside the primes contains infinitely many `k`-term arithmetic progressions.

The proof has two linked artifacts.

First, prove a relative Szemeredi theorem. If `nu : Z_N -> R_+` is `k`-pseudorandom and `0 <= f <= nu` with `E f >= delta`, then

`E_{x,r} f(x) f(x+r) ... f(x+(k-1)r) >= c(k,delta) - o(1)`.

Here `k`-pseudorandom means the specific linear forms and correlation conditions needed for `k`-term progression counting. The proof decomposes `f` into a bounded structured part and a Gowers-uniform error; dense Szemeredi handles the structured part, and a relative generalized von Neumann theorem removes the uniform error.

Second, build the prime majorant. Remove small-prime residue obstructions with the W-trick:

`W = product_{p <= w(N)} p`,

and use the modified prime weight supported on primes `Wn+1`. With

`Lambda_R(n) = sum_{d|n,d<=R} mu(d) log(R/d)`,

`R = N^{k^{-1}2^{-k-4}}`, and `epsilon_k = 1/(2^k (k+4)!)`, define on `[epsilon_k N, 2 epsilon_k N]`

`nu(n) = phi(W)/W * Lambda_R(Wn+1)^2 / log R`,

and set `nu(n) = 1` outside that interval. Goldston-Yildirim divisor-sum estimates show that this `nu` has mean `1 + o(1)`, is `k`-pseudorandom, and majorizes a constant multiple of the modified von Mangoldt prime weight.

Applying the relative theorem to that modified prime weight gives a positive count of nondegenerate progressions in the W-tricked primes. The safe interval prevents wraparound, and `n -> Wn+1` converts them into genuine arithmetic progressions of primes.
