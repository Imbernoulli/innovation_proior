I have `S` identical candies and `k` distinguishable children, each child taking `0..c` inclusive, and I count the integer solutions of `x_1 + ... + x_k = S` modulo a prime `M`. Input is `k c S M` on one stdin line; I print one integer. The scales split oddly: `k, S <= 10^6` but `c <= 10^9`, so the cap can dwarf the total (and may never bite), yet `k` and `S` are what a factorial table has to span. And `M` is prime with `M > S + k` — that promise is load-bearing, not decoration. The largest binomial top I ever form is `S + k - 1`, which is strictly below `M`, so `(S+k-1)!` carries no factor of `M` and its modular inverse exists; every factorial I invert is coprime to `M`. Products stay below `(10^9)^2 ~ 10^{18}`, inside `long long`, as long as I reduce after each multiply.

The obvious formulation is a bounded-composition DP: carry `dp[s]` = ways to reach partial sum `s` over the children seen so far, each adding a `0..c` window, transition by prefix sums. That is `O(k*S)`, up to `10^{12}` operations against a 1-second limit — far too slow to ship, though it makes a clean brute-force oracle on small inputs. The route that fits is inclusion-exclusion over the cap. The unbounded count of `k` nonnegative parts summing to `T` is stars-and-bars `C(T + k - 1, k - 1)`. A child violates its cap exactly when it holds `>= c + 1`; substituting `x_i = (c+1) + x_i'` removes that child's cap and drops the total by `c+1`. Inclusion-exclusion over which `j` children are forced to overflow — `C(k, j)` ways to pick them, and the leftover counted unbounded — gives

```
answer = sum_j (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1),
```

with at most `k+1` terms: `O(S + k)` precompute plus `O(k)` summation.

The only subtle part is where the sum stops. A term is real exactly while the leftover `S - j*(c+1)` is `>= 0` (a negative leftover has no distributions) and `j <= k`. I hand-run the sample `k=4, c=5, S=12` (`step = c+1 = 6`) to pin the boundary concretely:

- `j=0`: `rem=12`, `C(4,0)*C(15,3) = 1*455 = 455`, sign `+`.
- `j=1`: `rem=6`,  `C(4,1)*C(9,3)  = 4*84  = 336`, sign `-`.
- `j=2`: `rem=0`,  `C(4,2)*C(3,3)  = 6*1   = 6`,   sign `+`.
- `j=3`: `rem = 12 - 18 = -6 < 0` -> stop.

`455 - 336 + 6 = 125`, the sample answer. The last contributing term, `j=2`, sits exactly on `rem = 0`: two children maxed out absorb the whole total, a genuine configuration worth `+6`. So the stopping test must be inclusive, `S - j*(c+1) >= 0`; a strict `> 0` exits one term early, drops the `+6`, and reports `119`. One character, `>` versus `>=`, is the entire difference between right and wrong, and this off-by-one is precisely what the problem is built to punish, so the guard reads `S - j*step >= 0`.

One neighbouring off-by-one to keep straight: the unbounded count is `C(rem + k - 1, k - 1)` — `k` parts, `k - 1` bars among `rem + k - 1` slots — not `C(rem + k, k - 1)`, which would smuggle in a phantom slot and inflate every term.

For the binomials I need `C(N, r)` with `N` up to `S + k - 1`, plus `C(k, j)` for `j` up to `k`. A single factorial table of size `maxN = S + k` covers all of it: every top I request is `< maxN`, and `C(k, j)` reads only `fact[k]` with `k <= maxN`. Because `M` is prime and exceeds every argument I touch, I get `inv[maxN]` from one Fermat exponentiation and walk the inverse factorials downward. The `C` helper returns `0` on `r < 0`, `N < 0`, or `r > N`, which also lets any out-of-range term vanish on its own; I still bound the loop by `j <= k` so `fact[k]` is never read past intent.

Now the corners, where boundary code like this tends to die:

- `k = 0`: no children, so the only feasible total is `S = 0` (one distribution), else `0`. I special-case this before the factorial machinery — letting the loop run forms `C(rem - 1, -1) = 0` and would wrongly print `0` even for `S = 0`.
- `S = 0`, `k >= 1`: only `j = 0` survives, `C(k-1, k-1) = 1`, the all-zeros distribution.
- `S = k*c` (everyone maxed): exactly one distribution; the alternating sum collapses to `1`.
- `S > k*c` (infeasible): answer `0`, produced automatically — e.g. `k=3, c=2, S=7 -> 0`.
- `c = 0`: `step = 1`, every child forced to `0`, feasible only at `S = 0`.
- `c >= S` (cap never binds): `step = c+1 > S`, so only `j = 0` survives and the answer is the pure stars-and-bars `C(S + k - 1, k - 1)`. Pushing this to `k=10^6, c=10^9, S=10^6` also confirms `j*step` (up to `~10^{15}`) stays inside `long long`.

To close the loop I cross-check the inclusion-exclusion against the bounded-composition DP on several hundred random small cases, weighted toward the boundary-heavy seeds — `S` an exact multiple of `c+1`, which force a `rem = 0` final term — plus `S = 0`, `S = k*c`, `S > k*c`, small primes that still exceed `S + k`, and `10^9 + 7`. They agree everywhere, including precisely the seeds a `> 0` guard would have failed.

What ships is the `O(S + k)` inclusion-exclusion: special-case `k = 0` (and a defensive `S < 0`), build factorial and inverse-factorial tables to `S + k`, then run the alternating sum with the inclusive `>= 0` guard and the `C(rem + k - 1, k - 1)` top. The full program is in the answer.
