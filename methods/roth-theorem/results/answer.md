# Roth's Theorem

For every `delta > 0` there is an `N_0(delta)` such that every subset `A` of `[N]` with `N >= N_0(delta)` and `|A| >= delta N` contains a nontrivial three-term arithmetic progression `a, a+r, a+2r` with `r > 0`.

Equivalently, if `r_3(N)` is the largest size of a subset of `[N]` with no nontrivial three-term arithmetic progression, then `r_3(N) = o(N)`.

## Fourier Density-Increment Form

Let `A subset [N]` have actual density `delta' = |A|/N >= delta` and suppose `A` has no nontrivial three-term arithmetic progression. Embed `[N]` in an odd cyclic group `Z/N'Z`, extend `1_A` by zero, and write

`1_A = delta' 1_[N] + f`,

where `f = 1_A - delta' 1_[N]` is balanced on `[N]`.

For

`Lambda(f,g,h) = E_{n,r} f(n) g(n+r) h(n+2r)`,

Fourier inversion gives

`Lambda(f,g,h) = sum_alpha hat f(alpha) hat g(-2 alpha) hat h(alpha)`,

because the only surviving frequency triples satisfy `(alpha_1, alpha_2, alpha_3) = (alpha, -2 alpha, alpha)`.

The main term is `>> (delta')^3`, while a progression-free set contributes only degenerate triples, `O(1/N)`. For sufficiently large `N`, one of the seven error terms is therefore large. Plancherel and the Fourier identity above imply a large balanced coefficient:

`|E_{n in [N]} (1_A(n) - delta') e(-alpha n)|`

is at least a constant times `delta^2`.

Partition `[N]` into long arithmetic progressions `P_j`, plus an error set of size `O(epsilon N)`, so that `e(alpha n)` varies by at most `O(epsilon)` on each `P_j`. Taking `epsilon` sufficiently small compared with `delta^2`, the correlation forces

`sum_j |sum_{n in P_j} (1_A(n) - delta')| >> delta^2 N`.

Since the total balanced sum is zero up to the small discarded error set, the positive deviations alone are large. Hence some long subprogression `P` has

`|A cap P| / |P| >= delta' + c delta^2`

for an absolute `c > 0`.

Affine rescaling maps `P` to a shorter interval and preserves nontrivial three-term arithmetic progressions. Therefore the rescaled set is still progression-free but has higher density. Applying the density-increment dichotomy through the standard infimum argument for the critical density gives a contradiction unless the critical density is `0`. Thus `r_3(N) = o(N)`.
