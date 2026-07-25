I will present Roth's theorem in additive combinatorics and explain the Fourier-analytic density-increment proof that makes it work. The canonical name for the result I am describing is Roth's theorem on three-term arithmetic progressions. It states that for every fixed positive density `delta` there is a threshold `N_0(delta)` such that every subset `A` of `{1, 2, ..., N}` with `N >= N_0(delta)` and `|A| >= delta N` contains a nontrivial three-term arithmetic progression `a, a+r, a+2r` with `r > 0`. Equivalently, if `r_3(N)` denotes the largest size of a subset of `[N]` that contains no such progression, then `r_3(N) = o(N)`.

The theorem is not obvious because there are large progression-free sets. Classical digit-based constructions, and later sphere-based ones such as Behrend's construction, give subsets of `[N]` of size `N exp(-c sqrt(log N))` that avoid three-term progressions. Since `exp(-c sqrt(log N))` tends to zero as `N` grows, these examples still have zero density, but they show that a naive counting argument cannot succeed. Any valid proof must separate the regime of fixed positive density from the regime of slowly vanishing density.

The strategy I use is to turn the absence of a progression into a structural certificate. I embed the interval `[N]` into an odd cyclic group `Z/N'Z`, typically taking `N' = 2N+1`, and extend the indicator function `1_A` by zero outside `[N]`. The odd modulus is convenient only because multiplication by `2` is invertible. I then count three-term progressions with the trilinear form

`Lambda(f, g, h) = E_{n, r in Z/N'Z} f(n) g(n+r) h(n+2r)`.

If `A` has no nontrivial three-term progression, then `Lambda(1_A, 1_A, 1_A)` sees only the degenerate triples with `r = 0`, together with a harmless normalization, so it is `O(1/N)`. For a dense set this is far below the random prediction.

To understand the random prediction, split the indicator as `1_A = delta' 1_[N] + f`, where `delta' = |A|/N >= delta` is the actual density and `f = 1_A - delta' 1_[N]` is the balanced function, which has mean zero on `[N]`. Expanding `Lambda(1_A, 1_A, 1_A)` produces eight terms. The main term `Lambda(delta'1_[N], delta'1_[N], delta'1_[N])` is bounded below by a constant multiple of `(delta')^3`, while the full count is `O(1/N)` if `A` is progression-free. Therefore, for large `N`, at least one of the remaining seven error terms must have magnitude comparable to `(delta')^3`.

Fourier analysis controls those error terms. With the convention `hat f(alpha) = E_n f(n) e(-alpha n)` and the inversion formula `f(n) = sum_alpha hat f(alpha) e(alpha n)`, the expansion of `Lambda(f, g, h)` averages the phase

`e(alpha_1 n + alpha_2 (n+r) + alpha_3 (n+2r))`.

This average vanishes unless `alpha_1 + alpha_2 + alpha_3 = 0` and `alpha_2 + 2 alpha_3 = 0`, which forces `(alpha_1, alpha_2, alpha_3) = (alpha, -2 alpha, alpha)`. Hence

`Lambda(f, g, h) = sum_alpha hat f(alpha) hat g(-2 alpha) hat h(alpha)`.

The middle sign is important: it is `-2`, not `+2`. Using this identity together with Plancherel gives the bound

`|Lambda(u, v, w)| <= ||u||_2 ||v||_2 sup_xi |hat w(xi)|`,

and the same bound holds after permuting the three inputs. Since both `f` and `delta'1_[N]` have normalized `L^2` norm `O((delta')^{1/2})`, whichever slot contains a balanced copy of `f`, the large error term forces a large balanced Fourier coefficient:

`sup_xi |E_{n in [N]} (1_A(n) - delta') e(-xi n)| >= c delta^2`.

This is the first conversion: the combinatorial assumption of no progression implies an analytic bias of order `delta^2`.

A Fourier bias is still not a denser subinterval. To obtain one, I stop the phase `e(alpha n)` from oscillating. Diophantine recurrence gives a step `h` with `||alpha h|| <= epsilon / L` for a long length scale `L`, and then `[N]` can be partitioned into long arithmetic progressions `P_j` of common difference `h`, each of length at least `L`, plus an error set of size `O(epsilon N)`. On each `P_j`, the character `e(alpha n)` varies by at most `O(epsilon)`. Choosing `epsilon` as a small constant multiple of `delta^2`, the correlation cannot hide inside the discarded set or inside phase fluctuation, and we obtain

`sum_j |sum_{n in P_j} (1_A(n) - delta')| >= c delta^2 N`.

Because `f` has total sum zero on `[N]`, and the discarded error set is only `O(epsilon N)`, the signed sum over the progressions is `O(epsilon N)`. Combining this with the absolute-value lower bound shows that the positive deviations alone contribute a constant multiple of `delta^2 N`. Therefore some progression `P_j` satisfies

`|A cap P_j| / |P_j| >= delta' + c'' delta^2`.

This is the density increment: a long arithmetic progression on which `A` is denser by a definite amount depending only on `delta`.

Now I restart the argument. An affine map sends `P_j` to a shorter interval and preserves three-term progressions exactly: three points in arithmetic progression go to three points in arithmetic progression, and a nonzero common difference stays nonzero. So a progression-free `A` would produce a progression-free subset of a smaller interval with higher density. The standard endgame is an infimum argument. Let `delta_*` be the infimum of densities for which every sufficiently large subset of `[N]` must contain a three-term progression. If `delta_* > 0`, choose arbitrarily large progression-free sets with density just below `delta_*`. The density-increment argument moves one of them to an arbitrarily long subprogression with density above `delta_*`, provided the starting density is close enough that the `c delta^2` gain beats the gap. But densities above `delta_*` are already in the forced-progression range, so the subprogression contains a progression, and the original set contains one too. This contradiction forces `delta_* = 0`, which is exactly Roth's theorem.

Put together, this is the finished result, stated with its hypotheses and with the mechanism that proves it made explicit. Roth's theorem on three-term arithmetic progressions says that for every $\delta > 0$ there is a threshold $N_0(\delta)$ such that every $A \subseteq \{1, 2, \dots, N\}$ with $N \ge N_0(\delta)$ and $|A| \ge \delta N$ contains a nontrivial three-term arithmetic progression $a,\ a+r,\ a+2r$ with $r > 0$; equivalently, if $r_3(N)$ denotes the size of the largest subset of $[N]$ containing no such progression, then

$$r_3(N) = o(N).$$

The certificate that proves it is exactly the density-increment step derived above. A hypothetical progression-free set $A \subseteq [N]$ of density $\delta' \ge \delta$ is forced, by the Fourier identity

$$\Lambda(f,g,h) = \sum_{\alpha} \hat f(\alpha)\,\hat g(-2\alpha)\,\hat h(\alpha),$$

into a linear bias on its balanced part $f = 1_A - \delta' 1_{[N]}$,

$$\sup_{\xi} \big|\hat f(\xi)\big| \ge c\,\delta^2,$$

and that bias is upgraded, via the partition of $[N]$ into long arithmetic progressions on which the biased frequency is nearly constant, into an honest denser subprogression $P$ with

$$\frac{|A \cap P|}{|P|} \ge \delta' + c''\,\delta^2$$

for an absolute constant $c'' > 0$. Because rescaling $P$ back to a full interval preserves three-term arithmetic progressions exactly, this increment can be reapplied to any progression-free set sitting at density just below the critical threshold $\delta_*$, driving it strictly above $\delta_*$ and so, by the definition of $\delta_*$, into a regime that already forces a progression — which is a contradiction unless $\delta_* = 0$. That is Roth's theorem: no fixed positive density can avoid nontrivial three-term arithmetic progressions in all sufficiently large intervals.