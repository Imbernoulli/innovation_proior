I start by trying to count triples directly. If a set in `[N]` has density `delta`, the random model predicts about `delta^3` of the available pairs `(a,r)` should have `a`, `a+r`, and `a+2r` all in the set. That would finish the problem immediately, but it assumes exactly what I do not have: independence. The old digit and sphere constructions are the warning sign. They are sparse, but they show that additive structure can kill many three-term progressions without looking random at all.

So the count has to become diagnostic. I embed `[N]` into an odd cyclic group `Z/N'Z`, say with `N' = 2N+1`, and extend `1_A` by zero. The odd modulus matters only so multiplication by `2` is invertible. I count with

`Lambda(f,g,h) = E_{n,r in Z/N'Z} f(n) g(n+r) h(n+2r)`.

If `A` has no nontrivial three-term progressions, then `Lambda(1_A,1_A,1_A)` only sees the degenerate `r=0` triples, plus the harmless normalization, so it is `O(1/N)`. For large `N` that is tiny compared with the random main term I should get from a dense set.

Now the sign pattern has to be exact. With the convention `hat f(alpha) = E_n f(n)e(-alpha n)` and inversion `f(n) = sum_alpha hat f(alpha)e(alpha n)`, the Fourier expansion of `Lambda(f,g,h)` contains the average of

`e(alpha_1 n + alpha_2(n+r) + alpha_3(n+2r))`.

This average vanishes unless `alpha_1 + alpha_2 + alpha_3 = 0` and `alpha_2 + 2 alpha_3 = 0`. Hence the surviving frequencies are exactly `(alpha, -2 alpha, alpha)`, because `alpha n - 2 alpha(n+r) + alpha(n+2r) = 0`. Therefore

`Lambda(f,g,h) = sum_alpha hat f(alpha) hat g(-2 alpha) hat h(alpha)`.

Good, no missing sign: the middle coefficient is `-2`.

Write the actual density as `delta' = |A|/N` and split `1_A = delta' 1_[N] + f`, where `f = 1_A - delta' 1_[N]` has mean zero on `[N]`. The main term `Lambda(delta'1_[N], delta'1_[N], delta'1_[N])` is bounded below by a constant times `(delta')^3`, while the full triple count is tiny if there is no progression. So among the seven remaining trilinear terms, at least one has magnitude comparable to `(delta')^3`.

Each of those seven cases is controlled the same way. The identity above and Plancherel give, for example,

`|Lambda(u,v,w)| <= ||u||_2 ||v||_2 sup_xi |hat w(xi)|`,

and the same bound after permuting the three inputs. The functions `f` and `delta'1_[N]` both have normalized `L^2` norm `O((delta')^(1/2))`, so whichever slot contains a balanced copy, the large error term forces

`sup_xi |hat f(xi)| >= c (delta')^2 >= c delta^2`.

That is the first real conversion: absence of progressions gives a linear Fourier bias of size on the order of `delta^2`.

But a Fourier bias is still not a denser interval. It only says the balanced set sees some phase `e(alpha n)`. I need to stop that phase from oscillating. For a small `epsilon`, recurrence gives a step `h` with `||alpha h|| <= epsilon / L` for a long length scale `L`, and then `[N]` can be partitioned into arithmetic progressions `P_j` of common difference `h`, each of length at least `L`, plus an error set of size `O(epsilon N)`. On each `P_j`, the character changes by only `O(epsilon)`.

Choosing `epsilon` as a small constant multiple of `delta^2`, the correlation cannot hide in the discarded set or in phase fluctuation. It gives

`sum_j |sum_{n in P_j} (1_A(n) - delta')| >= c delta^2 N`.

The signs still matter. Since `f` has total sum zero on `[N]`, and the discarded error set is only `O(epsilon N)`, the signed sum over the progressions is `O(epsilon N)`. Combining this with the absolute-value lower bound, the positive deviations alone contribute a constant multiple of `delta^2 N`:

`sum_j max(sum_{n in P_j} (1_A(n) - delta'), 0) >= c' delta^2 N`.

One progression must therefore satisfy

`|A cap P_j| / |P_j| >= delta' + c'' delta^2`.

That is the density increment. It is not merely "some structure"; it is a long arithmetic progression on which the same set is denser by a definite amount depending only on the current positive density.

Now I can restart. An affine map sends `P_j` to a shorter interval, and it preserves three-term progressions exactly: three points in arithmetic progression go to three points in arithmetic progression, and a nonzero common difference stays nonzero. So a progression-free `A` produces a progression-free set of higher density on a still-long interval.

The clean endgame is to avoid pretending I know in advance how many shrinking steps I can take. Let `delta_*` be the infimum of densities for which every sufficiently large subset of `[N]` must contain a three-term progression. If `delta_* > 0`, take arbitrarily large progression-free sets with density just below `delta_*`. The density-increment argument moves one of them to an arbitrarily long subprogression with density above `delta_*`, after choosing the starting density close enough that the `c delta^2` gain beats the gap. But densities above `delta_*` are already in the forced-progression range, so the subprogression contains a progression, and the original set contains one too. Contradiction. Therefore `delta_* = 0`: fixed positive density cannot avoid nontrivial three-term arithmetic progressions in all sufficiently large intervals.
