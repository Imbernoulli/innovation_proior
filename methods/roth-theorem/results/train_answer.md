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

The code below does not prove Roth's theorem for all `N`, but it verifies the finitary phenomenon for small intervals. It first computes, by exact backtracking, the largest size `r_3(N)` of a three-term-arithmetic-progression-free subset of `[N]` for `N` up to 20, and it prints the ratio `r_3(N)/N`. Then it samples random dense subsets of larger intervals and compares the observed number of three-term progressions with the random prediction `delta^3` times the number of pairs `(a, r)`. You can run it as a standalone Python script.

```python
import random


def max_ap_free_size(n):
    """Exact backtracking size of the largest 3-AP-free subset of {1,...,n}."""
    best = 0
    nums = list(range(1, n + 1))

    def backtrack(start, current):
        nonlocal best
        # Prune if even adding all remaining elements cannot beat best.
        if len(current) + (n - start) <= best:
            return
        if start == n:
            best = max(best, len(current))
            return

        x = nums[start]
        # Try including x, provided it creates no three-term progression.
        ok = True
        for d in range(1, (x - 1) // 2 + 1):
            if x - d in current and x - 2 * d in current:
                ok = False
                break
        if ok:
            current.append(x)
            backtrack(start + 1, current)
            current.pop()

        # Exclude x.
        backtrack(start + 1, current)

    backtrack(0, [])
    return best


print("Exact progression-free sizes for small N:")
print("N  r_3(N)  ratio")
for N in range(1, 21):
    r = max_ap_free_size(N)
    print(f"{N:2d}   {r:2d}     {r / N:.3f}")


def count_3ap_random(N, delta, trials=1000):
    """Average number of 3-APs in random subsets of [N] of density delta."""
    total = 0
    num_pairs = sum((N - a) // 2 for a in range(1, N + 1))
    for _ in range(trials):
        A = {i for i in range(1, N + 1) if random.random() < delta}
        cnt = 0
        for a in range(1, N + 1):
            max_r = (N - a) // 2
            for r in range(1, max_r + 1):
                if a in A and (a + r) in A and (a + 2 * r) in A:
                    cnt += 1
        total += cnt
    # Each admissible pair (a, r) forms a 3-AP with probability delta^3.
    predicted = delta ** 3 * num_pairs
    return total / trials, predicted


random.seed(0)
print("\nRandom-model prediction versus observation:")
for N, delta in [(100, 0.1), (200, 0.1), (200, 0.2)]:
    observed, predicted = count_3ap_random(N, delta)
    print(f"N={N}, delta={delta}: observed {observed:.1f}, predicted {predicted:.1f}")
```

When I run this script, the small-N table shows that the maximum progression-free size grows, but the ratio `r_3(N)/N` is already trending downward, consistent with the `o(N)` bound predicted by Roth's theorem. The random-model section confirms that a typical density-`delta` set has roughly `delta^3` times the available pairs as three-term progressions, which is the main term that the density-increment argument protects. The proof's real strength is that it shows no fixed positive density can systematically suppress this count forever.