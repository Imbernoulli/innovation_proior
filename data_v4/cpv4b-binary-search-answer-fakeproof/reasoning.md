I'm handed a batch of quotas `K` (up to `10^12`, at most 50 of them), and for each I have to name the
smallest budget `m >= 1` at which the count `R(m) = #{(a, b) : a, b >= 1, a*b <= m}` reaches `K`. Two
facts settle the skeleton at once. `R` is non-decreasing — enlarging `m` only loosens `a*b <= m`, so
every pair counted at `m1` survives at any `m2 >= m1` — and a monotone predicate `R(m) >= K` flips
false→true exactly once, so the answer is the leftmost `m` where it holds: binary search on the
answer. The whole problem then lives in the predicate, because `R(m)` is the divisor-summatory
function `sum_{t<=m} d(t)`, and I have to compute it both fast and *exactly*. The sizes below make the
naive sum impossible and the slick closed form dangerous.

Sizing it. `R(m)` grows like `m ln m`, so `R(m) >= 10^12` lands near `m ~ 10^12 / ln(10^12) ~ 4*10^10`.
That fixes three things. `m` and every accumulator must be 64-bit — a trillion-scale count wraps a
32-bit int silently. `sqrt(m) ~ 2*10^5`, so an `O(sqrt m)` predicate is the target. And `50` quotas
times ~37 search steps times `2*10^5` is under `4*10^8` cheap operations, comfortably inside 2 seconds.
The naive `R(m) = sum_{a=1}^{m} floor(m/a)` (shelf `a` admits bins `1..floor(m/a)`) is
`O(m) ~ 4*10^10` per call — hopeless — so getting the sublinear form right is the entire job.

The sublinear form is the hyperbola count. Let `s = floor(sqrt(m))`.
Region A is the points with `a <= s`: for each such `a` the admissible `b` are `1..floor(m/a)`, giving
`sum_{a=1}^{s} floor(m/a)`. Region B, the points with `b <= s`, is identical by the `a<->b` symmetry.
A and B together cover every point with `a <= s` or `b <= s`; a point with both `a > s` and `b > s`
would have `a*b >= (s+1)^2 > m` (since `s+1 > sqrt(m)`), so it can't satisfy `a*b <= m` and nothing is
missed. The overlap, points with `a <= s` and `b <= s`, is doubly counted; there `a*b <= s*s <= m`
holds automatically, so the overlap is the full `s x s` block, exactly `s*s` points. Inclusion-exclusion
gives

```
R(m) = 2 * sum_{i=1}^{s} floor(m/i) - s*s,   s = floor(sqrt(m)).
```

The subtracted term is `s*s`, the area of the square — not `s`.

That distinction is worth pinning, because `-s` is a genuinely tempting misremembering: there's an `s`
sitting right there as the loop bound, and "subtract a fence-post `s`" feels natural. The definition on
small `m` separates the impostor from the truth — but notice the trap the small cases set. For every
`m` with `s = 1`, i.e. `m = 1, 2, 3`, the two corrections are *numerically equal*, since `s = s*s = 1`;
both reproduce `R(1)=1, R(2)=3, R(3)=5`, so a check that stops at `m=3` "confirms" the wrong formula.
The separation appears the instant `s` reaches 2, at `m = 4`: the definition gives
`floor(4/1)+floor(4/2)+floor(4/3)+floor(4/4) = 4+2+1+1 = 8`; the real form `2*(4+2) - 2*2 = 8` matches,
while `-s` gives `2*(4+2) - 2 = 10` and is exposed. So `-s*s` it is; I'll cross-check it against the
plain definition over the first few thousand `m` rather than trust one hand computation.

One more dependency: `s` must be *exactly* `floor(sqrt(m))`. If a floating `sqrtl` rounds up on a
near-square `m`, then `s*s` exceeds `m`, the overlap block picks up pairs with `a*b > m`, and the
count is wrong. A `long double` mantissa has room at `m ~ 4*10^10`, but a near-square can still round
the wrong way, so I pin `s` exactly: `r = (long long)sqrtl(m)`, then decrement while `r*r > m` and
increment while `(r+1)^2 <= m`. Those products, `~4*10^10` at `r ~ 2*10^5`, stay well inside 64-bit.

Now the search. My first instinct writes it as

```
long long lo = 1, hi = 70000000000LL;   // 7e10
while (lo <= hi) {
    long long mid = (lo + hi) / 2;
    if (R(mid) >= K) hi = mid;
    else lo = mid + 1;
}
```

and tracing the end-game on `K = 2` breaks it. Down at `lo=1, hi=2`: `mid=1`, `R(1)=1<2`, so `lo=2`;
now `lo=hi=2`, condition still true, `mid=2`, `R(2)=3>=2`, so `hi=2` — and `lo=hi=2` with the condition
*still* true, `mid=2` forever. It's the classic mismatch: `hi = mid` (not `mid-1`) only converges under
a strict `lo < hi`; paired with `lo <= hi` it stalls the instant `lo == hi == mid` on a true predicate.
The fix is the strict "leftmost-true" shape,

```
long long lo = 1, hi = 70000000000LL;
while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;   // subtractive, overflow-proof
    if (R(mid) >= K) hi = mid;
    else lo = mid + 1;
}
// lo == hi is the answer
```

Re-tracing `K=2`: at `lo=1,hi=2`, `mid=1`, `R(1)<2`, `lo=2`, loop exits, answer `2`. And `K=5`:
reaching `lo=1,hi=3`, `mid=2`, `R(2)=3<5`, `lo=3`, exit, answer `3` — both correct and terminating. I
also moved `mid` to the subtractive `lo + (hi-lo)/2`; here `lo+hi` fits anyway, but it's the form that
survives raising the bound.

The bound `hi = 7*10^10` has to actually satisfy the predicate for the largest legal `K`, or
leftmost-true would fall off the top and return `hi` spuriously. `R(7*10^10) ~ 1.76*10^12`, above
`10^12`, so every `K <= 10^12` has its threshold at or below `hi` and monotonicity does the rest. At
the very top this is tight enough to watch cross: for `K = 10^12` the answer is `40677885960`, with
`R(40677885959) < 10^12 <= R(40677885960)`.

Overflow, the one line that could bite, is `2*acc - s*s`. At the largest `m` the search probes, `acc`
is dominated by `floor(m/1) = m ~ 7*10^10` and totals `~9*10^11`, so `2*acc ~ 1.8*10^12` and
`s*s ~ 7*10^10` — both, and their difference, sit six orders of magnitude under the 64-bit ceiling
`~9.2*10^18`. `long long` throughout is enough; nothing needs 128-bit.

That covers the edges too. `K=1` returns `m=1`, because `lo` starts at `1` and `R(1)=1` (there's no
budget `0` to report, and `R(0)=0` satisfies nothing). Quotas landing between jumps of `R` (e.g.
`K=4`, with `R(2)=3 < 4 < 5 = R(3)`, answer `3`) and exactly on a jump (`K=5`, answer `3`) are both
handled by the `>= K` predicate, not `==`. Perfect-square `m` is exactly where `-s*s` earns its keep,
and the exact `isqrt` keeps that block count honest. Parsing is whitespace-agnostic `scanf`, with the
first read guarded so empty input exits rather than looping on garbage.
