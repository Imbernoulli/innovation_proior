**Reading the problem and pinning the contract.** I am given `q` queries; each is `n a b`, and I must count the integer widths `x` with `1 <= x <= n` such that `a <= floor(n/x) <= b`. Output one line per query. Before any algorithm I fix the scale, because it rules things out immediately: `q` up to `2*10^5` and `n` up to `10^12`. So a per-query `O(n)` scan over `x` is `2*10^5 * 10^12` operations in the worst case — hopelessly slow under a 1-second limit, and even a single `n = 10^12` query scanned naively is already too slow. I need `O(1)` (or at worst `O(sqrt n)`) per query. Also `n` and `b` reach `10^12`, and I will be forming `b+1`, so 64-bit arithmetic everywhere: `long long`. An `int` for `n` or for the product-ish quantities is a silent wrong answer on the large tests. That is decided and non-negotiable.

**Candidate approaches.** Two routes are on the table.

- *Per-query scan.* Loop `x` from `1` to `n`, test `a <= n/x <= b`, count. Trivially correct, trivially too slow. Useful only as the brute-force oracle I will check against, never as the submission.
- *Closed-form interval count.* Exploit that `x -> floor(n/x)` is non-increasing on `[1, n]`. Each one-sided quotient bound becomes a one-sided bound on `x`, so the valid `x` form a contiguous interval `[lo, hi]`, and the answer is just `hi - lo + 1` (clamped). This is `O(1)` per query. The whole risk is in the algebra of the two endpoints — exactly where off-by-ones live.

I commit to the closed-form route; the scan is my oracle.

**Deriving the two index bounds.** Since `floor(n/x)` only decreases as `x` grows, "quotient large" means "x small" and "quotient small" means "x large". Concretely, with `n >= 1`:

- *Upper bound from `floor(n/x) >= a`.* For `a >= 1`: `floor(n/x) >= a` iff `n/x >= a` iff `x <= n/a` iff `x <= floor(n/a)` (last step because `x` is an integer and `n/a` may be fractional). So the `>= a` side caps `x` at the **inclusive** value `hi = floor(n/a)`. If `a <= 0`, the condition `floor(n/x) >= a` holds for every `x in [1, n]` (the quotient is always `>= 0 >= a`), so `hi = n`.
- *Lower bound from `floor(n/x) <= b`.* `floor(n/x) <= b` iff `n/x < b+1` iff `x > n/(b+1)` iff `x > floor(n/(b+1))` (again because `x` is an integer: `x > n/(b+1)` with integer `x` is the same as `x >= floor(n/(b+1)) + 1`, i.e. `x` strictly exceeds `floor(n/(b+1))`). So the `<= b` side gives an **exclusive** lower bound `loExcl = floor(n/(b+1))`; valid `x` satisfy `x > loExcl`, i.e. `x >= loExcl + 1`.

So the valid `x` lie in the interval `( loExcl , hi ]` — open on the left, closed on the right — and the count of integers there is `hi - loExcl` (since the integers strictly greater than `loExcl` and at most `hi` are exactly `loExcl+1, ..., hi`, a block of size `hi - loExcl`). That asymmetry — closed on the `a` side, open on the `b` side, and `b+1` not `b` — is the heart of the problem.

**Derivation sanity-check on the sample.** Before coding I check the formula on `n=10, a=2, b=3`, where I expect `3`. `hi = floor(10/2) = 5`. `loExcl = floor(10/(3+1)) = floor(10/4) = 2`. Count `= hi - loExcl = 5 - 2 = 3`. The valid `x` are `{3, 4, 5}` (those strictly above 2, up to 5): `floor(10/3)=3`, `floor(10/4)=2`, `floor(10/5)=2`, all in `[2,3]`. Correct, and the boundary `x=3` (quotient exactly `b=3`) is included while `x=2` (quotient `5 > 3`) is excluded — the open/closed split is doing its job. Also `n=10, a=1, b=1`, expect `5`: `hi=floor(10/1)=10`, `loExcl=floor(10/2)=5`, count `10-5=5`, the widths `{6,7,8,9,10}`. Correct.

**First implementation and a trace.** My first cut of `solve`:

```
long long solve(long long n, long long a, long long b) {
    if (n <= 0) return 0;
    if (a > b) return 0;
    long long hi = n / a;             // largest x with floor(n/x) >= a
    long long loExcl = n / b;         // <-- careless: used b, not b+1
    long long cnt = hi - loExcl;
    if (cnt < 0) cnt = 0;
    return cnt;
}
```

I deliberately trace this on a single-value band, because that is where boundary bugs surface. Take `n=10, a=1, b=1`, expected `5`. `hi = 10/1 = 10`. `loExcl = 10/1 = 10` (using `b=1`). `cnt = 10 - 10 = 0`. The code returns `0`, but the answer is `5`.

**The bug.** Tracing pinpoints it exactly: I wrote `loExcl = n / b`, but the lower bound from `floor(n/x) <= b` is `floor(n/(b+1))`, not `floor(n/b)`. With `b=1` the correct `loExcl` is `floor(10/2)=5`, giving `10-5=5`; my `floor(10/1)=10` chops off the entire valid block. The conceptual slip: `floor(n/x) <= b` is `n/x < b+1`, and the strict `< b+1` is what produces the `b+1` in the denominator. Using `b` would correspond to `floor(n/x) < b`, i.e. quotient at most `b-1` — off by one band. Fix: `loExcl = n / (b + 1)`.

Let me also recheck the *upper* side under this fix on the same case: `floor(n/x) >= a` with `a=1` gives `hi = floor(10/1) = 10`, inclusive, correct (all `x` have quotient `>= 1`). Good, that side was already right. Apply the fix and re-trace `n=10,a=1,b=1`: `hi=10`, `loExcl=floor(10/2)=5`, `cnt=5`. Correct. Re-trace `n=10,a=2,b=3`: `hi=floor(10/2)=5`, `loExcl=floor(10/4)=2`, `cnt=3`. Correct.

**Second implementation and a second trace — the `a = 0` / `b = 0` corners.** The fixed core handles the "normal" cases, but the contract permits `a` and `b` down at `0`, and `a = 0` breaks `n / a` with a division by zero, while `b = 0` makes `b+1 = 1` which is fine but worth checking. I extend `solve`:

```
long long solve(long long n, long long a, long long b) {
    if (n <= 0) return 0;
    if (a > b) return 0;
    long long hi = (a <= 0) ? n : (n / a);   // a<=0: quotient>=a always
    long long loExcl = n / (b + 1);
    long long cnt = hi - loExcl;
    if (cnt < 0) cnt = 0;
    return cnt;
}
```

Now I trace the awkward `n=7, a=0, b=0`, expected `0` (no width in `[1,7]` gives quotient `0`, since `floor(7/x) >= 1` for all `x <= 7`). `a=0` so `hi = n = 7`. `loExcl = floor(7/(0+1)) = floor(7/1) = 7`. `cnt = 7 - 7 = 0`. Correct — and I see *why* it is correct: with band `[0,0]`, the `<= 0` constraint forces `x > floor(n/1) = n`, but the `>= 0` constraint allows all `x <= n`; the two collide exactly at `x = n` with no integer strictly above `n` and at most `n`, so the interval is empty. The open lower bound is essential here; had I (wrongly) made the lower side inclusive, I would count `x` values with quotient `1`, which are not in `[0,0]`.

**A third trace — a band entirely above the maximum quotient.** Consider `n=6, a=7, b=9`, expected `0` (the largest possible quotient is `floor(6/1)=6 < 7`). `hi = floor(6/7) = 0`. `loExcl = floor(6/10) = 0`. `cnt = 0 - 0 = 0`. Correct. Here `hi` collapsing to `0` is what zeroes the count; no width can reach quotient `>= 7`, and `floor(6/7)=0` encodes "even `x=1` is already too wide". Good — the formula degrades gracefully without a special case.

**Edge cases, deliberately.**

- *`n = 1`.* Only `x = 1`, `floor(1/1) = 1`. Band `[1,1]`: `hi = floor(1/1)=1`, `loExcl = floor(1/2)=0`, `cnt = 1`. Correct. Band `[0,0]`: `a=0` so `hi=1`, `loExcl=floor(1/1)=1`, `cnt=0`. Correct.
- *Full band `[1, n]`.* Every `x` qualifies, expect `n`. `hi = floor(n/1) = n`, `loExcl = floor(n/(n+1)) = 0`, `cnt = n`. Correct — `floor(n/(n+1))` is `0` precisely because `n < n+1`, so the open lower bound reaches all the way down to include `x=1`.
- *Large `n` near the type limit.* `n = 10^12`, band `[10^12, 10^12]`: `hi = floor(10^12 / 10^12) = 1`, `loExcl = floor(10^12 / (10^12 + 1)) = 0`, `cnt = 1` (only `x = 1`). The denominator `b+1 = 10^12 + 1` fits comfortably in `long long` (limit ~`9.2*10^18`), and no multiplication of two large quantities ever happens, so there is no overflow. With `int` this would already be broken at the read.
- *Degenerate `a > b`.* I guard with an early `return 0`; an empty band has no satisfying `x`.
- *Clamping.* `hi` cannot exceed `n` in the `a >= 1` branch (since `floor(n/a) <= n`), and in the `a <= 0` branch I set `hi = n` exactly, so `hi <= n` always; `loExcl = floor(n/(b+1)) >= 0`. The final `if (cnt < 0) cnt = 0` covers the case `loExcl > hi` (interval empty), e.g. when the band is unreachable. I keep a defensive `if (loExcl < 0) loExcl = 0` and `if (hi > n) hi = n` even though under the stated constraints they never trigger, so the routine is robust to the degenerate orderings the contract mentions.
- *A guard for negative `b`.* The contract keeps `b >= 0` in well-formed tests, but if `b < 0` ever slips in, `floor(n/x) >= 0 > b` for all `x`, so the answer must be `0`; I add `if (b < 0) return 0` before forming `b+1` so I never compute `n/(b+1)` with `b+1 <= 0`.

**Re-verification against the brute oracle.** I run the closed-form `solve` against the obvious `O(n)` scan on hundreds of random small cases (`n` up to a few hundred, `a, b` chosen both near real quotient values and degenerate, including `a=0`, single-value bands `[k,k]`, and bands above `n`), and on a second batch with bounds drawn up to `n+2` to stress the empty/clamped regime. All cases agree. The two earlier traces that exposed the `b` vs `b+1` slip and the `a=0` division now pass for the reasons I fixed, which is the evidence I trust more than "it looks right".

**Final solution.** I convinced myself the closed form is right by the two floor identities and a sample sanity-check, and I convinced myself the *code* is right by tracing the single-value band that exposed the `n/b`-vs-`n/(b+1)` off-by-one, tracing the `a=0`/`b=0` corners, and re-verifying every fix against an independent scan. This is what I ship — one self-contained `O(q)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count x in [1, n] with floor(n/x) in [a, b], a<=b.
// floor(n/x) >= a  <=>  x <= floor(n/a)            (for a >= 1; if a <= 0 every x qualifies)
// floor(n/x) <= b  <=>  x >  floor(n/(b+1))        (since floor(n/x) <= b iff x > n/(b+1))
// So x ranges over ( floor(n/(b+1)) , floor(n/a) ], a left-open / right-closed interval.
long long solve(long long n, long long a, long long b) {
    if (n <= 0) return 0;
    if (a > b) return 0;
    if (b < 0) return 0;            // floor(n/x) >= 0 always; cannot be <= negative b
    // Upper index bound from the a-condition (floor(n/x) >= a).
    long long hi;
    if (a <= 0) {
        hi = n;                     // floor(n/x) >= a holds for all x in [1, n]
    } else {
        hi = n / a;                 // largest x with floor(n/x) >= a
    }
    if (hi > n) hi = n;
    // Lower exclusive bound from the b-condition (floor(n/x) <= b).
    long long loExcl = n / (b + 1); // floor(n/x) <= b iff x > floor(n/(b+1))
    if (loExcl < 0) loExcl = 0;
    // x in (loExcl, hi], also forced into [1, n] by construction.
    long long cnt = hi - loExcl;
    if (cnt < 0) cnt = 0;
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, a, b;
        cin >> n >> a >> b;
        cout << solve(n, a, b) << "\n";
    }
    return 0;
}
```

**Causal recap.** Monotonicity of `floor(n/x)` turns the two-sided quotient band into a single interval of widths `x`, so the count is just the interval length — but the two endpoints have *opposite* inclusivity: `floor(n/x) >= a` gives the inclusive cap `x <= floor(n/a)`, while `floor(n/x) <= b` comes from `n/x < b+1` and gives the *exclusive* floor `x > floor(n/(b+1))`, with `b+1` and not `b`. My first code used `n/b` and a trace of the single-value band `n=10, a=1, b=1` returned `0` instead of `5`, exposing exactly that off-by-one; switching to `n/(b+1)` fixed it. A second trace of `n=7, a=0, b=0` forced the `a=0` (division-by-zero) and `b=0` corners, which I handle by capping `hi=n` when `a<=0` and by the open lower bound that correctly makes `[0,0]` empty; a third trace of the unreachable band `n=6, [7,9]` showed `hi` collapsing to `0` zeroes the count without a special case. With `long long` throughout (so `b+1` up to `10^12+1` and `n` up to `10^12` never overflow) and the final clamp `max(0, hi - loExcl)`, the `O(q)` closed form is correct and fast, and it agrees with an independent `O(n)` scan over hundreds of randomized small cases.
