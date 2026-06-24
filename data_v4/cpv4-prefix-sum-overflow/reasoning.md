**Reading the problem and pinning the contract.** I have `n` hourly net-flow readings `a[1..n]`, each a signed integer in `[-10^9, 10^9]`, and `q` audit queries. Each query is a window `[l, r]` and its answer is the plain sum `a[l] + a[l+1] + ... + a[r]`. The single thing I must print is the *grand total* — the sum of all `q` query answers. Input is `n q` on the first line, then the `n` readings, then `q` lines of `l r`; output is one integer. Before I touch an algorithm I want the scale fixed, because here the scale is the whole point and it dictates the data types. `n <= 10^5` and `|a[i]| <= 10^9`, so a single window sum can reach `n * max|a| = 10^5 * 10^9 = 10^{14}`. With `q <= 5*10^4` windows, each as large as `10^{14}`, the running total can reach `q * 10^{14} = 5*10^{18}` in magnitude, of *either* sign (windows can be all-negative). So I am dealing with numbers up to `5*10^{18}`. A 32-bit `int` tops out near `2.1*10^9`; even one window sum overflows it, and the total overshoots it by a factor of about `2*10^9`. A 64-bit `long long` tops out near `9.2*10^{18}`, so `5*10^{18}` fits with margin to spare — but only just over half the range, which means I must *not* accidentally widen the bound (e.g. by allowing larger `q`). I will use `long long` for every prefix value and for the accumulator. This is decision number one and it is non-negotiable; an `int` here is a silent wrong-answer, not a crash.

**Laying out the candidate approaches.** Two routes, and I want the one I can both prove correct *and* afford under a 1-second limit.

- *Naive per-query re-summation.* For each query `[l, r]`, loop `i` from `l` to `r` and add `a[i]`. Dead simple, obviously correct, `O(r-l+1)` per query. The cost is `O(n*q)` worst case: `10^5 * 5*10^4 = 5*10^9` additions. At a few hundred million simple ops per second that is tens of seconds — far past the limit. Correct but unaffordable. I keep it only as my mental brute-force oracle, not as the submission.
- *Prefix sums.* Precompute `prefix[0] = 0` and `prefix[i] = a[1] + ... + a[i]` in one `O(n)` pass. Then any window sum is one subtraction: `a[l] + ... + a[r] = prefix[r] - prefix[l-1]`, `O(1)` per query. Total work `O(n + q)`. This is the standard acceleration and it is clearly fast enough. The risk is not the idea — it is two transcription traps: the `prefix[l-1]` indexing (an off-by-one silently wrecks every query that starts at `l = 1`), and the data types (already settled: 64-bit).

I commit to prefix sums. The idea's correctness is elementary, so let me nail the indexing definition before coding, since that is where this family of bug lives.

**Deriving the prefix-sum identity and checking it on paper.** Define the array 1-indexed with a sentinel at 0: `prefix[0] = 0`, and for `i >= 1`, `prefix[i] = prefix[i-1] + a[i]`, i.e. `prefix[i] = a[1] + a[2] + ... + a[i]`. I claim the window sum `S(l, r) = a[l] + ... + a[r]` equals `prefix[r] - prefix[l-1]`. Proof by telescoping: `prefix[r] = a[1] + ... + a[r]` and `prefix[l-1] = a[1] + ... + a[l-1]`; subtracting cancels `a[1..l-1]` and leaves exactly `a[l] + ... + a[r]`. The crucial boundary is `l = 1`: then `prefix[l-1] = prefix[0] = 0`, and `S(1, r) = prefix[r] - 0 = prefix[r]`, which is correct because the window starts at the first element. This is exactly why I want the `prefix[0] = 0` sentinel and 1-indexing — it removes the `l = 1` special case. If instead I 0-indexed the readings and tried `prefix[r] - prefix[l-1]` with `l = 0`, I would read `prefix[-1]`, which is out of bounds; that is the trap I am steering around by design.

Let me hand-check the identity on the documented sample. Readings `a = [10^9, 10^9, 10^9, -5, 10^9]` (1-indexed), queries `[1,3], [2,5], [1,5]`. Prefix: `prefix[0]=0`, `prefix[1]=10^9`, `prefix[2]=2*10^9`, `prefix[3]=3*10^9`, `prefix[4]=3*10^9 - 5 = 2999999995`, `prefix[5]=2999999995 + 10^9 = 3999999995`. Now the queries:
- `[1,3]`: `prefix[3] - prefix[0] = 3*10^9 - 0 = 3000000000`. Direct sum `10^9+10^9+10^9 = 3*10^9`. Match.
- `[2,5]`: `prefix[5] - prefix[1] = 3999999995 - 10^9 = 2999999995`. Direct `10^9+10^9-5+10^9 = 2999999995`. Match.
- `[1,5]`: `prefix[5] - prefix[0] = 3999999995`. Direct sum of all five `= 3999999995`. Match.

Total `= 3000000000 + 2999999995 + 3999999995 = 9999999990`. That equals the documented answer, so the derivation reproduces the sample. Two things jump out from these numbers and they validate my type analysis: the very first window sum, `3000000000`, already exceeds `INT_MAX = 2147483647`; and the grand total `9999999990` is about `4.65 *` the 32-bit range. So even this tiny five-element sample would break a 32-bit solution — the overflow is not a remote large-test artifact, it is visible immediately.

**First implementation — and immediately a trace, because clean math transcribes dirty.** Here is my first cut. I am deliberately writing it the way a hurried version might look, then I will trace it.

```
int n, q; cin >> n >> q;
vector<int> prefix(n + 1, 0);          // (a) 32-bit prefix
for (int i = 1; i <= n; i++) { int x; cin >> x; prefix[i] = prefix[i - 1] + x; }
int total = 0;                          // (b) 32-bit accumulator
for (int k = 0; k < q; k++) {
    int l, r; cin >> l >> r;
    total += prefix[r] - prefix[l - 1];
}
cout << total << "\n";
```

I will trace this on the documented sample, because the sample's magnitudes are exactly where 32-bit dies. Reading: `n=5, q=3`, readings `[10^9, 10^9, 10^9, -5, 10^9]`. Building `prefix` as `int`: `prefix[1] = 0 + 10^9 = 1000000000` (fits, since `INT_MAX = 2147483647`). `prefix[2] = 1000000000 + 1000000000 = 2000000000` — still under `2147483647`, so it fits, barely. `prefix[3] = 2000000000 + 1000000000 = 3000000000` — but `3000000000 > 2147483647`, so as a 32-bit signed `int` this *overflows*. The wrapped value is `3000000000 - 2^{32} = 3000000000 - 4294967296 = -1294967296`. So `prefix[3]` is stored as `-1294967296`, not `3*10^9`. Everything downstream that uses `prefix[3]` (or `prefix[4]`, `prefix[5]`, which are built on it) is now garbage. The first query `[1,3]` would compute `prefix[3] - prefix[0] = -1294967296 - 0 = -1294967296`, a negative number where the true answer is `+3000000000`. The accumulator, also `int`, then keeps wrapping on top of that.

**Diagnosing bug #1 (the overflow).** The defect is precisely the type choice at lines (a) and (b). The arithmetic the problem requires — prefix values up to `10^{14}`, a total up to `5*10^{18}` — cannot be represented in 32 bits, and C++ silently wraps signed overflow into a garbage value rather than signaling. The trace makes it concrete: `prefix[3]` flips from `+3*10^9` to `-1294967296` the instant the partial sum crosses `INT_MAX`. To confirm I am not imagining it, I actually compiled exactly this `int` version and ran it on the sample: it printed `1410065398`, while the true total is `9999999990`. So the bug is real and observable, not theoretical. The fix is to make every quantity that can exceed the 32-bit range a `long long`: the prefix array and the accumulator. `long long` holds up to `9.2*10^{18}`, comfortably above the `5*10^{18}` worst case. (I also note the difference `prefix[r] - prefix[l-1]` is itself a window sum, magnitude `<= 10^{14}`, so doing the subtraction in 64-bit is safe; doing it in 32-bit would be wrong even before the accumulation.)

**Fixing #1 and re-verifying on the sample.** Switch the array and the accumulator to 64-bit:

```
vector<long long> prefix(n + 1, 0);
for (int i = 1; i <= n; i++) { long long x; cin >> x; prefix[i] = prefix[i - 1] + x; }
long long total = 0;
for (int k = 0; k < q; k++) { int l, r; cin >> l >> r; total += prefix[r] - prefix[l - 1]; }
```

Re-trace the sample with 64-bit: `prefix = [0, 10^9, 2*10^9, 3*10^9, 2999999995, 3999999995]`, no wrap. Query `[1,3]`: `3*10^9 - 0 = 3000000000`. Query `[2,5]`: `3999999995 - 10^9 = 2999999995`. Query `[1,5]`: `3999999995 - 0 = 3999999995`. Total `= 9999999990`. Matches the documented output, and matches my hand-derivation. The overflow is gone.

**Second trace, hunting the off-by-one — because the `l=1` boundary is the other classic trap.** Type-correct does not mean index-correct. I want to confirm the `prefix[l-1]` reference is right specifically at the left boundary, and that I have not introduced a subtler indexing slip. Let me trace a deliberately boundary-heavy tiny case: `n=3, q=2`, readings `a = [5, 7, 2]` (1-indexed), queries `[1,1]` and `[1,3]`. Prefix: `prefix[0]=0, prefix[1]=5, prefix[2]=12, prefix[3]=14`. Query `[1,1]`: `prefix[1] - prefix[0] = 5 - 0 = 5`. True window sum is just `a[1] = 5`. Correct. Query `[1,3]`: `prefix[3] - prefix[0] = 14 - 0 = 14`. True sum `5+7+2 = 14`. Correct. Total `= 19`.

Now let me imagine the off-by-one mistake explicitly to make sure my code does not have it: a common slip is to write `prefix[r] - prefix[l]` (forgetting the `-1`), which would *exclude* `a[l]` from the window. On query `[1,3]` that wrong form gives `prefix[3] - prefix[1] = 14 - 5 = 9`, dropping `a[1]=5` — wrong. My code writes `prefix[l - 1]`, so on `[1,3]` it uses `prefix[0]=0` and keeps `a[1]`. Good — the `-1` and the `prefix[0]=0` sentinel are both present and they are what make `l=1` correct. The other slip — 0-indexing the readings and then evaluating `prefix[l-1]` at `l=1` as `prefix[0]` but storing readings starting at index 0 — would misalign `prefix` with `a`; I avoid it entirely by storing readings 1-indexed so that `prefix[i]` and `a[i]` share the same `i`. I will keep the 1-indexed sentinel layout.

**Diagnosing what bug #2 *would* have been.** The episode above is a self-verify that caught the risk *before* it shipped: had I written `prefix[l]` instead of `prefix[l-1]`, every query would have silently dropped its first element, and crucially the *queries that start at `l=1` and span to `r` would lose the first reading* — exactly the windows most likely to be in the tests. The trace of `[1,3]` returning `9` instead of `14` is the signature of that bug, and my code does not exhibit it because I use `prefix[l-1]` against a `prefix[0]=0` sentinel. So the two traps this problem sets — overflow and the left-boundary off-by-one — are both now explicitly closed, the first by switching to `long long`, the second by the sentinel-plus-`(l-1)` indexing.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Minimal `n = q = 1`.* Input `1 1`, reading `[-10^9]`, query `[1,1]`. `prefix=[0, -10^9]`. Query: `prefix[1]-prefix[0] = -10^9 - 0 = -10^9`. Total `-10^9`. Correct — a single window of one negative element. I ran this; it printed `-1000000000`.
- *Whole-array window repeated many times (the positive overflow extreme).* `n=10^5`, all readings `10^9`, `q=5*10^4`, every query `[1, n]`. Each window sum is `10^5 * 10^9 = 10^{14}`; total `= 5*10^4 * 10^{14} = 5*10^{18}`. This is the largest the total can be; `5*10^{18} < 9.2*10^{18} = LLONG_MAX`, so it fits. I ran exactly this and got `5000000000000000000`, matching. With `int` it would be a wrapped mess.
- *The negative extreme.* Same shape but all readings `-10^9`: total `= -5*10^{18}`, which is `> LLONG_MIN = -9.2*10^{18}`, so it fits. I ran it and got `-5000000000000000000`. Correct, and confirms I have margin on *both* sides — which is why the `q <= 5*10^4` bound matters: a larger `q` would push `q*10^{14}` past `9.2*10^{18}` and I would need `unsigned`/`__int128`. Within the stated bound, `long long` is exactly sufficient.
- *Zeros and mixed signs.* Readings with zeros and a mix of large positives/negatives are handled identically — prefix sums and the subtraction are sign-agnostic; nothing special is needed. My random stress (below) includes these.
- *Performance.* `O(n + q)` with one `vector<long long>` of size `n+1` (about `0.8 MB`) and `sync_with_stdio(false)`. The max-scale run finished in `0.01 s` using `~4 MB`, well inside the `1 s` / `256 MB` budget.

**Sanity-checking the whole pipeline against an independent brute force.** Math checks and hand-traces are necessary but I want machine confirmation, so I wrote an independent Python brute force that answers each query by *direct re-summation* (`sum(a[l..r])`, the naive `O(n*q)` method I rejected for speed but trust for correctness), totals the results, and prints it. That brute uses a completely different mechanism — no prefix array, no subtraction — so agreement is real evidence, not a shared bug. A generator emits tiny random instances (`n, q <= 8`) mixing small values, full-range `+-10^9` values, and zeros, with random `[l, r]` windows. Across 400 random seeds my `long long` solution and the brute agreed on every case (0 mismatches), and on the documented sample it prints `9999999990`. Together with the boundary and overflow runs above, that is the evidence I trust.

**Final solution.** I convinced myself the *idea* is right by telescoping the prefix identity and reproducing the sample by hand, and I convinced myself the *code* is right by tracing the overflow to a precise wrap (`prefix[3]` flipping to `-1294967296` under `int`, the compiled `int` build printing `1410065398` instead of `9999999990`), by tracing the left-boundary indexing to confirm `prefix[l-1]` with a `prefix[0]=0` sentinel keeps `a[1]`, and by 400-case agreement with an independent brute plus the `+-5*10^{18}` extremes. This is what I ship — one self-contained file, 64-bit throughout:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    // prefix[i] = a[1] + a[2] + ... + a[i], prefix[0] = 0.
    // Values reach n * max|a| = 1e5 * 1e9 = 1e14, far beyond 32-bit range,
    // so prefix sums MUST be 64-bit.
    vector<long long> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Sum over each query window [l, r] is prefix[r] - prefix[l-1].
    // The grand total over up to 5e4 windows of magnitude up to 1e14
    // reaches ~5e18, which still fits in long long but overflows int many times over.
    long long total = 0;
    for (int k = 0; k < q; k++) {
        int l, r;
        cin >> l >> r;
        total += prefix[r] - prefix[l - 1];
    }

    cout << total << "\n";
    return 0;
}
```

**Causal recap.** The naive per-query re-summation is correct but `O(n*q) = 5*10^9` and too slow, so I switched to prefix sums where each window is `prefix[r] - prefix[l-1]` in `O(1)`; the telescoping proof and a by-hand sample reproduction confirmed the identity. The first cut used `int`, and tracing the sample showed `prefix[3]` wrapping from `+3*10^9` to `-1294967296` the moment the partial sum crossed `INT_MAX` — confirmed by compiling that version and watching it print `1410065398` instead of `9999999990` — so I made the prefix array and the accumulator `long long`, which holds the `~5*10^{18}` worst-case total with the stated `q <= 5*10^4` bound. A second trace pinned the left-boundary trap: `prefix[l-1]` against a `prefix[0]=0` sentinel keeps `a[1]` for `l=1` windows (whereas `prefix[l]` would drop it). Edge runs at `n=q=1` and the `+-5*10^{18}` extremes, plus 400-case agreement with an independent direct-summation brute, close it out.
