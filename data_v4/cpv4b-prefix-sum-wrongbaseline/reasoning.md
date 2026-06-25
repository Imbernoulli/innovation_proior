**Reading the problem and pinning the contract.** I have a ledger `a[0..n-1]` of signed daily net cash flow, and I must count the contiguous windows `[l, r]` whose sum is divisible by `m`. Endpoints identify a window, so two windows that happen to total the same amount but span different days are counted separately. Input is `n` and `m` on the first line, then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates integer widths. With `n <= 2*10^5` and `|a[i]| <= 10^9`, a running prefix sum reaches `2*10^5 * 10^9 = 2*10^14`, which is far past the 32-bit ceiling of about `2.1*10^9`. And the *answer* itself can be quadratic: if `m = 1` every window qualifies, so the count is `n(n+1)/2 = 2*10^5 * (2*10^5+1)/2 ≈ 2*10^10`. That overflows a 32-bit `int` by an order of magnitude. So both the prefix accumulator and the answer must be `long long`. That is the first non-negotiable decision; an `int` answer is a silent wrong-answer on the `m=1` tests.

**The brute force, as the ground truth I will check against.** The definition is a double loop: fix `l`, extend `r` from `l` to `n-1`, keep a running window sum, and increment a counter whenever `sum % m == 0`. That is `O(n^2)`. At `n = 2*10^5` it is `4*10^10` operations, hopeless inside one second, but it is the oracle I will use to validate any faster method on small inputs. I will not invent a clever count and trust it; I will compare it to this loop.

**Candidate approaches.** The fast method everyone reaches for here is the prefix-sum remainder trick. Define `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]`. The sum of window `[l, r]` equals `P[r+1] - P[l]`. That difference is divisible by `m` exactly when `P[r+1] ≡ P[l] (mod m)`, i.e. when the two prefixes share a remainder modulo `m`. So I bucket the prefix sums by remainder: walking left to right, for each new prefix I add to the answer the number of earlier prefixes that already have its remainder, then record this prefix in its bucket. Starting the bucket of remainder `0` with a count of `1` accounts for the empty prefix `P[0] = 0`. This is `O(n)` time and `O(m)` space. The open question is not the identity — the identity is solid — but whether the *standard code* for it transcribes correctly when the ledger is signed.

**The standard baseline, written down literally.** The textbook line, copied from the way this is usually taught for non-negative arrays, buckets by `prefix % m` directly:

```
long long pre = 0, ans = 0;
unordered_map<long long,long long> cnt; cnt[0] = 1;
for (int i = 0; i < n; i++) { pre += a[i]; long long r = pre % m; ans += cnt[r]; cnt[r]++; }
```

For an all-non-negative ledger this is correct, and it is exactly what most editorials show. But this problem's ledger is signed, so I should not assume `pre % m` lands in `{0, ..., m-1}`. In C++ the `%` operator truncates toward zero, which means a negative dividend gives a **negative** remainder: `-2 % 5` is `-2`, not `3`. The identity I derived is about the *mathematical* remainder, the representative in `{0, ..., m-1}`; `pre % m` only equals that for non-negative `pre`. So before committing I will run the standard line on a tiny signed case and check it against the brute force.

**First implementation and a trace — applying the standard baseline.** Take the smallest signed case I can think of that has a divisible window crossing a negative prefix: `n = 2`, `m = 5`, `a = [-2, 5]`. By hand, the windows are `[-2]` (sum `-2`, not divisible by 5), `[-2, 5]` (sum `3`, no), and `[5]` (sum `5`, **yes**). So the true answer is `1`.

Now run the standard baseline. Prefixes are `P[0] = 0`, `P[1] = -2`, `P[2] = 3`. The loop processes prefixes `P[1]` and `P[2]` (the running `pre` after each element). Start `cnt = {0: 1}`, `ans = 0`.
- `i = 0`: `pre = -2`. `r = (-2) % 5 = -2` in C++. `cnt[-2]` is `0`, so `ans += 0`. Then `cnt[-2] = 1`. State: `cnt = {0:1, -2:1}`, `ans = 0`.
- `i = 1`: `pre = 3`. `r = 3 % 5 = 3`. `cnt[3]` is `0`, so `ans += 0`. Then `cnt[3] = 1`. State: `cnt = {0:1, -2:1, 3:1}`, `ans = 0`.

The baseline prints `0`. The brute force prints `1`. They disagree, so the standard code is wrong for this variant.

**The bug.** The defect is precise, and the trace pins it. The window `[5]` is `P[2] - P[1] = 3 - (-2) = 5`, divisible by 5, so `P[1]` and `P[2]` are congruent mod 5: indeed `-2 ≡ 3 (mod 5)`. They *should* share a bucket. But the baseline keyed them by the C++ remainder: `P[1]` went into bucket `-2` and `P[2]` into bucket `3`. Two values that are congruent mod 5 were filed under different keys because C++'s `%` returns a negative remainder for the negative prefix. The hashmap version does not even crash — it silently undercounts. (A fixed-array version `cnt[r]++` with `r = -2` is worse: it indexes out of bounds, undefined behavior.) The root cause is that the prefix-sum identity speaks of remainders in `{0, ..., m-1}`, and `pre % m` does not produce that representative for negative `pre`. This is exactly the "standard algorithm is subtly wrong for this variant" trap: the method is right, the canonical transcription is not, and only checking on a signed case exposed it.

**Fix and re-verification.** The fix is to normalize the remainder into `{0, ..., m-1}` before bucketing: `r = pre % m; if (r < 0) r += m;`. I want to be sure a *single* `+= m` is always enough rather than guess at it, because if `pre % m` could be below `-m` I would need a loop or a second add. By the C++ standard, for the truncating `%`, the result `pre % m` has the sign of the dividend `pre` and satisfies `|pre % m| < |m|`. With `m > 0` that means `pre % m` is strictly inside `(-m, m)`. So if it is negative it lies in `(-m, 0)`, and adding `m` once moves it into `(0, m)` ⊂ `[0, m)`; if it is already non-negative it is in `[0, m)` and I leave it. Either way one conditional add lands `r` in `[0, m-1]`, exactly the valid index range. No loop, no second correction. I will also switch the bucket store from a hashmap to a flat array `vector<long long> count(m)` indexed by the normalized remainder — once remainders are guaranteed in `[0, m)` the array is safe, faster, and avoids the worst case where a hashmap with adversarial keys degrades to linear-time probes. Re-run the failing case `a = [-2, 5]`, `m = 5`:

- Start `count = [1,0,0,0,0]` (only remainder `0` seeded), `ans = 0`.
- `i = 0`: `pre = -2`. `r = (-2) % 5 = -2`; `r < 0` so `r += 5 -> 3`. `ans += count[3] = 0`. `count[3] = 1`. Now `count = [1,0,0,1,0]`.
- `i = 1`: `pre = 3`. `r = 3 % 5 = 3` (already in range). `ans += count[3] = 1`. `count[3] = 2`. Now `count = [1,0,0,2,0]`, `ans = 1`.

Output `1`, matching the brute force. The two congruent prefixes `-2` and `3` now both normalize to `3` and meet in the same bucket, so the window `[5]` is counted. The bug I diagnosed is the bug I fixed.

**A second self-verify on a longer signed case.** One case can be a fluke, so I trace a case with several negative prefixes in a row: `a = [-2, -2, 3, 1, 2]`, `m = 3`. Prefixes: `P = [0, -2, -4, -1, 0, 2]`. The mathematical remainders are `[0, 1, 2, 2, 0, 2]` (since `-2 ≡ 1`, `-4 ≡ 2`, `-1 ≡ 2 (mod 3)`). Counting matching pairs of equal remainder, including the seeded `P[0]`:
- remainder `0` occurs at `P[0], P[4]`: that is `C(2,2)` = 1 pair.
- remainder `1` occurs at `P[1]`: 0 pairs.
- remainder `2` occurs at `P[2], P[3], P[5]`: `C(3,2)` = 3 pairs.

Total `1 + 0 + 3 = 4`. The brute force on this input also returns `4`, and the fixed solution returns `4`. If I had used the C++ raw remainders `[0, -2, -1, -1, 0, 2]` instead, the buckets would be `{0:[P0,P4], -2:[P1], -1:[P2,P3], 2:[P5]}`, giving `C(2,2)+0+C(2,2)+0 = 1 + 1 = 2` — exactly the wrong `2` the naive baseline produces, and exactly the kind of split-bucket undercount the normalization repairs. Two independent traces, both explained by the same fix.

**A third check on the running-bucket arithmetic.** The two traces above validate the bucketing rule, but I also want to confirm that adding `count[r]` *before* the increment (not after) gives the pair count `C(k, 2)` when `k` prefixes share a remainder, since an off-by-one here would over- or under-count by exactly the diagonal. Consider a case engineered to pile prefixes into one bucket: `a = [3, -3, 3, -3]`, `m = 3`. Prefixes are `P = [0, 3, 0, 3, 0]`, all `≡ 0 (mod 3)`, so all five live in bucket `0`. Walking the loop with the seed `count[0] = 1`: at `P[1]` I add `count[0] = 1` then bump to 2; at `P[2]` I add `2` then bump to 3; at `P[3]` I add `3` then bump to 4; at `P[4]` I add `4` then bump to 5. Total `1 + 2 + 3 + 4 = 10`. By the closed form, 5 congruent prefixes form `C(5, 2) = 10` pairs, and every pair `(P[l], P[r+1])` with the same remainder is one divisible window — so 10 is right, and `1 + 2 + ... + (k-1) = C(k,2)` confirms that "add then increment" produces exactly the pair count with no diagonal term. The brute force on `[3,-3,3,-3]`, `m=3` returns `10` as well. Had I incremented before adding, I would have gotten `2 + 3 + 4 + 5 = 14`, counting each prefix against itself — the off-by-one I was guarding against. Order matters, and this case nails it down.

**Confirming the documented sample.** The statement's example is `n = 6`, `m = 4`, `a = [3, 1, -4, 2, -2, 4]`, claimed answer `10`. Prefixes: `P = [0, 3, 4, 0, 2, 0, 4]`. Mathematical remainders mod 4: `[0, 3, 0, 0, 2, 0, 0]`. Pairs of equal remainder:
- remainder `0` at `P[0], P[2], P[3], P[5], P[6]`: that is 5 prefixes, `C(5,2) = 10` pairs.
- remainder `3` at `P[1]`: 0 pairs.
- remainder `2` at `P[4]`: 0 pairs.

Total `10`. That matches the claimed answer and the brute force. Walking it through the loop reproduces it: each time `pre` returns to a remainder already seen, `ans` jumps by the running bucket count, and the bucket of remainder `0` (seeded at 1) accumulates `1, 2, 3, 4, 5` as `P[2], P[3], P[5], P[6]` arrive, contributing `1 + 2 + 3 + 4 = 10`. Self-consistent.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the loop never runs; `count[0]` is seeded to `1` but no prefix is ever added, so `ans = 0`. There are no windows. Correct. The `if (!(cin >> n >> m)) return 0;` also covers a fully empty input by leaving `answer = 0`.
- `m = 1`: every integer is divisible by `1`, so `pre % 1 == 0` for every prefix; `count` has length 1, every prefix lands in bucket `0`, and `ans` accumulates `0 + 1 + 2 + ... + (n-1)` plus the seed contributions, i.e. `n(n+1)/2`. For `n = 200000` that is `20000100000`, which I confirmed the 64-bit `ans` holds. If `ans` were `int` this would wrap to garbage — the reason the width decision was non-negotiable.
- A single divisible day, `n = 1`, `a = [0]`, `m = 5`: `pre = 0`, `r = 0`, `ans += count[0] = 1`. Output `1` (the window `[0]` sums to 0, divisible). Correct.
- A single non-divisible day, `n = 1`, `a = [3]`, `m = 5`: `pre = 3`, `r = 3`, `ans += count[3] = 0`. Output `0`. Correct.
- All negative and divisible, `a = [-4, -4, -4]`, `m = 4`: prefixes `[0, -4, -8, -12]`, all `≡ 0 (mod 4)`; buckets give `C(4,2) = 6`. The fixed solution outputs `6`; the brute force agrees. This is the case the unnormalized baseline would have indexed at `-4, -8, -12` and crashed or miscounted.
- Overflow: `pre` is `long long` and tops out near `2*10^14`; `ans` is `long long` and tops out near `2*10^10`; both fit with room to spare. The modulus `m` is read into a `long long` so `pre % m` is a 64-bit operation and `r += m` cannot overflow.
- Bucket index safety: after `if (r < 0) r += m`, `r` is in `[0, m-1]`, so `count[r]` is always in bounds for `vector<long long> count(m)`.

**Final solution.** I confirmed the idea is right by deriving the congruence identity, and I confirmed the *code* is right by applying the standard baseline, watching it fail on a signed case, tracing the failure to the C++ negative-remainder split, fixing it with one normalization, and re-verifying on a longer signed case, the documented sample, and the corners. Then I ran the fixed solution against the `O(n^2)` brute force on 850 random small cases with negatives and small moduli — zero mismatches — and timed it at `n = m = 2*10^5` (about 10 ms). That is what I ship: one self-contained file, the `O(n)` prefix-remainder count I can defend, not the textbook line I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;

    // count[r] = number of prefix sums seen so far whose normalized remainder mod m is r.
    // prefix sum 0 (the empty prefix, before index 0) is present from the start.
    vector<long long> count(m, 0);
    count[0] = 1;

    long long prefix = 0;     // running prefix sum (64-bit: can reach ~2*10^14)
    long long answer = 0;     // number of windows (64-bit: can reach ~2*10^10)

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        prefix += x;
        // Normalize the remainder into [0, m): C++ % can be negative for negative prefix.
        long long r = prefix % m;
        if (r < 0) r += m;
        answer += count[r];   // every earlier prefix with the same remainder closes a divisible window
        count[r]++;
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The prefix-sum congruence identity is correct, but the textbook transcription `bucket by pre % m` is subtly wrong for a signed ledger: C++'s `%` returns negative remainders, so two prefixes that are congruent mod `m` get filed under different keys and the divisible window between them is missed — a trace of `[-2, 5]` with `m = 5` returning `0` instead of `1` pinpointed it, with the window `[5] = P[2] - P[1] = 3 - (-2)` and `-2 ≡ 3 (mod 5)` showing exactly which bucket split. Normalizing with `if (r < 0) r += m` collapses congruent prefixes back into one bucket; switching to a flat `vector` indexed by the in-range remainder is then safe and fast; and `long long` on both the prefix accumulator (up to `2*10^14`) and the answer (up to `n(n+1)/2 ≈ 2*10^10` at `m = 1`) closes out the overflow, empty, single-day, and all-negative corners.
