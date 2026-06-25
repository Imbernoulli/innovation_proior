**Reading the problem and pinning the contract.** I have a row of `n` containers with positive
weights `w[0..n-1]`. A left crane peels off a prefix of `i` containers, a right crane peels off a
suffix of `j` containers, the two ranges may not overlap (`i + j <= n`), and the *combined* weight
they lift must fit one shared fuel tank: `pref[i] + suf[j] <= B`, where `pref[i]` is the sum of the
first `i` weights and `suf[j]` is the sum of the last `j`. I want to maximize the count `i + j`.
Input is `n` and `B` on the first line, then the `n` weights; I print a single integer.

Before any algorithm I fix the data types, because they decide whether the code is even capable of
being correct. `n <= 2*10^5` and `w[i] <= 10^9`, so a prefix sum can reach `2*10^5 * 10^9 = 2*10^14`,
and `B` itself can be as large as `10^18`. Both blow past 32-bit (`~2.1*10^9`). So every weight,
every prefix/suffix sum, and `B` must be 64-bit `long long`. The *answer* is a count `<= n <= 2*10^5`,
which fits in `int`, but I will not risk mixing — I will compare and store it as `long long` and only
worry about narrowing at the very end. That is decision one and it is non-negotiable.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can
*prove*, not the one that types fastest.

- *Greedy from the cheaper end.* Two pointers at the ends; repeatedly lift whichever end's next
  container is lighter, as long as it fits the remaining fuel; stop when neither fits. It is `O(n)`,
  three lines, and intuitively appealing — to maximize *count* you want to spend the least fuel per
  container, so always grab the cheapest available. The risk is structural: the budget is *shared*
  and global, but this rule decides locally, which is exactly the configuration where greedy tends to
  betray you. I will not trust it until I have tried to break it.
- *Two-pointer sweep over prefix lengths.* Enumerate every affordable prefix length `i`; for each,
  take the largest suffix `j` that the leftover fuel `B - pref[i]` and the overlap bound `n - i`
  allow. If the best `j` moves monotonically as `i` changes, one sliding pointer makes this `O(n)`.
  The risk is not the idea but the *transcription* — the pointer direction and the overlap cap are
  easy to get subtly wrong.

**Stress-testing the greedy before committing.** "Cheapest-first feels optimal" is precisely how
wrong solutions get shipped, so let me actually attack it with a concrete instance instead of
hand-waving. Take `w = [3, 4, 6, 6, 3, 4]`, indices `0..5`, and a shared budget `B = 10`.

Run the cheaper-end greedy. Left end is `w[0] = 3`, right end is `w[5] = 4`; the left is lighter, so
lift it. Fuel used `= 3`, the left pointer advances. Now the left end is `w[1] = 4`, the right end is
still `w[5] = 4`; tie, take the left, fuel used `= 7`. Now the left end is `w[2] = 6` and the right
end is `w[5] = 4`; the right is lighter, but `7 + 4 = 11 > 10`, it does not fit; the left's `6` does
not fit either (`7 + 6 = 13`). Neither end fits, so greedy stops having removed **2** containers.

Is 2 optimal? Let me hunt for a configuration the greedy structurally could not reach. Take the left
crane's single container `w[0] = 3` (cost 3), and let the right crane take the last *two*,
`w[5] = 4` and `w[4] = 3` (cost `4 + 3 = 7`). Combined cost `3 + 7 = 10 <= B`, no overlap
(`1 + 2 = 3 <= 6`), and that is **3** containers — strictly more than greedy's 2. So the greedy is
wrong, and I can now see *why*: by eagerly burning `4` units on `w[1]` from the left it starved the
tank, and that extra-cheap container on the right (`w[4] = 3`) was reachable only by *not* having
spent on the left. The cheapest single step is not the cheapest path to a large count when one tank
feeds both ends. The verification paid off — it killed an approach I would otherwise have shipped.
Greedy from the cheaper end is out.

**Deriving the correct method and its monotonicity.** I switch to the sweep. Fix the prefix length
`i`; the prefix is forced to be exactly `w[0..i-1]` with cost `pref[i]`, so the only freedom left is
the suffix length `j`, and I want it as large as possible subject to two caps:

- *fuel*: `suf[j] <= B - pref[i]`, and
- *no overlap*: `j <= n - i`.

Both `pref` and `suf` are nondecreasing (weights are positive), so for fixed `i` the feasible `j`
form a prefix `0..J(i)` of suffix lengths, and `J(i)` is the largest `j` meeting both caps. The
answer is `max over affordable i of (i + J(i))`. Computing each `J(i)` by binary search would be
`O(n log n)`; I want `O(n)`, which needs a monotonicity I can defend.

Claim: `J(i)` is **non-increasing** as `i` increases by 1. Two things happen when `i` grows: `pref[i]`
grows (or stays, but weights are strictly positive so it strictly grows), shrinking the fuel cap
`B - pref[i]`; and `n - i` shrinks, tightening the overlap cap. A tighter cap can only lower the
largest feasible `j`, never raise it. So `J(i+1) <= J(i)`. That is exactly the condition for a single
pointer that starts large and only slides *inward* (downward) as `i` increases — total movement
`<= n`, giving `O(n)`.

Let me sanity-check the monotonicity numerically on the sample `w = [1, 7, 7, 8, 2, 8]`, `B = 11`.
Prefix sums `pref = [0, 1, 8, 15, 23, 25, 33]`; suffix sums `suf = [0, 8, 10, 18, 25, 32, 33]`
(reading from the right: `8, 2, 8, 7, 7, 1`). For `i = 0`, leftover `11`, largest `j` with
`suf[j] <= 11` and `j <= 6` is `j = 2` (`suf[2] = 10 <= 11`, `suf[3] = 18 > 11`). For `i = 1`,
leftover `10`, `suf[2] = 10 <= 10` so `j = 2`; cap `n - i = 5`, fine. For `i = 2`, leftover
`11 - 8 = 3`, `suf[1] = 8 > 3` so `j = 0`. So `J(0) = 2, J(1) = 2, J(2) = 0` — non-increasing, as
claimed. And the running max `i + J(i)` is `max(0+2, 1+2, 2+0) = 3`, matching the stated sample
answer `3`. Both the monotonicity and the formula check out on a concrete case.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first
cut of the sweep grows the suffix pointer `j` from `0`, extending it while it fits:

```
long long best = 0;
int j = 0;
for (int i = 0; i <= n; i++) {
    if (pref[i] > B) break;                 // larger prefixes only cost more
    while (j < n - i && suf[j + 1] <= B - pref[i]) j++;
    if (j > n - i) j = n - i;
    best = max(best, (long long)i + j);
}
```

Something about *growing* `j` while the feasible `j` is supposed to *shrink* with `i` looks
backwards, so I trace the smallest input that could expose it: `w = [2, 5]`, `B = 6`. Here
`pref = [0, 2, 7]`, `suf = [0, 5, 7]` (suffix order `5, 2`... wait, reading from the right the last
container is `w[1] = 5`, then `w[0] = 2`, so `suf = [0, 5, 7]`). The true optimum: with `B = 6` I can
afford at most one container of weight `5`, or the single `2`, so the best count is `1`.

Trace the buggy code. `i = 0`: `pref[0] = 0 <= 6`; grow `j` while `j < 2` and `suf[j+1] <= 6`:
`suf[1] = 5 <= 6` so `j = 1`; `suf[2] = 7 <= 6`? no. `j` stays `1`. `best = 0 + 1 = 1`. `i = 1`:
`pref[1] = 2 <= 6`; grow `j` while `j < n - i = 1`: but `j = 1` is not `< 1`, so the loop body never
runs and `j` stays `1`. The overlap cap `j > n - i = 1`? `1 > 1` is false, so no clamp. `best`
becomes `max(1, 1 + 1) = 2`.

**Diagnosing the first bug.** The code returns `2`, but `2` is infeasible: `i = 1` means the prefix
took `w[0] = 2`, `j = 1` means the suffix took `w[1] = 5`, and `pref[1] + suf[1] = 2 + 5 = 7 > B = 6`.
The defect is exactly the backwards direction I worried about. When `i` grew from `0` to `1`, the
leftover fuel dropped from `6` to `4`, so the feasible suffix length should have *dropped* from `1`
to `0` (since `suf[1] = 5 > 4`). But my pointer only ever *grows*; it had no machinery to walk `j`
back down when the fuel cap tightened, so it kept the stale `j = 1` from the previous iteration and
counted an over-budget pair. A grow-only pointer is correct only when the feasible bound is
non-decreasing; here it is non-increasing, so the pointer must slide the other way.

**Fixing and re-verifying the first bug.** I flip the pointer: start `j = n` (the largest it could
ever be) and, each iteration, first clamp it to the overlap bound `n - i`, then *shrink* it while the
fuel cap is violated:

```
long long best = 0;
int j = n;
for (int i = 0; i <= n; i++) {
    if (pref[i] > B) break;
    if (j > n - i) j = n - i;                   // respect no-overlap first
    while (j > 0 && suf[j] > B - pref[i]) j--;   // shrink until the suffix fits
    best = max(best, (long long)i + j);
}
```

Re-trace `w = [2, 5]`, `B = 6`, with `pref = [0, 2, 7]`, `suf = [0, 5, 7]`, start `j = 2`. `i = 0`:
`pref[0] = 0`; clamp `j` to `n - 0 = 2` (no change); shrink while `suf[j] > 6`: `suf[2] = 7 > 6` so
`j = 1`, `suf[1] = 5 > 6`? no. `best = 0 + 1 = 1`. `i = 1`: `pref[1] = 2 <= 6`; clamp `j` to
`n - 1 = 1` (no change, `j = 1`); shrink while `suf[1] = 5 > B - pref[1] = 4`: yes, `j = 0`.
`best = max(1, 1 + 0) = 1`. `i = 2`: `pref[2] = 7 > 6`, break. Answer `1`. Correct, and the pointer
moved `2 -> 1 -> 0` monotonically downward, never back up — the amortized-`O(n)` guarantee holds.
The case that broke before now passes, and it passes for the reason I fixed.

**Second trace — the overlap cap is load-bearing, so I attack it directly.** The fix above *has* a
`if (j > n - i) j = n - i;` line, but let me make sure I actually need it and that omitting it is a
real, not theoretical, bug. Consider the regime where fuel is plentiful so the budget never bites and
only the overlap can stop double-counting: `w = [1, 1, 1, 1]`, `B = 100`. The true answer is `4` —
even if both cranes were greedy, there are only `4` containers and `i + j <= n = 4`.

Trace the same loop but with the overlap clamp **deleted**, start `j = 4`. `i = 0`: shrink while
`suf[j] > 100`: `suf[4] = 4 <= 100`, no shrink, `j = 4`; `best = 0 + 4 = 4`. `i = 1`:
`pref[1] = 1 <= 100`; *no* overlap clamp; shrink while `suf[4] = 4 > 99`? no; `j` stays `4`;
`best = max(4, 1 + 4) = 5`. `i = 2`: `best = max(5, 2 + 4) = 6`. Continuing, `i = 4` gives
`4 + 4 = 8`. The code reports `8` — it counted all four containers as the prefix *and* all four again
as the suffix, removing each twice. Without the `j <= n - i` clamp the prefix and suffix silently
overlap whenever the budget is loose, doubling the count. So the overlap cap is not decoration; it is
the only thing enforcing `i + j <= n`. With the clamp restored, `i = 1` first sets `j = min(4, 3) = 3`,
giving `1 + 3 = 4`, and every `i` yields exactly `4`, so the answer is `4` — correct. Two independent
bugs, two concrete traces, two precise fixes.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the loop runs only `i = 0`; `pref[0] = 0 <= B`; `j` clamps to `min(n, n - 0) = 0`; nothing
  to shrink; `best = 0`. Nothing to remove — correct.
- `n = 1`, `w = [3]`, `B = 5`: `i = 0` -> `j` clamps to `1`, `suf[1] = 3 <= 5`, `best = 1`;
  `i = 1` -> `pref[1] = 3 <= 5`, `j` clamps to `0`, `best = max(1, 1) = 1`. Answer `1` — the one
  container removed by exactly one crane (the prefix-vs-suffix tie is harmless because the count is
  the same). With `B = 2 < 3`: `i = 0` -> shrink `j` from `1` to `0` (`suf[1] = 3 > 2`); `i = 1` ->
  `pref[1] = 3 > 2`, break. Answer `0` — correct.
- `B = 0`: every `pref[i]` for `i >= 1` is `>= w[0] >= 1 > 0`, so the loop breaks immediately after
  `i = 0`, where `j` shrinks to `0`. Answer `0` — remove nothing, correct.
- `B >= total weight`: the budget never bites, so only the overlap cap matters; for every `i` the
  suffix takes the full `n - i`, and `i + (n - i) = n` is reported. Answer `n` — remove everything,
  correct (this is exactly the `[1,1,1,1]` case above).
- Optimum entirely on one side: if the right end is too heavy to ever afford, every `J(i)` is `0` and
  the answer is the largest affordable prefix length `pmax`; the sweep finds it at `i = pmax` with
  `j = 0`. Symmetrically `i = 0` with a large `j` covers an all-suffix optimum. Both extremes are in
  the sweep's range `i = 0..pmax`.
- Overflow: `pref`, `suf`, and `B` are `long long`; the largest sum `~2*10^14` and `B <= 10^18` both
  fit with room. I never add anything to a sentinel; `B - pref[i]` is computed only after checking
  `pref[i] <= B`, so it is `>= 0` and cannot underflow. Safe.
- Output: a single integer and newline; `cin >>` skips arbitrary whitespace, so the input format is
  forgiving and the empty second line at `n = 0` is fine.

**Final solution.** I convinced myself the *idea* is right by disproving the cheaper-end greedy with
a concrete counterexample (`[3,4,6,6,3,4]`, `B = 10`: greedy 2 vs the reachable 3) and by proving and
numerically checking the monotonicity that makes one sliding pointer valid; and I convinced myself the
*code* is right by tracing two failing variants to precise causes — a backwards (grow-only) pointer
that counted an over-budget pair on `[2,5]`, and a missing overlap clamp that double-counted on
`[1,1,1,1]` — and re-verifying each fix and every corner. That is what I ship: one self-contained
`O(n)` two-pointer sweep, the method I can defend rather than the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // Left crane removes a prefix of i containers; right crane removes a suffix
    // of j containers. They share one fuel budget B, so pref[i] + suf[j] <= B,
    // and they must not overlap, so i + j <= n. Maximize i + j.
    //
    // pref[i] = sum of first i containers; suf[j] = sum of last j containers.
    vector<long long> pref(n + 1, 0);
    for (int i = 0; i < n; i++) pref[i + 1] = pref[i] + w[i];
    vector<long long> suf(n + 1, 0);
    for (int j = 0; j < n; j++) suf[j + 1] = suf[j] + w[n - 1 - j];

    // Sweep i (prefix count) upward over every affordable prefix. For each i the
    // suffix may use at most n - i containers and at most B - pref[i] fuel. As i
    // increases, pref[i] grows so the suffix budget shrinks; the overlap bound
    // n - i also shrinks. Hence the largest affordable suffix count j is
    // non-increasing in i, so a single pointer sliding inward gives O(n) total.
    long long best = 0;
    int j = n; // largest suffix count we will ever consider; shrinks as i grows
    for (int i = 0; i <= n; i++) {
        if (pref[i] > B) break;            // no longer affordable; larger i is worse
        if (j > n - i) j = n - i;          // respect the no-overlap bound
        while (j > 0 && (suf[j] > B - pref[i])) j--; // shrink until suffix fits
        best = max(best, (long long)i + j);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The cheaper-end greedy looked optimal but a single traced counterexample
(`[3,4,6,6,3,4]`, `B = 10`: greedy 2 vs the reachable 3) showed that burning fuel on the locally
cheapest container starves the shared tank and loses count, so I moved to a sweep over prefix lengths
where, for each affordable prefix `i`, the best suffix `J(i)` is non-increasing in `i` (proved from
`pref` rising and `n - i` falling, and checked numerically on `[1,7,7,8,2,8]`); that monotonicity
lets one pointer slide inward in `O(n)`, but my first cut grew the pointer the wrong way and a trace
of `[2,5]`, `B = 6` returning the over-budget `2` pinned it, while a second trace of `[1,1,1,1]`,
`B = 100` returning `8` showed the `j <= n - i` overlap clamp is the only guard against
double-counting; flipping the pointer to shrink and restoring the clamp, over 64-bit accumulators,
closes the budget, overlap, empty, single-element, and overflow corners.
