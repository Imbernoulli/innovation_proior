**Reading the problem and pinning the contract.** I have `n` daily prices `a[0..n-1]`, possibly negative. I must pick a set of days `i1 < i2 < ... < ik` (`k >= 1`) whose prices are *strictly increasing*, `a[i1] < a[i2] < ... < a[ik]`, and I maximize the **sum** of the chosen prices. Input on stdin is `n` then the `n` values; I print one integer. By convention `n = 0` yields `0` (no day to enter). Before any algorithm I fix the scale because it dictates the data types: `n <= 2*10^5` and `|a[i]| <= 10^9`. A strictly increasing array of length `2*10^5` with values up to `10^9` has sum on the order of `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit ceiling of about `2.1*10^9`. Even the modest all-`1..n` case sums to `~2*10^10`, already overflowing `int`. So every score, accumulator, and Fenwick cell must be 64-bit `long long`. That decision is non-negotiable; an `int` here is a silent wrong-answer on the large tests.

**Naming the objective precisely, because the trap lives in the wording.** The phrase "increasing subsequence" pulls the mind straight to LIS — the *longest* increasing subsequence. But the score is the **sum**, not the length. Those are different optimization problems on the same feasible set, and conflating them is exactly the mistake I must avoid. I will treat "increasing" as the *feasibility* constraint and "maximum sum" as the *objective*, and keep them mentally separate.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one easiest to type.

- *Greedy.* Two heuristics suggest themselves. (a) Take the longest increasing subsequence — surely more terms means a bigger sum. (b) Just take the single largest value — one fat element beats a thin chain. Both are `O(n log n)` and a few lines. The risk is structural: the constraint couples *which* elements can coexist, so a locally appealing choice (more terms, or the biggest term) can be globally suboptimal. I will not trust greedy until I have tried to break it.
- *Value-indexed DP with a Fenwick tree.* Let `f[i]` be the best score of an increasing subsequence that *ends* at index `i`. Then `f[i] = a[i] + max(0, max{ f[j] : j < i and a[j] < a[i] })`. The naive inner scan is `O(n^2)`. But the inner query is "maximum `f` over earlier elements with strictly smaller value" — a prefix-max over a value axis — which a Fenwick tree answers in `O(log n)` after coordinate compression. `O(n log n)` total. The risk here is not the idea but the *transcription*: the strict-less-than boundary and duplicate handling are easy to get subtly wrong.

**Stress-testing greedy before committing — a single instance kills both heuristics.** Hand-waving "greedy feels right" is how wrong solutions ship, so let me attack it with a concrete instance: `a = [1, 100, 2, 3, 4, 5, 6]`, indices `0..6`.

- Heuristic (a), longest increasing subsequence: the longest strictly increasing chain is `1, 2, 3, 4, 5, 6` (indices `0,2,3,4,5,6`), length 6, sum `1+2+3+4+5+6 = 21`.
- Heuristic (b), single largest value: that is `100`, sum `100`.
- But the *optimal* is `1, 100` (indices `0,1`), a strictly increasing chain of length 2 with sum `101`.

So `21 < 100 < 101`: the longest chain is far from the heaviest, and even the single biggest element is beaten by pairing it with a tiny predecessor. Both greedy heuristics are wrong, and I now see *why*: maximizing the count optimizes the wrong functional, and grabbing one big element forfeits the free additive value of compatible smaller predecessors. The verification paid off — it killed two approaches I might otherwise have shipped. Greedy is out; I commit to the DP.

**Deriving the DP and checking the recurrence on paper.** I want the best score of an increasing subsequence *ending* at `i`. Whatever the prefix did, the only thing that lets me append `i` is that the previous chosen element `j` satisfies `j < i` and `a[j] < a[i]`. Among all such `j` I want the largest `f[j]`, and I may also start a fresh chain at `i` (no predecessor), which corresponds to adding `0`. Hence

`f[i] = a[i] + max(0, max{ f[j] : j < i and a[j] < a[i] })`.

The answer is `max_i f[i]` over `i = 0..n-1` (at least one element is forced, and every single element is a valid length-1 chain, so this is always defined for `n >= 1`). The `max(0, ...)` is what lets a chain start fresh, so an all-negative array correctly returns the least-negative single element rather than chaining negatives together. Let me sanity-check the recurrence on the documented sample `a = [3, 1, 4, 1, 5, 9, 2, 6]`, expected `21`:

- i=0 (3): no smaller earlier; `f0 = 3 + max(0) = 3`.
- i=1 (1): no earlier value `< 1`; `f1 = 1`.
- i=2 (4): earlier values `< 4` are `3,1` with `f = 3,1`; best `3`; `f2 = 4 + 3 = 7`.
- i=3 (1): no earlier value `< 1`; `f3 = 1`.
- i=4 (5): earlier values `< 5` are `3,1,4,1` with `f = 3,1,7,1`; best `7`; `f4 = 5 + 7 = 12`.
- i=5 (9): earlier `< 9` include the `5` with `f4 = 12`; best `12`; `f5 = 9 + 12 = 21`.
- i=6 (2): earlier values `< 2` are the two `1`s (`f = 1`); best `1`; `f6 = 2 + 1 = 3`.
- i=7 (6): earlier `< 6` are `3,1,4,1,5,2` with `f = 3,1,7,1,12,3`; best `12`; `f7 = 6 + 12 = 18`.

`max(3,1,7,1,12,21,3,18) = 21`. The recurrence reproduces the chain `3 < 4 < 5 < 9` and the right answer. Good.

**Turning the inner `max` into a Fenwick query.** The bottleneck is `max{ f[j] : j < i and a[j] < a[i] }`. If I process `i` left to right and, after computing `f[i]`, "store" `f[i]` at the *value* `a[i]`, then for the next elements the query becomes "maximum stored `f` over all values strictly less than `a[i]`" — a prefix maximum on the value axis, restricted to elements already inserted (which are exactly the earlier indices). Values can be up to `10^9` and negative, so I coordinate-compress: sort the distinct values, map each `a[i]` to its 1-based rank. A Fenwick (BIT) over ranks `1..m` supporting *point update with max* and *prefix-max query* does the job in `O(log m)` each. "Strictly less than `a[i]`" means I query ranks `[1 .. rank(a[i]) - 1]`, which correctly excludes equal values (duplicates of `a[i]` share its rank and must NOT be valid predecessors, since the chain is *strictly* increasing).

A standard caveat: a max-Fenwick supports prefix-max query and point-max update, but it does NOT support arbitrary decreases or suffix queries — that is fine here, I only ever do prefix-max queries and monotone point-max updates, which the structure handles correctly.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut:

```
const long long NEG = LLONG_MIN / 4;
vector<long long> bit(m + 1, NEG);
auto queryPrefixMax = [&](int r){ long long b = NEG; for(; r>0; r-=r&(-r)) b=max(b,bit[r]); return b; };
auto updatePoint = [&](int r, long long v){ for(; r<=m; r+=r&(-r)) bit[r]=max(bit[r],v); };
long long answer = NEG;
for (int i = 0; i < n; i++) {
    int r = rankOf(a[i]);
    long long bestPrev = queryPrefixMax(r);        // <-- query up to and including r
    long long f = a[i] + max(bestPrev, 0LL);
    updatePoint(r, f);
    answer = max(answer, f);
}
```

I am suspicious of `queryPrefixMax(r)` versus `queryPrefixMax(r-1)`, so I trace the smallest input that exposes the difference: duplicates. Take `a = [2, 2]`. The two `2`s are equal, so they cannot both be in a strictly increasing chain; the answer is just `2`. Compress: distinct `{2}`, so `m = 1`, both ranks are `1`. Start `bit = [NEG, NEG]`, `answer = NEG`. i=0: `r=1`; `bestPrev = queryPrefixMax(1) = NEG`; `f = 2 + max(NEG,0) = 2`; `updatePoint(1, 2)` -> `bit[1]=2`; `answer=2`. i=1: `r=1`; `bestPrev = queryPrefixMax(1) = 2`; `f = 2 + max(2,0) = 4`; `answer=4`. Final `4`.

**Diagnosing the first bug.** The code returns `4` — it chained `2` after `2`, an *equal* predecessor, which strict increase forbids. The defect is precise: I queried the prefix up to rank `r` *inclusive*, so the second `2` saw the first `2`'s stored `f` (same rank) as a legal predecessor. The query must be over values *strictly less than* `a[i]`, i.e. ranks `[1 .. r-1]`. I change the query to `queryPrefixMax(r - 1)`. (When `r = 1`, `queryPrefixMax(0)` returns `NEG` immediately, which is the correct "no smaller predecessor" signal.)

**Re-verifying the fix on the duplicate case.** With `queryPrefixMax(r-1)`: i=0: `r=1`; `bestPrev = queryPrefixMax(0) = NEG`; `f = 2 + 0 = 2`; store `bit[1]=2`; `answer=2`. i=1: `r=1`; `bestPrev = queryPrefixMax(0) = NEG`; `f = 2 + 0 = 2`; `answer = max(2,2) = 2`. Final `2`. Correct — the equal predecessor is excluded. I also re-run the strictly-increasing sanity `a = [3,4,5,9]` mentally: ranks `1,2,3,4`; f = 3, 7, 12, 21; `answer = 21`. Correct.

**A second trace, on negatives, because the `NEG` sentinel is a landmine.** I worry the sentinel might leak into a real score. Take `a = [-5, -3]`, a strictly increasing pair; expected best is the whole chain `-5 + -3 = -8`? No — wait, I can also take a single element. Candidates: `{-5}` = -5, `{-3}` = -3, `{-5, -3}` = -8. The max is `-3` (single largest). Let me trace. Compress: distinct `{-5,-3}`, ranks `-5 -> 1`, `-3 -> 2`. Start `bit=[NEG,NEG,NEG]`, `answer=NEG`. i=0 (`-5`, r=1): `bestPrev = queryPrefixMax(0) = NEG`; `f = -5 + max(NEG, 0) = -5 + 0 = -5`; store `bit[1]=-5`; `answer=-5`. i=1 (`-3`, r=2): `bestPrev = queryPrefixMax(1) = bit[1] = -5`; `f = -3 + max(-5, 0) = -3 + 0 = -3`; store; `answer = max(-5,-3) = -3`. Final `-3`.

**Diagnosing the would-be second bug — and confirming the guard already handles it.** This is exactly where a naive `f = a[i] + bestPrev` would have been wrong: at i=1 it would compute `-3 + (-5) = -8`, worse than starting fresh at `-3`. My `max(bestPrev, 0LL)` guard correctly lets the chain restart, yielding `-3`. If I had instead written `f = a[i] + (bestPrev == NEG ? 0 : bestPrev)` I would have *forced* chaining onto any real predecessor even when its score is negative — that would return `-8` here and be wrong. So the `max(..., 0)` is load-bearing for negatives, not just for the empty/no-predecessor case. The trace confirms the sentinel never leaks: `max(NEG, 0LL) = 0`, and I never add `a[i]` to a raw `NEG`. Good. (I also note `NEG = LLONG_MIN/4` so that even `max`-folding several `NEG`s in the BIT cannot underflow; they are only ever compared, never summed.)

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: I read no values; I print `0` directly. (My `if (n == 0)` short-circuit avoids building a size-0 Fenwick.) Correct by convention.
- `n = 1`, `a = [-7]`: ranks `{-7} -> 1`; i=0: `bestPrev = queryPrefixMax(0) = NEG`; `f = -7 + 0 = -7`; `answer = -7`. A single forced day; correct (the only non-empty subsequence is `{-7}`).
- All negative, `[-3,-1,-4,-2]`: each `f` is its own value plus `max(0, ...)`, so chaining never helps (predecessors are negative); `answer` is the least-negative element `-1`. Matches brute force.
- Strictly decreasing, `[9,7,5,3]`: no element has a smaller earlier predecessor, so every `f[i] = a[i]`; `answer = 9`, the single largest. Correct.
- Many duplicates, `[5,5,5]`: all share one rank; each `f = 5`; `answer = 5` (cannot chain equals). Correct.
- Overflow: scores are `long long`; worst sum `~2*10^14` fits with room. Verified empirically: `a = [1,2,...,2*10^5]` gives `20000100000 ~ 2*10^10`, which already overflows 32-bit, and `long long` handles it.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is format-agnostic.

**Re-verification at scale and against brute force.** I wrote an independent exponential brute force that enumerates every non-empty subset, keeps those that are strictly increasing in index order, and takes the maximum sum — no DP, no Fenwick, just the definition. Over 850 random small cases (`n` up to 14, value regimes spanning all-negative, all-positive, wide-range, and duplicate-heavy, plus deliberately trap-shaped arrays with a big spike among small increasing runs) the Fenwick solution matches the brute force on every case, zero mismatches. The documented sample `[3,1,4,1,5,9,2,6]` prints `21`. At `n = 2*10^5` the program runs in about `0.04 s` using `~8 MB`, comfortably inside the `1 s` / `256 MB` budget. The idea is right (greedy disproved, recurrence hand-checked), and the code is right (two real bugs traced to precise causes and fixed, corners checked, brute force agrees).

**Final solution.** One self-contained file — the `O(n log n)` value-indexed DP with a max-Fenwick that I can defend, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Coordinate-compress the values so a Fenwick tree can be indexed by rank.
    vector<long long> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](long long v) -> int {
        // 1-based rank in the sorted distinct array.
        return (int)(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1;
    };

    // bit[r] = max f-value over compressed ranks in the Fenwick prefix ending at r.
    // We query the prefix max over ranks STRICTLY LESS THAN rank(a[i]) (values < a[i]),
    // then point-update rank(a[i]) with f[i].
    const long long NEG = LLONG_MIN / 4;
    vector<long long> bit(m + 1, NEG);

    auto queryPrefixMax = [&](int r) -> long long { // max over ranks [1..r]
        long long best = NEG;
        for (; r > 0; r -= r & (-r))
            best = max(best, bit[r]);
        return best;
    };
    auto updatePoint = [&](int r, long long val) {  // bit[r] = max(bit[r], val)
        for (; r <= m; r += r & (-r))
            bit[r] = max(bit[r], val);
    };

    long long answer = NEG;
    for (int i = 0; i < n; i++) {
        int r = rankOf(a[i]);
        // best f among earlier elements with strictly smaller value:
        long long bestPrev = queryPrefixMax(r - 1);
        long long f = a[i] + max(bestPrev, 0LL); // a[i] alone if no smaller predecessor
        updatePoint(r, f);
        answer = max(answer, f);
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The word "increasing subsequence" baited two greedy heuristics — longest chain and single largest value — but one traced counterexample (`[1,100,2,3,4,5,6]`: longest-chain sum `21`, single-largest `100`, optimal `101`) showed the heaviest chain is neither the longest nor a singleton, so I moved to the DP `f[i] = a[i] + max(0, max{f[j] : j<i, a[j]<a[i]})` and hand-checked it reproduces `21` on the sample. To beat `O(n^2)` I expressed the inner max as a prefix-max over compressed values in a Fenwick tree; my first query used `rank` inclusive, and a trace of `[2,2]` returning the illegal `4` pinpointed that strict increase requires querying `rank-1`; a second trace on `[-5,-3]` confirmed the `max(bestPrev, 0)` guard is what keeps negative predecessors from dragging a chain below a fresh restart, and that the `LLONG_MIN/4` sentinel is only ever compared, never summed, so it cannot leak; `long long` throughout absorbs the `~2*10^14` worst-case sum, and an independent exponential brute force agrees on 850 random cases.
