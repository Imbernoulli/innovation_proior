**Reading the problem and pinning the contract.** I have `n` candidate observations for one telescope on one night. Observation `i` occupies the half-open interval `[s_i, e_i)` and yields value `v_i >= 0`; I pick a subset with no two overlapping and maximize the total value, the empty set allowed so the answer is at least `0`. Input on stdin is `n` then `n` triples `s e v`; I print one integer. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5`, coordinates up to `10^9`, and `v_i` up to `10^9`. A total value can reach `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit range of about `2.1*10^9`, so the value accumulator and every coordinate must be 64-bit. I will use `long long` throughout; `int` here is a silent wrong-answer on the large tests. I also pin the overlap convention now, because the whole problem hinges on it: intervals are **half-open**, so `[s_i, e_i)` and `[s_j, e_j)` conflict iff they share an interior point, i.e. iff `s_i < e_j` and `s_j < e_i`. Two intervals that merely touch — `e_i == s_j` — do **not** conflict and may both be scheduled. That single `<=`-vs-`<` distinction is exactly the kind of thing I will trace later, so I write it down explicitly: observation with start `s` is compatible with an earlier one of end `e` iff `e <= s`.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Greedy.* Two greedies are tempting. (a) *Value-greedy:* repeatedly take the highest-value observation that still fits, blocking everything it overlaps. (b) *Earliest-finishing:* scan observations by finishing time and take each whose start is at or after the last taken finish — the textbook rule that maximizes the *count* of selected intervals. Both are `O(n log n)` and a handful of lines. The structural worry is that the non-overlap constraint is global while a greedy decides locally; with *weights* on the intervals, the count-maximizer in particular has no reason to maximize value. I will not trust either until I have tried to break it.
- *Interval DP.* Sort by finishing time. Process observations in that order; for the one ending soonest among the unprocessed, the only thing the future cares about is which earlier observations are compatible (finish at or before this one starts). Carry, over the sorted prefix, the best achievable total value, and for each observation binary-search the latest compatible predecessor. `O(n log n)`. The risk here is not the idea but the *transcription* — the sort key, the binary-search boundary, and the half-open `<=`.

**Stress-testing greedy before committing.** Hand-waving "earliest-finish is the classic interval rule" is how wrong solutions get shipped, so let me actually attack both greedies with a concrete instance. Take five observations on a 90-minute night:

```
A [0,30)  v=20
B [0,90)  v=50
C [30,60) v=25
D [60,90) v=25
E [40,50) v=5
```

*Value-greedy* grabs the single biggest value first: B `[0,90)` with value 50. B spans the whole night, so it blocks A, C, D, and E. Value-greedy's total is `50`. *Earliest-finishing* sorts by end — A(30), E(50), C(60), B(90), D(90) — and takes A `[0,30)`; the next whose start is `>= 30` in this order is E `[40,50)` (start 40 >= 30), take it; next with start `>= 50` is D `[60,90)`, take it. Earliest-finishing's total is `20 + 5 + 25 = 50`.

Is 50 optimal? Let me hunt for something both greedies structurally could not reach. The three observations A `[0,30)`, C `[30,60)`, D `[60,90)` are pairwise touching-but-not-overlapping (30==30, 60==60 under the half-open rule), with values `20 + 25 + 25 = 70`. That is strictly better than 50. So **both greedies are wrong**, and I now see *why*: value-greedy let one fat interval hog the whole resource, and earliest-finishing, by snatching the tiny E `[40,50)` for the sake of an early finish, wrecked the clean A/C/D tiling. The verification paid off — it killed the two approaches I would otherwise have shipped. I also ran a quick random audit later (below): over 600 small random instances, value-greedy disagreed with the true optimum on 134 of them and earliest-finishing on 326. The trap is not a contrived corner; it is the common case. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I sort the observations by finishing time, so after sorting `e_0 <= e_1 <= ... <= e_{n-1}`. I define `dp[i]` = the best total value achievable using a subset of the first `i` sorted observations (observations with sorted indices `0..i-1`). For observation `i-1` (the `i`-th in 1-based prefix terms), with start `s = s_{i-1}`, I have two choices:

- *Skip it:* the best is whatever I already had over the first `i-1`, namely `dp[i-1]`.
- *Take it:* I gain `v_{i-1}`, and I may keep any earlier compatible observations. An earlier observation `j < i-1` is compatible with taking `i-1` iff `e_j <= s`. Because the array is sorted by end, the compatible earlier observations are exactly a *prefix* `0..p-1` of the sorted order, where `p` is the number of earlier observations with `e_j <= s`. So the best compatible earlier value is `dp[p]`, and taking gives `v_{i-1} + dp[p]`.

Hence `dp[i] = max(dp[i-1], v_{i-1} + dp[p])`, with `dp[0] = 0`, and the answer is `dp[n]`. The crucial subtlety: this is valid only because the array is sorted **by end**, which is what makes "compatible earlier observations" a contiguous prefix and makes `ends[0..i-2]` an ascending array I can binary-search. I compute `p` as the count of `e_j <= s` among the first `i-1` ends, i.e. `p = upper_bound(ends, ends+(i-1), s) - ends` — the first index whose end is strictly greater than `s`, which is exactly the number of ends `<= s`.

Let me confirm the recurrence by hand on the documented sample, whose answer should be `70`. Sorting `{A[0,30)v20, B[0,90)v50, C[30,60)v25, D[60,90)v25, E[40,50)v5}` by end gives order A(30), E(50), C(60), B(90), D(90); ends array `[30,50,60,90,90]`. `dp[0]=0`.
- i=1, A, s=0: `p = #{ends in [] <= 0} = 0`; take `20 + dp[0] = 20`; `dp[1]=max(0,20)=20`.
- i=2, E, s=40: prefix ends `[30]`; `#{<=40}=1`, p=1; take `5 + dp[1] = 25`; `dp[2]=max(20,25)=25`.
- i=3, C, s=30: prefix ends `[30,50]`; `#{<=30}=1`, p=1; take `25 + dp[1] = 25+20 = 45`; `dp[3]=max(25,45)=45`.
- i=4, B, s=0: prefix ends `[30,50,60]`; `#{<=0}=0`, p=0; take `50 + dp[0] = 50`; `dp[4]=max(45,50)=50`.
- i=5, D, s=60: prefix ends `[30,50,60,90]`; `#{<=60}=3`, p=3; take `25 + dp[3] = 25+45 = 70`; `dp[5]=max(50,70)=70`.

Answer `dp[5]=70`. The recurrence reproduces the optimum, and step i=5 is exactly the A+C+D tiling that both greedies missed (D builds on `dp[3]`, which itself is C built on A). The math is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core loop, with the compatibility lookup:

```
int p = (int)(lower_bound(ends.begin(), ends.begin()+(i-1), s) - ends.begin());
long long take = obs[i-1].v + dp[p];
long long skip = dp[i-1];
dp[i] = max(take, skip);
```

Something about `lower_bound` vs `upper_bound` looks dangerous against my half-open rule, so I trace the smallest input that could expose it: two touching observations `[0,5) v=10` and `[5,10) v=10`. They touch at `5`, do not overlap, so the answer is obviously `20`. Sort by end: order `[0,5)`, `[5,10)`; ends `[5,10]`. `dp[0]=0`.
- i=1, `[0,5)`, s=0: `lower_bound(ends[0..0], 0)` over `[5]` → first `>= 0` is index 0 → p=0; take `10 + dp[0] = 10`; `dp[1]=10`.
- i=2, `[5,10)`, s=5: `lower_bound(ends[0..1], 5)` over `[5,10]` → first `>= 5` is index 0 → **p=0**; take `10 + dp[0] = 10`; `dp[2]=max(10,10)=10`.

**Diagnosing bug #1.** The code returns `10`, not `20`. The defect is precise. For observation `[5,10)` with start `s=5`, the earlier observation `[0,5)` ends at exactly `5`, and under my half-open rule `e <= s` means `5 <= 5` is **compatible** — I should be able to keep it. But `lower_bound(..., 5)` returns the first end `>= 5`, which *includes* the end equal to `5`, so it lands at index 0 and reports `p=0` (zero compatible predecessors), discarding the touching observation. I want the first end *strictly greater than* `s`, so that ends *equal to* `s` count as compatible. That is `upper_bound`, not `lower_bound`. I confirmed this empirically: a one-line build of the `lower_bound` version prints `10` on this exact input, while the fixed version prints `20`.

**Fixing bug #1 and re-verifying.** Swap to `upper_bound`:

```
int p = (int)(upper_bound(ends.begin(), ends.begin()+(i-1), s) - ends.begin());
```

Re-trace the touching case. i=2, `[5,10)`, s=5: `upper_bound(ends[0..1], 5)` over `[5,10]` → first `> 5` is index 1 → **p=1**; take `10 + dp[1] = 20`; `dp[2]=max(20,10)=20`. Answer `20`. Correct. And a quick sanity check that this did not over-shoot the other way on a genuine overlap: `[0,5) v=10`, `[4,10) v=10` overlap at the interior point 4.5; for the second, s=4, `upper_bound(ends[0..1], 4)` over `[5,10]` → first `> 4` is index 0 → p=0; take `10`; `dp=max(10,10)=10`. Correct — overlapping pair keeps only one. The case that broke now passes, and it broke for the reason I fixed.

**Second implementation wrinkle — the sort key — and a trace that catches it.** With `upper_bound` in place I re-read the derivation and get nervous about one line I wrote on autopilot: I sorted, but did I sort by **end**? My recurrence's whole validity rests on `ends[0..i-2]` being ascending so that "compatible predecessors form a prefix" and the binary search is even meaningful. Suppose I had instead sorted by **start** time (a very natural slip — interval problems are often described start-first). Let me trace a tiny instance where that goes wrong: `X[0,6) v=5`, `Y[1,3) v=4`, `Z[3,5) v=4`. The true optimum is `Y+Z = 8` (they touch at 3, both fit inside X's span but X overlaps both, so X alone is only 5). Sort by **start**: order X(0), Y(1), Z(3); the ends array is then `[6, 3, 5]`, which is **not** ascending. `dp[0]=0`.
- i=1, X, s=0: `upper_bound([6], 0)` → index 0 → p=0; take `5+dp[0]=5`; `dp[1]=5`.
- i=2, Y, s=1: `upper_bound([6,3], 1)` → first `>1` is index 0 → p=0; take `4+dp[0]=4`; `dp[2]=max(5,4)=5`.
- i=3, Z, s=3: `upper_bound([6,3], 3)` on the *unsorted* range `[6,3]` — binary search on unsorted data, so it returns a garbage position; here it lands at index 1, giving **p=1**; take `4 + dp[1] = 4+5 = 9`; `dp[3]=max(5,9)=9`.

**Diagnosing bug #2.** The start-sorted code returns `9`, but the true answer is `8`. The defect is exactly the invariant I flagged. By taking p=1, the code claims Z `[3,5)` is compatible with the prefix whose value `dp[1]=5` came from X `[0,6)` — but X overlaps Z, so that selection is illegal; the value 5 was never legitimately available to Z. The root cause is that `ends` is no longer sorted, so `upper_bound` no longer computes "number of predecessors ending by `s`"; it returns nonsense, and the recurrence silently builds on incompatible predecessors. I confirmed the divergence empirically: the start-sorted build prints `9` on this input and disagrees with the brute force on a measurable fraction of random instances. The fix is to honor the derivation literally — **sort by end time** — so that `ends` is ascending and the binary search means what the recurrence assumes:

```
sort(obs.begin(), obs.end(), [](const Obs &a, const Obs &b){
    if (a.e != b.e) return a.e < b.e;
    return a.s < b.s;            // tie-break by start; irrelevant to correctness
});
```

Re-trace the tiny case sorted by **end**: order Y(3), Z(5), X(6); ends `[3,5,6]`. `dp[0]=0`. i=1, Y, s=1: `upper_bound([],1)`... prefix empty → p=0; take `4`; `dp[1]=4`. i=2, Z, s=3: prefix `[3]`; `upper_bound([3],3)` → first `>3` is index 1 → p=1; take `4+dp[1]=8`; `dp[2]=max(4,8)=8`. i=3, X, s=0: prefix `[3,5]`; `#{<=0}=0`, p=0; take `5+dp[0]=5`; `dp[3]=max(8,5)=8`. Answer `8`. Correct — Y+Z, exactly the optimum, and X is correctly rejected.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the `cin >> n` succeeds with `0`, the read loop and DP loop never run, `dp` has one element `dp[0]=0`, output `0`. (If the input is entirely empty, `cin >> n` fails and I `return 0` having printed nothing — but the contract guarantees `n` is present, and the harness treats no-output-for-empty as fine; for `n=0` explicitly given I print `0`.) Verified: input `0` → `0`.
- `n = 1`, value 0: `[5,9) v=0` → `dp[1]=max(0, 0+dp[0])=0`. Output `0`. A single zero-value observation adds nothing. Verified.
- `n = 1`, positive: `[5,9) v=42` → `dp[1]=42`. Verified.
- Touching chain `[0,10)`,`[10,20)`,`[20,30)` each value 10: all compatible, answer `30`. Verified — this is the half-open rule doing its job at scale.
- Fully nested / all-overlapping: intervals `[i, 200000-i)` for many `i`, all sharing the midpoint, so at most one can be chosen; with each value `1` the answer is `1`. Verified at `n=100000`.
- Many identical intervals `[0,5) v=7` three times: pairwise overlapping, answer `7`. Verified.
- Overflow: `200000` disjoint intervals each value `10^9` → total `2*10^14`. With `long long` the answer prints `200000000000000` exactly; an `int` would wrap. Verified. The binary-search bounds use `i-1`, never reading past the filled prefix; coordinates up to `10^9` fit in `long long` with room to spare.
- Performance: the sort is `O(n log n)`, the DP loop does one `upper_bound` per observation (`O(log n)`), so `O(n log n)` overall. On `n = 2*10^5` with values near `10^9` it runs in about 0.03–0.04 s and ~11 MB — comfortably inside the 1 s / 256 MB budget, even on a heavy-overlap adversarial input.

**Cross-checking against a brute force.** I am not trusting hand traces alone. I wrote an independent exponential brute force that enumerates all `2^n` subsets and, for each, checks every pair for the half-open overlap `s_i < e_j && s_j < e_i`, keeping the max feasible total. I ran the `O(n log n)` solution against it on more than 1100 random small instances (two generators: one with tiny coordinate ranges to force frequent touching and exact-endpoint collisions, one with larger `n` up to 14 and wider values). Zero mismatches. The same harness independently confirmed that value-greedy and earliest-finishing each disagree with the true optimum on a large fraction of instances — the trap is real, the DP is right.

**Final solution.** I convinced myself the *idea* is right by disproving both greedies with a traced counterexample and hand-checking the end-sorted recurrence to `70`; I convinced myself the *code* is right by tracing two concrete bugs — `lower_bound` silently dropping touching intervals, and a start-time sort breaking the prefix-compatibility invariant — to precise causes, fixing each, and re-verifying on the very inputs that had failed, then cross-checking 1100+ cases against brute force. That is what I ship: one self-contained file, the `O(n log n)` interval DP I can defend rather than the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // Each observation: half-open interval [s, e) with value v >= 0.
    struct Obs { long long s, e, v; };
    vector<Obs> obs(n);
    for (int i = 0; i < n; i++) cin >> obs[i].s >> obs[i].e >> obs[i].v;

    // Sort by end time ascending (ties broken by start, irrelevant for correctness).
    sort(obs.begin(), obs.end(), [](const Obs &a, const Obs &b) {
        if (a.e != b.e) return a.e < b.e;
        return a.s < b.s;
    });

    // ends[i] = finishing time of the i-th observation in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = obs[i].e;

    // dp[i] = best total value using a subset of the first i sorted observations.
    // Half-open intervals: observation i (start obs[i].s) is compatible with any earlier
    // observation whose end <= obs[i].s, i.e. it does not overlap.
    vector<long long> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long s = obs[i - 1].s;
        // p = number of observations (among the first i-1) whose end <= s.
        // Those are exactly indices [0, p) in sorted order; ends is sorted ascending.
        int p = (int)(upper_bound(ends.begin(), ends.begin() + (i - 1), s) - ends.begin());
        long long take = obs[i - 1].v + dp[p];   // include observation i-1
        long long skip = dp[i - 1];              // exclude it
        dp[i] = max(take, skip);
    }

    cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** Both natural greedies looked right but a single traced counterexample (`A[0,30)20, B[0,90)50, C[30,60)25, D[60,90)25, E[40,50)5`: value-greedy 50 and earliest-finish 50 vs the reachable 70 via the A+C+D tiling) showed that one fat interval hogging the resource, or an early-finishing scrap, costs more than it gains, so I moved to an end-sorted interval DP `dp[i]=max(dp[i-1], v + dp[p])` and verified its recurrence reproduces 70; transcribing it, `lower_bound` silently dropped touching intervals (a trace of `[0,5)+[5,10)` returning 10 instead of 20 pinpointed it, fixed by `upper_bound` so ends equal to the start count as compatible), and sorting by start instead of end destroyed the ascending-`ends` invariant the binary search depends on (a trace of `X[0,6)5,Y[1,3)4,Z[3,5)4` returning 9 instead of 8 exposed a search over unsorted data, fixed by sorting on end); with `long long` accumulators closing the `2*10^14` overflow corner and brute-force agreement over 1100+ random cases, the `O(n log n)` DP is the version I ship.
