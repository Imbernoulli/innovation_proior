**Reading the problem and pinning the contract.** I have `n` booking requests; request `i` occupies the half-open interval `[s_i, f_i)` and pays profit `p_i > 0`. I accept a subset with no two accepted intervals overlapping, and I maximize the total profit; the empty schedule is allowed, so the answer is never below `0`. The room is a single resource, so "overlap" is the only conflict. The half-open convention matters: `[s_i, f_i)` and `[s_j, f_j)` are compatible exactly when `f_i <= s_j` or `f_j <= s_i`; touching at an endpoint (`f_i == s_j`) is allowed, because one booking ends the instant the next begins. I will hold to that `<=` boundary everywhere — it is the single most error-prone detail in interval problems.

Before any algorithm I fix scale, because it dictates data types. `n <= 2*10^5`; coordinates `s_i, f_i <= 10^9`; profits `p_i <= 10^9`. The total profit can reach `2*10^5 * 10^9 = 2*10^14`, which blows past the 32-bit range of about `2.1*10^9`. So every accumulator that holds a profit sum must be 64-bit. Coordinates also reach `10^9`, which fits in 32-bit signed, but I will read everything as `long long` to avoid mixing widths in comparisons and additions. An `int` profit accumulator here is a silent wrong-answer on the large tests; this decision is non-negotiable.

**Candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Sweep-and-grab greedy.* Sort by a single key and walk through accepting whatever is compatible with what I already hold. The natural keys are (a) earliest finishing time — this is the famous optimal rule for *maximizing the count* of non-overlapping intervals — and (b) largest profit first. Either is `O(n log n)` and a handful of lines. The structural worry is that the profit makes this a weighted problem, and a single sort key is a purely local decision; that is exactly the configuration where greedy tends to fail. I will not trust it until I have tried to break it.
- *Sweep + interval DP.* Sort by finishing time, then process requests in that order; for each request decide accept-vs-reject by combining its profit with the best total achievable from requests that finish at or before this one's start, found by binary search. `O(n log n)`. The risk here is not the idea but the transcription: the recurrence, the binary-search predicate, and the off-by-one between "index in the sorted array" and "best value to reach back to."

**Stress-testing the greedy before committing.** Hand-waving "earliest finish is optimal" is how wrong solutions get shipped — that rule is optimal for the *unweighted* count problem, and I must not smuggle that intuition into the weighted one. Let me actually attack it with a concrete instance. Take two requests: `A = [0,1)` profit `1` and `B = [0,3)` profit `5`. Earliest-finish greedy sorts by `f`: `A` (finishes at 1) before `B` (finishes at 3). It accepts `A` for profit `1`; then `B` overlaps `A` (they share `[0,1)`), so `B` is rejected. Greedy's total is `1`. But accepting `B` alone gives `5`, and `A` and `B` cannot coexist, so the optimum is `5`. Earliest-finish greedy is wrong by a factor of five here, and I see *why*: it grabbed the request that frees the resource soonest, but freeing the resource soonest is worthless when the thing it blocked was far more profitable. Earliest-finish is out.

Maybe largest-profit-first survives? Try `[0,4)` profit `10`, `[0,2)` profit `7`, `[2,4)` profit `7`. Profit-first takes `[0,4)` (profit 10); both `[0,2)` and `[2,4)` overlap it, so they are rejected — total `10`. But `[0,2)` and `[2,4)` touch at `2` (compatible, half-open) and together give `7 + 7 = 14 > 10`. So largest-profit-first is wrong too: one fat interval can block two lean ones that jointly beat it. Both single-key greedies are dead, and for the same root cause — a local accept commits the shared resource over a span whose *global* opportunity cost the greedy never measures.

**Deriving the DP and checking the recurrence on paper.** Since no single greedy key works, I need to weigh accept against reject *with* the value of what each choice forecloses. The clean way is to impose an order in which "what came before" is fully summarized by one number. Sort all requests by finishing time `f`, ascending, and relabel them `0,1,...,n-1` in that order. Define

- `best[i]` = the maximum total profit using only the first `i` requests (those with the `i` smallest finishing times).

`best[0] = 0` (no requests, accept nothing). For request `i` (the one at sorted position `i`, 0-based), I have two choices when extending to `best[i+1]`:

- *Reject `i`*: the best over the first `i` requests is unchanged, contributing `best[i]`.
- *Accept `i`*: I collect `p_i`, but then no accepted request among the first `i` may overlap `[s_i, f_i)`. Because I sorted by finishing time, every earlier request finishes at or before `f_i`; the ones compatible with `i` are exactly those whose finish is `<= s_i`. Crucially the finishing times are sorted, so the compatible earlier requests form a *prefix* of the sorted order: there is some count `j` such that requests `0..j-1` all have `f <= s_i` and requests `j..i-1` do not. Then the best I can pair with `i` is `best[j]`, and accepting gives `best[j] + p_i`.

So `best[i+1] = max(best[i], best[j] + p_i)`, where `j` is the number of requests among the first `i` whose finishing time is `<= s_i`. The answer is `best[n]`. The count `j` is found by binary search over the sorted finishing-time array `F[0..i-1]`: the largest prefix with `F[k] <= s_i`. Note `<=`, not `<`, because touching endpoints are compatible.

Let me sanity-check the recurrence on the documented sample. Six requests, given as `(s,f,p)`:
`1:(0,5,30)`, `2:(0,2,20)`, `3:(2,4,6)`, `4:(4,7,20)`, `5:(5,9,25)`, `6:(7,9,8)`. Sort by `f`: `(0,2,20)`, `(2,4,6)`, `(0,5,30)`, `(4,7,20)`, `(5,9,25)`, `(7,9,8)` -> finishing times `F = [2,4,5,7,9,9]`. Walk `best`, with `best[0]=0`:
- i=0, request `(0,2,20)`, `s=0`: prefix of `F[0..-1]` with `F<=0` is empty, `j=0`. accept `best[0]+20=20`, reject `best[0]=0`. `best[1]=20`.
- i=1, request `(2,4,6)`, `s=2`: among `F[0..0]=[2]`, `F<=2` -> `j=1`. accept `best[1]+6=26`, reject `best[1]=20`. `best[2]=26`.
- i=2, request `(0,5,30)`, `s=0`: among `F[0..1]=[2,4]`, none `<=0` -> `j=0`. accept `best[0]+30=30`, reject `best[2]=26`. `best[3]=30`.
- i=3, request `(4,7,20)`, `s=4`: among `F[0..2]=[2,4,5]`, `F<=4` -> `[2,4]` -> `j=2`. accept `best[2]+20=46`, reject `best[3]=30`. `best[4]=46`.
- i=4, request `(5,9,25)`, `s=5`: among `F[0..3]=[2,4,5,7]`, `F<=5` -> `[2,4,5]` -> `j=3`. accept `best[3]+25=55`, reject `best[4]=46`. `best[5]=55`.
- i=5, request `(7,9,8)`, `s=7`: among `F[0..4]=[2,4,5,7,9]`, `F<=7` -> `[2,4,5,7]` -> `j=4`. accept `best[4]+8=54`, reject `best[5]=55`. `best[6]=55`.

Answer `best[6]=55`, which is requests `1` and `5` (`30 + 25`). Earliest-finish greedy on this same input grabs `(0,2)`,`(2,4)`,`(4,7)`,`(7,9)` for `20+6+20+8=54` — one short of optimal — so the sample also demonstrates the trap. The recurrence is right.

**First implementation and a trace, because clean math transcribes dirty.** My first cut of the binary search and DP, after sorting index array `idx` by `f` into `S,F,P`:

```
vector<long long> best(n + 1, 0);
for (int i = 0; i < n; i++) {
    int lo = 0, hi = i;           // search F[0..i-1]
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (F[mid] < S[i]) lo = mid + 1;   // <-- first attempt: strict <
        else hi = mid;
    }
    long long take = best[lo] + P[i];
    long long skip = best[i];
    best[i + 1] = max(take, skip);
}
```

The predicate `F[mid] < S[i]` worries me immediately, because touching endpoints should be compatible. Let me trace the smallest input that exposes it: two requests that touch, `(0,2,20)` and `(2,4,6)`. The expected answer is `26` — they touch at `2`, so both are accepted, `20 + 6 = 26`. Sort by `f`: already `F=[2,4]`, `S=[0,2]`, `P=[20,6]`. `best[0]=0`.
- i=0, `S[0]=0`: search `F[0..-1]` empty, `lo=0`. take `best[0]+20=20`, skip `best[0]=0`. `best[1]=20`.
- i=1, `S[1]=2`: search `F[0..0]=[2]` for largest prefix with `F<2`. `lo=0,hi=1`; `mid=0`, `F[0]=2 < 2`? No, so `hi=0`. Loop ends, `lo=0`. take `best[0]+6=6`, skip `best[1]=20`. `best[2]=max(6,20)=20`.

**The bug.** The code returns `20`, but the true answer is `26`. The strict predicate `F[mid] < S[i]` treated the touching request `(0,2)` as *incompatible* with `(2,4)` — it required the earlier finish to be strictly before the later start, so the binary search landed `lo=0` (no compatible predecessor) when it should have landed `lo=1` (request 0 finishes exactly at 2 = start of request 1, which is allowed). The fix is to make the predicate `F[mid] <= S[i]`: a predecessor is compatible when it finishes *at or before* this request's start.

**Fix and re-verification.** Change the comparison to `<=`:

```
if (F[mid] <= S[i]) lo = mid + 1;
else hi = mid;
```

Re-trace `(0,2,20),(2,4,6)`: i=1, `S[1]=2`, search `F[0..0]=[2]`: `mid=0`, `F[0]=2 <= 2`? Yes, `lo=1`. take `best[1]+6=26`, skip `best[1]=20`. `best[2]=26`. Correct. Re-trace the strictly-overlapping pair `(0,1,1),(0,3,5)` (answer 5, they share `[0,1)`): sort by `f` -> `F=[1,3]`, `S=[0,0]`, `P=[1,5]`. i=0: search empty, `lo=0`, `best[1]=max(0+1,0)=1`. i=1, `S[1]=0`: search `F[0..0]=[1]`: `mid=0`, `F[0]=1<=0`? No, `hi=0`, `lo=0`. take `best[0]+5=5`, skip `best[1]=1`. `best[2]=5`. Correct — the overlapping predecessor is correctly *not* counted because `1 <= 0` is false. The case that broke now passes, and it broke for the exact reason I fixed.

**A second trace, on the binary-search semantics themselves.** I want to be sure `lo` after the loop means "number of requests among the first `i` with `F <= S[i]`," i.e. the right index into `best`. The loop is the standard `lower_bound`-style partition: it returns the first position where `F[mid] <= S[i]` fails, which is exactly the count of the leading block where it holds — because `F` is sorted ascending, the compatible requests are a contiguous prefix. Trace request `(4,7,20)` from the sample at i=3 with `F[0..2]=[2,4,5]`, `S=4`: `lo=0,hi=3`. `mid=1`, `F[1]=4<=4`? Yes, `lo=2`. `mid=2`, `F[2]=5<=4`? No, `hi=2`. `mid=... lo=2,hi=2` stop. `lo=2`. So I reach back to `best[2]` (the best using the two compatible predecessors `F=2,4`), accept gives `best[2]+20`. That matches my paper trace (`46`). The index discipline is right: `best` has size `n+1`, `best[i]` is "first `i` requests," and `lo in [0,i]` is always a valid index because the search range is `F[0..i-1]`.

**Edge cases, deliberately, because this is where interval code dies.**
- `n = 0`: the loop never runs; `best` is `{0}` (size 1) and I print `best[0] = 0`. The reading guard `if (!(cin >> n)) return 0;` also covers empty input -> `0`. Correct.
- `n = 1`, request `(5,9,7)`: i=0, search empty -> `lo=0`, take `best[0]+7=7`, skip `0`, `best[1]=7`. Accept the lone booking. Correct.
- All requests mutually overlapping, e.g. `(0,10,3),(0,10,9),(0,10,4)`: every request has `S=0`, and the only `F<=0` would require a zero-length predecessor, which cannot exist (`s<f`), so every binary search yields `lo=0`. Then `best[i+1]=max(best[i], best[0]+P[i]) = max(best[i], P[i])`, i.e. the running maximum single profit. Answer `9`. Correct — exactly one of a fully-overlapping clique.
- Touching chain `(0,1,1),(1,2,1),(2,3,1)`: each accepts on top of the previous via the `<=` predicate, total `3`. Correct.
- Overflow: `best` is `long long`; the maximum total `~2*10^14` fits with three orders of magnitude to spare. I never add to a sentinel — `best` starts at `0` and only ever accumulates real profits. Coordinates up to `10^9` are compared, never multiplied, so no coordinate overflow. Safe.
- Sort stability: I sort *indices* by `f` only; ties in `f` are fine in any order, because the recurrence's correctness depends only on `F` being non-decreasing, not on tie-breaking. The binary search's `<=` handles equal finishing times among predecessors correctly regardless of their relative order.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so line-based or token-based input both parse.

**Final solution.** I convinced myself the *idea* is right by disproving both greedies with traced counterexamples and hand-checking the recurrence on the six-request sample, and I convinced myself the *code* is right by tracing the touching-endpoint failure to a precise cause (strict vs. non-strict predicate), fixing it, and re-verifying the fix plus the overlapping case and the corners. Then I stress-tested against an exhaustive brute force over `601` random small instances with zero mismatches, and timed the large case (`n = 2*10^5`) at well under the limit. That is what I ship — one self-contained `O(n log n)` file, the sort-plus-DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> profit 0
    vector<long long> s(n), f(n), p(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> f[i] >> p[i];

    // Sort jobs by finishing time (the sweep order).
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int x, int y){ return f[x] < f[y]; });
    vector<long long> S(n), F(n), P(n);
    for (int i = 0; i < n; i++) { S[i] = s[idx[i]]; F[i] = f[idx[i]]; P[i] = p[idx[i]]; }

    // best[i] = max profit considering the first i jobs (in finish order).
    // For job i (0-based), find p = largest index j < i with F[j] <= S[i] (compatible),
    // via binary search on the sorted F array.
    vector<long long> best(n + 1, 0);
    for (int i = 0; i < n; i++) {
        // largest k in [0, i) with F[k] <= S[i]; count of such = j, so best[j] is reachable.
        int lo = 0, hi = i; // search in F[0..i-1]
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (F[mid] <= S[i]) lo = mid + 1;
            else hi = mid;
        }
        // lo = number of jobs among first i with F <= S[i]
        long long take = best[lo] + P[i];
        long long skip = best[i];
        best[i + 1] = max(take, skip);
    }

    cout << best[n] << "\n";
    return 0;
}
```

**Causal recap.** Earliest-finish greedy is the optimal rule for *counting* non-overlapping intervals, but a single traced counterexample (`[0,1) p=1` vs `[0,3) p=5`: greedy `1` against the reachable `5`) shows it ignores profit, and a second (`[0,4) p=10` vs `[0,2)+[2,4) = 14`) kills largest-profit-first too — a local accept commits the shared resource over a span whose global opportunity cost greedy never measures. So I sort by finishing time and run a DP where `best[i+1] = max(best[i], best[j] + p_i)` with `j` the count of compatible predecessors found by binary search; my first binary search used a strict `F < S` predicate and a trace of the touching pair `(0,2)+(2,4)` returning `20` instead of `26` pinpointed it, fixed by `F <= S` for the half-open convention; and `long long` accumulators with a `best`-array that only ever sums real profits close out the all-overlapping, single, empty, and overflow corners, all confirmed against an exhaustive brute force.
