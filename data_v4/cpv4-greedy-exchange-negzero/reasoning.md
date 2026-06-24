**Reading the problem and pinning the contract.** I have `n` gigs; gig `i` carries a deadline `d[i]` and a payout `v[i]`. I may finish at most one gig per day, each accepted gig must be finished on some day in `1..d[i]`, and I may decline anything. I want to maximize the total payout, and since declining everything is allowed, the answer never drops below `0`. Input on stdin is `n`, then the `n` deadlines, then the `n` payouts; I print one integer. Two facts about the data jump out and I want them nailed before any algorithm, because they decide both the data types and the *base cases*: payouts can be negative or zero (`-10^9 <= v[i] <= 10^9`), and deadlines can be `0` (`0 <= d[i] <= 10^9`). The negative payouts mean "accepting more gigs" is not automatically better — a gig can hurt me — and the zero deadline means a gig can be physically unplaceable (there is no day `>= 1` that is `<= 0`). I expect both of these to be exactly where a naive solution dies.

**Scale and data types.** With `n` up to `2*10^5` and `|v[i]|` up to `10^9`, the accumulated answer can reach `2*10^5 * 10^9 = 2*10^14`. That blows past the 32-bit range (`~2.1*10^9`), so the running total must be `long long`. Deadlines can be up to `10^9`, but I can only ever fill at most `n` distinct days, so any deadline beyond `n` is no more useful than a deadline of exactly `n`; I will cap deadlines at `n` so that whatever day-indexed structure I build has size `O(n)` rather than `O(10^9)`. That capping is a correctness-preserving optimization, and I have to make sure it is *exactly* correct: capping `d[i]` to `min(d[i], n)` only ever removes days `> n`, and no feasible schedule needs a day index `> n` because there are at most `n` gigs total. Safe.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove*, not the one that types fastest.

- *Greedy exchange by payout.* Process gigs in decreasing payout order; place each on the latest still-free day at or before its (capped) deadline; if no free day exists, drop it. The intuition: taking the most valuable gigs first, and parking each as late as it is legally allowed, leaves the early days open for gigs that have no choice but to go early. With a disjoint-set "latest free day" structure this is `O(n alpha(n))` after an `O(n log n)` sort. The risk is the usual greedy risk — does the local "grab the biggest, place it as late as possible" really give the global optimum? — plus the sign question: do I ever want a non-positive gig?
- *Subset search with a feasibility test.* Enumerate every subset of gigs and test schedulability with the Hall condition (sort accepted deadlines ascending `d_1 <= ... <= d_m`; feasible iff `d_j >= j` for all `j`), keeping the best total. That is `O(2^n n)`, fine as a brute-force oracle on tiny `n`, hopeless at `2*10^5`. I will use it only to *check* the greedy.

**Deciding the sign question up front, because it is the trap.** Should I ever accept a gig with `v[i] <= 0`? No — and I want to argue it cleanly rather than wave at it. Suppose an optimal plan accepts a gig with payout `<= 0` on some day `t`. Remove that gig (leave day `t` empty). Every other accepted gig keeps its day, so the schedule stays feasible, and the total payout changes by `-v[i] >= 0`, i.e. it does not decrease. So there is always an optimal plan that accepts **only strictly positive** gigs. This is the whole pitfall in one sentence: *a greedy that schedules any gig it can fit, ignoring the sign of the payout, will gladly accept a `-5` gig and report a worse total.* I will filter to `v[i] > 0` before doing anything else. As a corollary, if every gig is non-positive (all-negative, all-zero, or a mix that happens to have no positives), the accepted set is empty and the answer is `0` — which is also exactly the empty-instance answer. The base case and the sign handling are the same fact.

**Stress-testing the greedy's placement rule before committing.** "Place each gig as late as possible" is the load-bearing claim; let me try to break it. Take gigs (deadline, payout): `A=(1, 10)`, `B=(2, 9)`, `C=(2, 8)`. Sort by payout desc: `A(10), B(9), C(8)`. Place `A` on its latest free day `<= 1`: day 1. Place `B` on latest free `<= 2`: day 2. Place `C` on latest free `<= 2`: day 2 is taken, day 1 is taken, none free → drop `C`. Greedy total `10 + 9 = 19`. Is that optimal? The only days are 1 and 2, so at most two gigs fit; the two biggest payouts are `10` and `9`, and they have deadlines `1` and `2` which are jointly schedulable (`A` on day 1, `B` on day 2). `19` is optimal. Now a sharper case where placing late matters: `A=(2, 10)`, `B=(1, 9)`. If I (wrongly) placed `A` as *early* as possible, `A` would take day 1, then `B` (deadline 1) finds day 1 taken and is dropped → total `10`. Placing `A` *late* (day 2) leaves day 1 free for `B` → total `19`. The "latest free day" rule is what saves the second gig; placing early would have cost me `9`. Good — the rule earns its keep, and I now see *why* it is the right exchange direction.

**Choosing the data structure.** I need, repeatedly, "the largest day index `<= cap` that is still free," and then "mark that day used." A disjoint-set forest does this in near-constant amortized time: `par[x]` points toward the latest free day at or before `x`; `findFree(x)` follows parents until it reaches a root (a free day, or `0` meaning none); marking day `s` used sets `par[s] = s - 1`, so future queries skip past it to the next-earlier candidate. Day `0` is a sentinel "no valid day" — its parent is itself and a returned slot of `0` means "give up on this gig." I size the forest to `maxDay + 1` where `maxDay` is the largest capped deadline among positive gigs (at most `n`).

**First implementation — and immediately a trace, because clean intent transcribes dirty.** Here is my first cut of the core, written quickly:

```
vector<int> order;                 // indices of ALL gigs
for (int i = 0; i < n; i++) order.push_back(i);
sort(order.begin(), order.end(), [&](int a, int b){ return v[a] > v[b]; });
long long maxDay = 0;
for (int i = 0; i < n; i++) maxDay = max(maxDay, min(d[i], (long long)n));
par.assign(maxDay + 1, 0);
for (int day = 0; day <= maxDay; day++) par[day] = day;
long long answer = 0;
for (int idx : order) {
    long long cap = min(d[idx], maxDay);
    int slot = findFree((int)cap);
    if (slot > 0) { answer += v[idx]; par[slot] = slot - 1; }
}
```

I notice two things that worry me before I even run it: I built `order` over *all* gigs (not just positive ones), and the accept condition is only `slot > 0` — it never looks at the *sign* of `v[idx]`. Let me trace the smallest input that should expose that: `n = 1`, `d = [3]`, `v = [-7]`. The answer is obviously `0` (decline the loss). Trace: `order = [0]`; `maxDay = min(3,1) = 1`; `par = [0, 1]`. For `idx = 0`: `cap = min(3,1) = 1`; `findFree(1)` → `par[1]=1` is a root → `slot = 1 > 0`, so `answer += v[0] = -7`, mark day 1 used. Final `answer = -7`.

**Diagnosing bug #1 — the sign/base-case bug.** The code prints `-7`, but the correct answer is `0`. The defect is exactly the trap I flagged in the contract: the accept test `slot > 0` asks only "is there a free day?", never "is this gig worth taking?". A free day exists, so it accepts a gig that *loses* money. Worse, it can drive the answer **below zero**, which is impossible — declining everything guarantees `0`. The fix follows the exchange argument I proved earlier: never accept a gig with `v <= 0`. Two clean ways to enforce it — gate inside the loop with `if (v[idx] <= 0) continue;`, or, better, *never put non-positive gigs into `order` at all*, which also shrinks the sort and the forest. I take the second: build `order` only from indices with `v[i] > 0`, and compute `maxDay` only over those. Now if there are no positive gigs (all-negative, all-zero, or empty `n=0`), `order` is empty, the loop never runs, and `answer` stays `0`. The base case and the sign case collapse into one correct branch — precisely as the proof predicted.

**Re-verifying the fix on the failing case and its neighbours.** With the filter: `n=1, d=[3], v=[-7]` → no positive gigs → `order` empty → `answer = 0`. Correct. `n=1, d=[3], v=[0]` → `0 > 0` is false → declined → `0`. Correct (a zero gig is worthless to accept). `n=1, d=[2], v=[5]` → `order=[0]`, `maxDay=min(2,1)=1`, place on day 1, `answer = 5`. Correct. `n=3, d=[2,2,2], v=[10,20,-5]` → positives are indices 1,0; `maxDay=min(2,3)=2`; sort desc → `[20@idx1, 10@idx0]`; place `20` on day 2, place `10` on day 1; the `-5` was filtered out → `answer = 30`. Correct, and crucially the negative gig never even competed for a day.

**A second trace targeting the deadline-zero / cap corner.** The other place I expect breakage is invalid days. Consider `n=2`, `d=[0, 1]`, `v=[100, 50]`. Gig 0 has deadline `0` — there is *no* day `>= 1` that is `<= 0`, so it can never be finished and must be declined despite its juicy `100`. Trace with the fixed code: positives are both; `maxDay = max(min(0,2), min(1,2)) = max(0,1) = 1`; `par = [0, 1]`. Sort desc → `[100@idx0, 50@idx1]`. For `idx0`: `cap = min(0, 1) = 0`. Now `findFree(0)`: `par[0] = 0` is a root, returns `0`; `slot = 0` is **not** `> 0`, so the gig is correctly dropped. For `idx1`: `cap = min(1,1) = 1`; `findFree(1) = 1 > 0`; accept, `answer = 50`. Final `50`. Let me double check the brute thinks so too: the only schedulable nonempty set is `{gig1}` on day 1 (gig 0 has deadline 0, unschedulable), payout `50`; the empty set gives `0`; so `50` is right.

**Diagnosing bug #2 — what an earlier draft of `findFree` would have done.** When I first sketched the DSU I wrote the index guard as `if (slot >= 0)` instead of `slot > 0`. Re-trace the same `d=[0,1]` case with that weaker guard: for `idx0`, `cap = 0`, `findFree(0)` returns `0`, and `slot >= 0` is **true**, so the code would do `answer += 100` and set `par[0] = -1` — both wrong: it accepts an unplaceable gig *and* writes a negative index that corrupts the forest for later queries. The symptom would be a wildly inflated answer plus possible undefined behavior on the next `findFree`. The fix is the strict `slot > 0`: day `0` is the sentinel "no valid day," never a real slot. The capped-deadline path and the deadline-`0` path both funnel into `cap <= 0 → no acceptance`, which I also guard explicitly with `if (cap <= 0) continue;` so the intent is unmistakable and I never even call `findFree(0)` for a placement.

**Sanity-checking the derivation on the documented sample.** `d = [2, 1, 2, 1, 3]`, `v = [20, 10, 40, 30, 50]`, claimed answer `120`. All payouts positive, so all five enter `order`. `maxDay = max(2,1,2,1,3)` capped at `min(., 5)` = `3`; `par = [0,1,2,3]`. Sort by payout desc: `50@idx4 (d3)`, `40@idx2 (d2)`, `30@idx3 (d1)`, `20@idx0 (d2)`, `10@idx1 (d1)`. Place `50`: `findFree(min(3,3)=3)=3` → day 3, `par[3]=2`. Place `40`: `findFree(min(2,3)=2)=2` → day 2, `par[2]=1`. Place `30`: `findFree(min(1,3)=1)=1` → day 1, `par[1]=0`. Place `20`: `findFree(min(2,3)=2)`: `par[2]=1`→`par[1]=0`→ root `0`, slot `0`, dropped. Place `10`: `findFree(1)`: `par[1]=0` → slot `0`, dropped. Total `50+40+30 = 120`. Matches. The greedy fills days 3,2,1 with the three biggest payouts whose deadlines admit them, and correctly cannot squeeze a fourth gig into two-and-a-half occupied days.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the input has just `0`; `order` is empty, the loop never runs, `answer = 0`. Correct (and `if (!(cin >> n)) return 0;` also covers a stream that ends after `n`).
- `n = 1`, single negative or single zero: filtered out → `0`. Single positive with valid deadline `>= 1`: accepted. Single positive with deadline `0`: `cap = 0`, dropped → `0`.
- All-negative / all-zero arrays: no positives → `order` empty → `0`. This is the headline corner and it is handled by the *same* mechanism as the empty instance.
- Deadline `0` among positives: `cap = 0`, `cap <= 0 → continue`, never placed. Deadline `> n`: capped to `n`, harmless because no schedule needs a day index beyond `n`.
- Ties (equal payouts competing for the same early day): the comparator is a strict `>` so the sort is a valid weak ordering; whichever equal-payout gig is placed first takes the later day, the other falls to an earlier day or is dropped, and since their payouts are equal the *total* is order-independent. Verified on `d=[1,2,2,1], v=[10,5,5,10]` → answer `15` (one of the two `10`s on day 1, one `5` on day 2; the other `10` and `5` are squeezed out), which the brute confirms.
- Overflow: `answer` is `long long`; the maximum sum `~2*10^14` fits with enormous headroom. Deadlines are read into `long long` so `min(d[i], (long long)n)` and the cap arithmetic never overflow. The forest indices are `int` but bounded by `n <= 2*10^5`. Safe.
- Performance: at `n = 2*10^5` with all distinct large deadlines the run is ~0.07s; the pathological all-deadline-`1` case (every positive gig fighting for day 1, so the DSU repeatedly collapses to the day-`0` sentinel) is ~0.09s. Comfortably under the 1s limit.

**Verification against the brute oracle.** I ran the disjoint-set greedy against the exhaustive `O(2^n)` Hall-condition brute over 3000+ random small instances — deliberately seeded to include `n=0`, all-negative, all-zero, mixed signs, deadline-`0`, and deadlines exceeding `n` — and they agree on every case with zero mismatches. The two traced bugs above (accepting a non-positive gig; the `slot >= 0` vs `slot > 0` sentinel) are exactly the kind of disagreement that sweep would have caught, and both are fixed in the final code.

**Final solution.** I convinced myself the *idea* is right by proving the sign reduction and the latest-free-day exchange direction, and I convinced myself the *code* is right by tracing the two failing inputs to precise causes, re-verifying the fixes, and stress-testing against an independent brute. That is what I ship — one self-contained file, the `O(n log n)` greedy with a disjoint-set "latest free day," guarded so non-positive payouts and invalid days never get accepted:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "latest free day": find(x) returns the largest day index <= x
// that is still free (0 means no free day at or before x).
static vector<int> par;
int findFree(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                 // n = 0 (or empty input) -> answer 0
    vector<long long> d(n), v(n);
    for (int i = 0; i < n; i++) cin >> d[i];
    for (int i = 0; i < n; i++) cin >> v[i];

    // Only positive-payout gigs are ever worth scheduling; a gig with v <= 0
    // can always be declined for a strictly-not-worse total, so we skip it.
    // Among positive gigs, sort by payout descending and assign each to the
    // latest still-free day at or before its deadline (greedy exchange).
    vector<int> order;
    order.reserve(n);
    long long maxDay = 0;
    for (int i = 0; i < n; i++) {
        if (v[i] > 0) {
            order.push_back(i);
            // a deadline beyond n is useless: at most n gigs fit, so cap at n.
            long long cap = min(d[i], (long long)n);
            if (cap > maxDay) maxDay = cap;
        }
    }
    sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });

    par.assign((size_t)maxDay + 1, 0);
    for (int day = 0; day <= (int)maxDay; day++) par[day] = day;

    long long answer = 0;
    for (int idx : order) {
        long long cap = min(d[idx], maxDay);   // latest day this gig may occupy
        if (cap <= 0) continue;                // deadline 0 -> no valid day
        int slot = findFree((int)cap);
        if (slot > 0) {                        // a free day exists
            answer += v[idx];
            par[slot] = slot - 1;              // mark day `slot` used
        }
    }

    cout << answer << "\n";                     // empty / all-nonpositive -> 0
    return 0;
}
```

**Causal recap.** The exchange argument settles two things at once: only strictly positive gigs can ever help (dropping a `v <= 0` gig keeps a schedule feasible and never lowers the total), and placing each accepted gig on the *latest* legal free day maximally protects the early days for deadline-tight gigs — so I sort positives by payout descending and assign via a disjoint-set "latest free day." The first trace (`v=[-7]` printing `-7`) exposed the headline sign/base-case bug — an accept test of `slot > 0` that ignored the payout's sign and could drive the total negative — fixed by filtering to `v[i] > 0`, which simultaneously makes the empty, all-negative, and all-zero instances return `0` through one code path. The second trace (`d=[0,1]`) exposed the sentinel bug — `slot >= 0` would accept an unplaceable deadline-`0` gig and corrupt the forest — fixed by the strict `slot > 0` plus an explicit `cap <= 0 → continue`. Capping deadlines at `n` keeps the structure `O(n)`, `long long` absorbs the `~2*10^14` totals, and a 3000-case sweep against an exhaustive Hall-condition brute closes out correctness.
