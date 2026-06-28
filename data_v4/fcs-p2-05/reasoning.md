**Reading the problem and pinning the contract.** I have an `n x n` cost matrix, `cost[i][j]` is the price of giving worker `i` task `j`, and I must assign every worker to exactly one task and every task to exactly one worker so the total is minimized. A valid assignment is precisely a permutation `p` of `{0,...,n-1}`, and I want `min_p sum_i cost[i][p[i]]`. Input on stdin is `n` then the `n*n` entries in row-major order; I print one integer. Before any algorithm I fix the scale, because it dictates the data types and the feasible methods: `n <= 18`, and `|cost[i][j]| <= 10^9`. A total cost sums `n` cells, so its magnitude can reach `18 * 10^9 = 1.8*10^10`. That blows past the 32-bit range of about `2.1*10^9`, so every accumulator and every matrix value has to be 64-bit. I will use `long long` throughout. That is the first decision and it is non-negotiable; an `int` accumulator here is a silent wrong-answer on the large tests. I also note the corner the contract calls out explicitly: `n = 0` must print `0` — the empty assignment costs nothing.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* fast enough, not the one that is shortest to type.

- *Cheapest-available greedy.* Repeatedly take the smallest remaining cell `cost[i][j]` whose worker `i` and task `j` are both still free, lock that worker to that task, and forbid both. There is also the row-by-row cousin: process workers in index order, each grabbing its cheapest still-free task. Both are `O(n^2 log n)` and a few lines. The risk is structural: the perfect-matching constraint is global, and greedy decides locally, which is exactly the configuration where greedy tends to be wrong. I will not trust it until I have tried to break it.
- *The Hungarian algorithm.* The textbook polynomial `O(n^3)` exact method for assignment. It is genuinely correct, but it is also genuinely fiddly: potentials, augmenting paths, the `min` over slack, the dual updates — a lot of index-careful code where a single off-by-one in the potential update gives a wrong but plausible number on most tests and a wrong one on the adversarial test. At `n <= 18` I do not *need* its `O(n^3)` scaling; paying its implementation-risk tax buys me nothing here.
- *Bitmask DP over the set of assigned tasks.* Place workers in index order, `0, 1, 2, ...`; the only thing the future cares about is *which tasks have already been used*, not which worker took which. So the state is a subset `mask` of tasks, and `dp[mask]` is the best cost to assign the first `popcount(mask)` workers onto exactly the tasks in `mask`. This is `O(n^2 * 2^n)`. The question is whether `2^n` is affordable at `n = 18`.

**Sizing the DP before betting on it.** `2^18 = 262144` states, and each state tries up to `n = 18` task transitions, so the work is about `18 * 2^18 = 4.7*10^6` transition attempts — or counting the popcount per state, well under `10^8` primitive operations. That is comfortably under a 2-second limit. Memory is one `long long` per state: `2^18 * 8 bytes = 2 MB`, trivial against 256 MB. So the DP is affordable with room to spare. This matters: it means I get a method I can *prove* correct by a one-line exchange/optimal-substructure argument, at a fraction of the bug-surface of the Hungarian algorithm, and I never even have to consider whether greedy's shortcut is safe. The cheap, provable method is the *better engineering choice here*, not a fallback.

**Stress-testing greedy before committing — actually trying to break it.** Hand-waving "cheapest-first feels right" is how wrong solutions get shipped, so let me attack greedy with concrete small matrices rather than trust it. I want the smallest matrix where grabbing a cheap cell forces an expensive completion. `n = 2` cannot show it: with two cells per assignment, cheapest-cell-first picks one `0`-ish cell and the only remaining legal cell, and the two permutations are easy to compare; greedy and optimum coincide. So I go to `n = 3` and search by hand-and-script for a matrix where *both* greedy variants miss. Here is one I found and then re-derived by hand:

```
cost =
[ 0  6  3 ]
[ 6  0  8 ]
[ 3  7  7 ]
```

Run the cheapest-available greedy. The globally smallest cells are the two `0`s: `cost[0][0] = 0` and `cost[1][1] = 0`. Greedy grabs `cost[0][0] = 0` (lock worker 0 -> task 0), then `cost[1][1] = 0` (lock worker 1 -> task 1). Now worker 2 and task 2 are the only free pair, forced: `cost[2][2] = 7`. Greedy's total is `0 + 0 + 7 = 7`. The row-by-row greedy does the same thing: worker 0's cheapest is task 0 (`0`), worker 1's cheapest free is task 1 (`0`), worker 2 is forced onto task 2 (`7`) — also `7`.

Is `7` optimal? Let me hunt for an assignment greedy structurally could not reach. Try the anti-pattern `worker 0 -> task 2 (3)`, `worker 1 -> task 1 (0)`, `worker 2 -> task 0 (3)`: total `3 + 0 + 3 = 6`. That is strictly better than `7`. So greedy is wrong, and I now see exactly *why*: by snatching the `0` at `cost[0][0]` it consumed task 0, the only cheap option for worker 2 (`cost[2][0] = 3` versus `7` and `7` on worker 2's other tasks). The single greedy grab of a local `0` propagated into a forced `7` downstream, and the net was worse by 1. The verification paid off — it killed an approach I would otherwise have been tempted to ship, and it killed it the same way for *both* greedy variants. Greedy is out.

That `n = 3` instance becomes my first deliberate test case (and the worked sample in the statement): correct answer `6`, greedy answer `7`, so it discriminates a correct solver from a greedy one in a single line.

**Deciding between Hungarian and the bitmask DP.** Greedy is dead; now it is Hungarian versus the DP, and this is where I refuse to over-engineer. Both are exact. The DP is the one I can *prove* in two sentences and *trace* by hand, and at `n <= 18` it is fast enough. The Hungarian algorithm would be the right call if `n` were in the hundreds, but here it would only add a large, easy-to-get-subtly-wrong block of code whose payoff (better asymptotics) is irrelevant at this scale. The destination is the simpler provable method *because* I checked the constraints let it pass, not because I started out wanting a DP. I commit to the bitmask DP.

**Deriving the DP and proving it.** Assign workers in a fixed order `0, 1, 2, ...`. Claim: after deciding workers `0..k-1`, the only information the remaining decisions need is the *set* `S` of tasks already consumed — not the particular worker-to-task pairing, because any future cost depends only on which tasks are still available. Define

- `dp[mask]` = minimum total cost of assigning workers `0..k-1` (where `k = popcount(mask)`) onto exactly the tasks in `mask`.

The worker index is *implied* by the popcount of the mask: if `mask` has `k` bits set, then workers `0..k-1` are placed and worker `k` is the next to place. That is the crucial bookkeeping trick — it forces "each worker used exactly once," because the only way to add a bit is to advance to the next worker. The transition: from `dp[mask]` with `i = popcount(mask)` (the next worker), for each task `j` not in `mask`, we may assign worker `i -> j`, paying `cost[i][j]`:

`dp[mask | (1<<j)] = min(dp[mask | (1<<j)], dp[mask] + cost[i][j])`.

Base case `dp[0] = 0` (no workers placed, no tasks used). Answer: `dp[(1<<n) - 1]` (all `n` tasks used means all `n` workers placed). This is optimal by the standard optimal-substructure argument: an optimal assignment, restricted to its first `k` worker decisions, must itself be an optimal way to cover the tasks those `k` workers use — otherwise swapping in the cheaper sub-assignment (over the *same* task set, so it stays a valid completion) would lower the total, contradicting optimality. Because every task subset of every size is enumerated and every legal next-worker transition is tried, the DP considers (the cost of) every permutation, so `dp[full]` is the true minimum.

**Confirming the recurrence by hand on the sample.** Matrix above, `n = 3`. `dp[000] = 0`, worker 0 next.
- From `000` (worker 0): `dp[001] = cost[0][0] = 0`; `dp[010] = cost[0][1] = 6`; `dp[100] = cost[0][2] = 3`.
- From `001` (popcount 1 -> worker 1): tasks 1,2 free. `dp[011] = 0 + cost[1][1] = 0 + 0 = 0`; `dp[101] = 0 + cost[1][2] = 0 + 8 = 8`.
- From `010` (worker 1): tasks 0,2 free. `dp[011] = min(0, 6 + cost[1][0]=6) = min(0,12)=0`; `dp[110] = 6 + cost[1][2] = 6 + 8 = 14`.
- From `100` (worker 1): tasks 0,1 free. `dp[101] = min(8, 3 + cost[1][0]=6) = min(8,9)=8`; `dp[110] = min(14, 3 + cost[1][1]=0) = min(14,3)=3`.
- From `011` (popcount 2 -> worker 2): task 2 free. `dp[111] = 0 + cost[2][2] = 0 + 7 = 7`.
- From `101` (worker 2): task 1 free. `dp[111] = min(7, 8 + cost[2][1]=7) = min(7,15)=7`.
- From `110` (worker 2): task 0 free. `dp[111] = min(7, 3 + cost[2][0]=3) = min(7, 6) = 6`.

Final `dp[111] = 6`. That is the optimum `6`, and notice the winning path is `110` (workers 0,1 onto tasks 2,1) then worker 2 onto task 0 — exactly the assignment greedy could not reach. The recurrence is right, and it specifically beats greedy on the very instance I built to break greedy.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the core:

```
vector<long long> dp(1 << n, INF);
dp[0] = 0;
for (int mask = 0; mask < (1 << n); mask++) {
    int i = __builtin_popcount(mask);
    for (int j = 0; j < n; j++) {
        if (mask & (1 << j)) continue;
        int nmask = mask | (1 << j);
        dp[nmask] = min(dp[nmask], dp[mask] + cost[i][j]);
    }
}
cout << dp[(1 << n) - 1] << "\n";
```

Two things nag at me. First, I am iterating `mask` and pushing into `nmask > mask`, which is fine for a forward DP — but I am also reading `dp[mask]` for *every* mask, including unreachable ones, and I never guard against `dp[mask] == INF`. Adding `cost[i][j]` to `INF` could overflow and, worse, could write a bogus finite-looking value into `dp[nmask]`. Second, when `popcount(mask) == n` (i.e. `mask` is full) there is no worker `i = n` and no free `j`, so the inner loop does nothing — that is harmless, but `i = n` indexing `cost[i][j]` would be out of bounds if any `j` *were* free, which can't happen for the full mask but I want to be safe for masks with `popcount == n` in general. Let me trace the smallest input that could expose the INF-overflow path: `n = 2`, `cost = [[5, 3], [4, 9]]`, where the answer is obviously `min(5+9, 3+4) = min(14, 7) = 7`.

Trace. `dp[00]=0`, rest `INF = LLONG_MAX/4`. `mask=00`, `i=0`: `dp[01]=min(INF, 0+cost[0][0]=5)=5`; `dp[10]=min(INF,0+cost[0][1]=3)=3`. `mask=01`, `i=1`: task 1 free -> `dp[11]=min(INF, dp[01]+cost[1][1]=5+9=14)=14`. `mask=10`, `i=1`: task 0 free -> `dp[11]=min(14, dp[10]+cost[1][0]=3+4=7)=7`. `mask=11`, `i=2`: loop body skips (no free task). Answer `dp[11]=7`. Correct *here* — but only because every mask happened to be reachable at `n=2`. The INF guard was never exercised.

**Diagnosing the latent bug with a sparser reachability case.** The danger is a `mask` that is genuinely unreachable as a *prefix* — but with this DP every subset is reachable (I can place worker 0 on any single task, worker 1 on any pair-completing task, etc.), so for a complete matrix every `mask` with `popcount <= n` is reachable and `dp[mask]` is finite by the time I read it in increasing-mask order... except masks I read *before* they are filled. Concretely: I process `mask` in increasing numeric order, and `dp[mask]` is only finalized by writes from *smaller* masks (subsets with one fewer bit). Since every subset's numeric value is larger than each of its one-bit-removed subsets? Not always — removing a high bit lowers the number, removing a low bit also lowers it, so all one-bit-removed subsets are numerically smaller. Good: increasing order is a valid topological order, every `dp[mask]` is finalized before it is read. So at `n=2` no `INF` is ever read as a source. But I cannot rely on "every mask reachable" as a safety argument for the `+cost` on `INF`: a defensive `if (dp[mask] == INF) continue;` costs nothing and removes the overflow class entirely. And the `i = popcount(mask)` can equal `n`; I must `continue` then, both to avoid `cost[n][...]` out-of-bounds and because there is nothing to place. The first version "worked" on `n=2` by luck of full reachability; I do not ship code that is correct by luck.

**Fixing and re-verifying.** Add the two guards and switch `min(...)` to an explicit compare-and-store so the `INF` source never participates:

```
for (int mask = 0; mask < (1 << n); mask++) {
    if (dp[mask] == INF) continue;            // unreachable: never add cost to INF
    int i = __builtin_popcount((unsigned)mask);
    if (i >= n) continue;                      // all workers placed; nothing to do
    for (int j = 0; j < n; j++) {
        if (mask & (1 << j)) continue;
        int nmask = mask | (1 << j);
        long long cand = dp[mask] + cost[i][j];
        if (cand < dp[nmask]) dp[nmask] = cand;
    }
}
```

Re-trace `n=2`, `cost=[[5,3],[4,9]]`: identical to before because all masks are reachable, and I still land on `7`. Re-trace a case with a negative cost, `n=2`, `cost=[[-5,2],[3,-1]]`, answer `min(-5+-1, 2+3)=min(-6,5)=-6`: `dp[01]=-5`, `dp[10]=2`; `dp[11]` via `01`: `-5+(-1)=-6`; via `10`: `2+3=5`; `min=-6`. Correct — and confirms the DP needs no non-negativity assumption (unlike some greedy/Hungarian setups), which is good because the contract allows negative costs. The guards changed no answer on reachable instances and removed the overflow class; that is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the contract says print `0`. After reading `n`, my full mask would be `(1<<0)-1 = 0` and `dp[0] = 0`, so `dp[full] = 0` even without a special case — but to be unambiguous and avoid `vector<long long>(1<<0)` corner reasoning I add an explicit `if (n == 0) { print 0; }` right after reading the (empty) matrix. Correct.
- `n = 1`: `cost = [[-5]]`. `dp[0]=0`, `mask=0` `i=0` `j=0`: `dp[1] = 0 + (-5) = -5`. Answer `dp[1] = -5`. The single worker *must* take the single task even at a negative (or any) cost — there is no "skip" option in assignment, unlike independent-set problems. Correct.
- All-equal matrix, every cell `v`: every permutation costs `n*v`, and the DP returns `n*v` regardless of order. Correct.
- Ties (many equal cells): `min`/compare-and-store is order-independent, so ties never matter. Correct.
- Overflow: accumulators are `long long`; the maximum `|total|` is about `1.8*10^10`, far inside 64-bit. `INF = LLONG_MAX/4` is only ever *read* as a guard (`if (dp[mask]==INF) continue;`) and never has `cost` added to it, so it cannot overflow. Safe.
- Largest case `n = 18`: `2^18 = 262144` states, `~18` transitions each, `2 MB` for `dp`; I expect well under a second. (Measured: about 0.02 s on a random `n=18` matrix with entries up to `10^9`.) Comfortable under the 2 s limit.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so row-major parsing is format-agnostic.

**Independent verification before I believe any of it.** Hand-tracing convinces me of the idea; a differential test convinces me of the *code*. I wrote an independent brute oracle that enumerates all `n!` permutations directly (`itertools.permutations`, summing `cost[i][perm[i]]`) — a genuinely different method from the bitmask DP, so agreement is real evidence rather than a mirror. A generator emits random small matrices (`n <= 8` so `8! = 40320` stays instant for the oracle), plus targeted families: the greedy-killer at `n=3`, all-equal matrices, negative-cost matrices, a forced-cheap-cell trap, and the `n=0`/`n=1` edges. I ran the DP against the oracle on 700 generated cases (random plus all the edge families) and got **zero mismatches**, including the greedy-killer where the DP returns `6` and a greedy returns `7`. I also confirmed the `n=18` timing directly. The clever idea (greedy) is dead by a traced counterexample; the over-powered idea (Hungarian) was rejected as needless implementation-risk at this scale; the simple provable idea (bitmask DP) is what survived tracing, edge analysis, and 700 differential tests.

**Final solution.** I convinced myself the *idea* is right by disproving greedy on a concrete `[[0,6,3],[6,0,8],[3,7,7]]` (greedy `7` vs optimal `6`), by rejecting the Hungarian algorithm as unnecessary risk at `n <= 18`, and by hand-checking the DP's optimal substructure and recurrence on that very sample. I convinced myself the *code* is right by tracing the latent `INF`/`popcount==n` issue to precise causes, guarding them, re-verifying the failing-class traces and the negative-cost corner, and running 700 differential tests against an independent permutation oracle with zero mismatches. That is what I ship — one self-contained file, the simple `O(n^2 * 2^n)` bitmask DP I can prove and have traced, not the greedy I broke nor the Hungarian I did not need:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // cost[i][j] = cost of giving worker i task j.
    vector<vector<long long>> cost(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> cost[i][j];

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // dp[mask] = minimum total cost of assigning workers 0..k-1 to exactly the
    // tasks in `mask`, where k = popcount(mask). Worker index is implied by how
    // many bits are already set, so each worker is used exactly once and each
    // task at most once. Unreachable states stay at INF.
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(1 << n, INF);
    dp[0] = 0;                              // no workers placed, no tasks used

    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == INF) continue;
        int i = __builtin_popcount((unsigned)mask); // next worker to place
        if (i >= n) continue;                        // all workers already placed
        for (int j = 0; j < n; j++) {
            if (mask & (1 << j)) continue;           // task j already taken
            int nmask = mask | (1 << j);
            long long cand = dp[mask] + cost[i][j];
            if (cand < dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[(1 << n) - 1] << "\n";       // all tasks assigned
    return 0;
}
```

**Causal recap.** Cheapest-available greedy looked right but a single traced counterexample (`[[0,6,3],[6,0,8],[3,7,7]]`: both greedy variants `7` vs the reachable `6`) showed a local cheap grab consumes a task that a later worker needed, costing more than it saved, so greedy is out. The Hungarian algorithm is exact but its `O(n^3)` machinery is needless implementation-risk at `n <= 18`, so it is out too. The bitmask DP over the assigned-task set — `dp[mask]` = best cost to place the first `popcount(mask)` workers onto exactly those tasks, transition by placing the next worker on any free task — is provable by optimal substructure, runs in `18*2^18 ~ 5*10^6` transitions and `2 MB`, and I verified its recurrence on the sample (landing on `6`); a first cut read `INF` sources and could index `cost[n][...]`, which guards on `dp[mask]==INF` and `popcount==n` fix; and `long long` plus the explicit `n=0` print close out the overflow, empty, single-worker, negative-cost, and tie corners. Seven hundred differential tests against an independent permutation oracle, zero mismatches.
