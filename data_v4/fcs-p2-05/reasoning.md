The scale decides everything here: `n <= 18` and `|cost[i][j]| <= 10^9`. Two consequences fall out. A full assignment sums `n` cells, so `|total|` can reach `18 * 10^9 = 1.8*10^10` — past the `~2.1*10^9` ceiling of a 32-bit `int`, so every matrix entry and every accumulator has to be `long long`. That wouldn't crash; it would be a silent wrong answer surfacing only on the large-magnitude tests, so I lock it down first. And `n <= 18` puts `2^n` subsets in reach — exactly the regime where a bitmask DP over task-subsets beats anything heavier. One corner from the contract: `n = 0` must print `0`, the empty assignment; the I/O is plain — read `n`, then `n*n` entries row-major, print one integer.

The cheapest thing to type is cheapest-available greedy: repeatedly lock the smallest still-free cell and forbid its worker and task, or the row-by-row cousin where each worker in index order grabs its cheapest free task. Both are `O(n^2 log n)` and a few lines. But the perfect-matching constraint is global while greedy commits locally — the classic setup for greedy to be wrong — so I try to break it. `n = 2` can't discriminate: there are only two permutations and greedy effectively compares them. At `n = 3` this matrix does the job:

```
[ 0  6  3 ]
[ 6  0  8 ]
[ 3  7  7 ]
```

Greedy grabs the two `0`s at `cost[0][0]` and `cost[1][1]`, which forces worker 2 onto task 2 at `7`, total `0+0+7 = 7`. But `0->2, 1->1, 2->0` costs `3+0+3 = 6`. The `0`-grab at `cost[0][0]` consumed task 0, worker 2's only cheap option (`3` versus `7,7`), and the forced completion cost more than the local grab saved. Both greedy variants land on `7`; the optimum is `6`. Greedy is out — and this instance doubles as a clean discriminating test.

The other exact route is the Hungarian algorithm at `O(n^3)`. It is correct, but its potentials, augmenting paths, and dual updates are real bug-surface where a subtle slip yields a plausible-but-wrong number on the adversarial test, and its better asymptotics buy nothing until `n` is in the hundreds. At `n <= 18` the bitmask DP is exact, provable in two sentences, and fast enough — I take it.

**The DP.** Place workers in index order. The only thing future decisions depend on is *which* tasks are already used, not who took them, so the state is a subset `mask` of tasks and

- `dp[mask]` = minimum cost to place workers `0..k-1` onto exactly the tasks in `mask`, where `k = popcount(mask)`.

The next worker to place is `i = popcount(mask)`, implied by the mask — which is precisely what enforces "each worker used once," because the only way to add a task bit is to advance exactly one worker. Transition: for each task `j` not in `mask`,

`dp[mask | (1<<j)] = min(dp[mask | (1<<j)], dp[mask] + cost[i][j])`.

Base `dp[0] = 0`; answer `dp[(1<<n) - 1]`. Optimal substructure gives correctness: an optimal assignment, restricted to its first `k` workers, must itself optimally cover the task set those workers use — otherwise swapping in a cheaper sub-assignment over the *same* task set (still a valid completion) would lower the total, a contradiction. Every subset of every size and every legal next-worker transition is enumerated, so `dp[full]` is the true minimum over all `n!` permutations.

The cost is affordable: `2^18 = 262144` states times up to `18` transitions is `~4.7*10^6` operations, comfortably under 2 s, and `dp` is one `long long` per state, `2 MB` against a 256 MB limit.

I iterate `mask` upward and push into the strictly larger `mask | (1<<j)`. That order is valid because every one-bit-removed subset is numerically smaller, so `dp[mask]` is already finalized when I read it. With a complete cost matrix every subset is in fact reachable, so no `dp[mask]` is ever `INF` at read time — but I still guard `if (dp[mask] == INF) continue;` so the sentinel never has a cost added to it (which would both write a bogus finite value into an unreachable state and risk overflow), and `if (popcount(mask) >= n) continue;` so `cost[i][j]` is never indexed at the nonexistent worker `i = n`. Both guards cost nothing and close those failure classes outright rather than leaving correctness resting on reachability luck.

Tracing the `n=3` sample, the DP lands on `6` via `dp[110]` (workers 0,1 on tasks 2,1) then worker 2 onto task 0 — the exact assignment greedy could not reach. On `n=2`, `cost=[[-5,2],[3,-1]]`, it gives `min(-5-1, 2+3) = -6`: the recurrence assumes nothing about sign, so the contract's negative costs are handled directly, with none of the non-negativity fuss some Hungarian setups want.

The remaining corners fall out cleanly. `n = 0`: `dp[(1<<0)-1] = dp[0] = 0` already, but I add an explicit early print so I never reason about `vector<long long>(1<<0)`. `n = 1`: the lone worker takes the lone task at whatever cost — assignment has no "skip" option, so a negative single cell is correct, not something to floor at zero. All-equal matrices give `n*v` for every permutation, and ties are order-independent under compare-and-store. `INF = LLONG_MAX/4` is only ever read as a guard, never accumulated into, so it cannot overflow.

For the code I lean on a differential test against an independent oracle that enumerates all `n!` permutations directly — a different method, so agreement is real evidence — over 700 cases: random small matrices (`n <= 8`, so `8!` stays instant for the oracle), plus the greedy-killer, all-equal, negative-cost, forced-cheap-cell, and `n=0`/`n=1` families. Zero mismatches, and a direct `n=18` run finished in about 0.02 s, far under the limit.

The full program is in the answer: the `O(n^2 * 2^n)` bitmask DP, `long long` throughout, the two guards, and the explicit `n=0` print.
