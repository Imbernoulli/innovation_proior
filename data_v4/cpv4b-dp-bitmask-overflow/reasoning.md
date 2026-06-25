**Reading the problem and pinning the contract.** I have `n` couriers and `n` zones, a cost matrix `c[i][j]` for sending courier `i` to zone `j`, and I must pick a one-to-one assignment — a permutation `p` with courier `i` going to zone `p[i]` — minimizing `sum_i c[i][p[i]]`. Input is `n` then the `n*n` matrix row by row; output is one integer, the minimum total. Before any algorithm I fix the scale, because it decides the data types. The constraints say `n <= 20` and `0 <= c[i][j] <= 10^9`. A full assignment has `n` edges, so its cost can be as large as `20 * 10^9 = 2*10^10`. The signed 32-bit ceiling is `2^31 - 1 = 2147483647`, about `2.147*10^9`. So the largest legitimate answer is roughly **nine to ten times** what an `int` can hold. Every accumulator that sums edge costs must be 64-bit. I write that down now and will come back to it — this is exactly the kind of problem where an `int` is a silent wrong answer on the big tests, not a crash.

**Laying out the candidate approaches.** Three routes are on the table.

- *Brute force over permutations.* Try all `n!` assignments and take the min. It is obviously correct and I will use it as my oracle, but `20!` is about `2.4*10^18` — utterly infeasible as the shipped solution. Good for checking, useless for submitting.
- *Greedy per courier.* For each courier in turn, send it to its currently-cheapest free zone. Three lines, `O(n^2)`. The danger is structural: two couriers may both want the same cheap zone, and resolving that locally can wreck the global total. I will try to break this before trusting it.
- *Bitmask DP over used zones.* Place couriers in the fixed order `0, 1, 2, ...`; the count of already-used zones equals the number of couriers placed, which equals `popcount(mask)`. So the mask alone tells me which courier is next, and the state is just the mask. `dp[mask]` = min cost to assign couriers `0 .. popcount(mask)-1` onto exactly the zones in `mask`. `O(2^n * n)` time, `O(2^n)` memory. This is the one I expect to ship; I need to get the recurrence and the arithmetic exactly right.

**Stress-testing greedy before committing.** "Greedy feels fine" is how wrong solutions get shipped, so let me attack it with a concrete instance. Take `n = 2`, costs

```
c[0][0]=1  c[0][1]=2
c[1][0]=1  c[1][1]=9
```

Greedy processes courier 0 first: its cheapest free zone is zone 0 (cost 1). Then courier 1 has only zone 1 left, cost 9. Greedy total `1 + 9 = 10`. But the other assignment — courier 0 -> zone 1 (cost 2), courier 1 -> zone 0 (cost 1) — totals `3`. Greedy is off by more than a factor of three because it grabbed a cheap edge for courier 0 that courier 1 needed far more. So greedy-per-courier is wrong, and I see *why*: the cheapest single edges (`c[0][0]=1` and `c[1][0]=1`) both point at zone 0, which is an illegal assignment, and resolving the conflict by first-come-first-served is arbitrary. Greedy is out. (The problem statement even warns that the cheapest edges can form an illegal assignment — this is that case.)

**Deriving the DP and checking the recurrence on paper.** The order of placement is fixed: courier `k = popcount(mask)` is the next to assign when the current used-zone set is `mask`. The transition is: pick any unused zone `z` (a clear bit of `mask`), send courier `k` there at cost `c[k][z]`, and move to `mask | (1<<z)`:

- `dp[mask | (1<<z)] = min over z not in mask of ( dp[mask] + c[popcount(mask)][z] )`.

Base case `dp[0] = 0` (no courier placed, no zone used, cost 0). The answer is `dp[(1<<n) - 1]` — every zone used means every courier placed. Let me confirm on the sample `c = [[4,2,8],[4,3,7],[3,1,6]]`, claimed answer `12` via courier0->zone0 (4), courier1->zone2 (7), courier2->zone1 (1).

I push forward from `dp[0]=0`. Courier 0 is placed from masks with popcount 0, i.e. only `mask=000`. From `dp[000]=0`: zone0 -> `dp[001]=4`, zone1 -> `dp[010]=2`, zone2 -> `dp[100]=8`. Courier 1 is placed from popcount-1 masks. From `dp[001]=4` (courier1): zone1 -> `dp[011]=4+3=7`, zone2 -> `dp[101]=4+7=11`. From `dp[010]=2`: zone0 -> `dp[011]=min(7, 2+4)=6`, zone2 -> `dp[110]=2+7=9`. From `dp[100]=8`: zone0 -> `dp[101]=min(11, 8+4)=11`, zone1 -> `dp[110]=min(9, 8+3)=9`. Courier 2 is placed from popcount-2 masks. From `dp[011]=6` the only free zone is 2: `dp[111]=6 + c[2][2]=6+6=12`. From `dp[101]=11` free zone 1: `dp[111]=min(12, 11 + c[2][1]=11+1)=12`. From `dp[110]=9` free zone 0: `dp[111]=min(12, 9 + c[2][0]=9+3)=12`. Final `dp[111]=12`. Matches the sample, and the path `dp[010]=2 -> dp[011]=6 -> dp[111]=12` is courier0->zone1(2), courier1->zone0(4)... wait, let me not over-claim the *which-permutation*; the value is what matters and it is `12`, agreeing with the contract. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core:

```
int full = 1 << n;
vector<long long> dp(full, INF);
dp[0] = 0;
for (int mask = 0; mask < full; mask++) {
    if (dp[mask] == INF) continue;
    int courier = __builtin_popcount(mask);
    for (int z = 0; z < n; z++) {
        if (mask & (1 << z)) continue;
        int nmask = mask | (1 << z);
        dp[nmask] = min(dp[nmask], dp[mask] + c[courier][z]);
    }
}
cout << dp[full - 1] << "\n";
```

I want to trace the smallest input that could expose a structural slip. Use `n = 1`, `c = [[7]]`, answer obviously `7`. `full = 2`. `dp = [0, INF]`. `mask=0`: `dp[0]=0`, `courier = popcount(0) = 0`. Loop `z=0`: bit not set, `nmask = 1`, `dp[1] = min(INF, 0 + c[0][0]=7) = 7`. `mask=1`: `dp[1]=7 != INF`, `courier = popcount(1) = 1`. Now the inner loop runs `z=0`: bit 0 *is* set in mask 1, so `continue`; loop ends with nothing assigned. Output `dp[1] = 7`. Correct here.

But that `mask=1` step bothers me: `courier` became `1`, which is `n`, and I then indexed `c[courier][z] = c[1][z]` *conceptually* in the loop body — it was only saved because every zone bit was already set so the body never ran. Let me trace a case where a full mask still has the body reachable... actually for a *full* mask all zone bits are set, so the inner body always `continue`s. The real danger is a mask with popcount `== n` but that is necessarily the full mask, so the body never executes. So no out-of-bounds here. Still, I will guard `courier == n` explicitly so the intent is visible and `c[courier]` is never even contemplated out of range. Let me also trace `n=2` on the greedy-breaker `c=[[1,2],[1,9]]`, true answer `3`. `dp=[0,INF,INF,INF]`. `mask=00`: courier0. z0 -> `dp[01]=0+1=1`; z1 -> `dp[10]=0+2=2`. `mask=01`(=1): courier1. z0 set -> skip; z1 -> `dp[11]=min(INF, 1 + c[1][1]=9)=10`. `mask=10`(=2): courier1. z0 -> `dp[11]=min(10, 2 + c[1][0]=1)=3`; z1 set -> skip. `mask=11`: courier2==n, body all-skip. Output `dp[11]=3`. Correct — and it beats greedy's 10, exactly as the math promised.

**Diagnosing the real bug: a 32-bit accumulator that silently overflows, caught by tracing a large case.** The small traces pass, which is precisely the trap — the defect this problem is built around does not show on small numbers. I deliberately construct a large-magnitude case and trace the arithmetic. Take `n = 3` with every entry equal to `800000000` (`8*10^8`):

```
800000000 800000000 800000000
800000000 800000000 800000000
800000000 800000000 800000000
```

Every permutation costs `3 * 800000000 = 2400000000`, so the answer is `2.4*10^9`. Now suppose — as in a first instinct — I had declared `c` as `vector<vector<int>>` and accumulated with `int cand = dp[mask] + c[courier][z];` (or kept `dp` itself as `int`). Trace the accumulation along one assignment path. Start `0`. Place courier 0: `0 + 800000000 = 800000000` — fits in `int` (`< 2147483647`). Place courier 1: `800000000 + 800000000 = 1600000000` — still fits (`1.6*10^9 < 2.147*10^9`). Place courier 2: `1600000000 + 800000000 = 2400000000`. **This exceeds `2^31 - 1 = 2147483647`.** In 32-bit two's complement the value wraps to `2400000000 - 2^32 = 2400000000 - 4294967296 = -1894967296`. So an `int` accumulator reports the optimal total as `-1894967296` — a *negative* delivery cost, which is nonsense, with no crash and no warning.

I verified the wrap value arithmetically: `((2400000000 + 2^31) mod 2^32) - 2^31 = -1894967296`, and the two earlier partial sums `800000000` and `1600000000` are both below the ceiling, which is why the overflow appears only on the *third* addition — i.e. only on large enough `n` with large enough entries. This is exactly the "trace a large case to catch the overflow" episode: the bug is invisible on `n = 1` and `n = 2` small-value traces and only surfaces when a partial sum first crosses `2.147*10^9`. The fix is to hold `c`, `dp`, and every intermediate sum in `long long` (range about `9.2*10^18`, dwarfing the `2*10^10` worst case). I had already typed `vector<vector<long long>> c` and `vector<long long> dp` in the framework — the lesson here is that the *intermediate* `dp[mask] + c[courier][z]` must also be `long long`, which it is automatically once both operands are `long long`. If I had written `int cand = ...` the right-hand side would still compute in `long long` but the assignment would truncate; so I keep the candidate as `long long cand` (or fold the `min` inline on `long long` operands).

**A second bug: a fragile INF sentinel.** With everything in `long long`, what value is `INF`? My first instinct was `INF = 1e9`. Trace why that is wrong: real costs reach `10^9`, and a partial sum reaches `2*10^10`, so `1e9` is *smaller* than achievable real totals — a legitimate `dp[mask]` could exceed `INF` and I would mistake a reachable state for unreachable, or worse, `min` against a too-small INF would clobber real answers. Concretely, on the all-`8*10^8` case, after two placements `dp[mask] = 1.6*10^9 > 1e9 = INF`, so my `if (dp[mask] == INF) continue;` would not skip it (good, it is not exactly equal), but my *initialization* of unreached masks to `1e9` means an unreached mask looks cheaper than a reached one costing `1.6*10^9` — the `min` would then never prefer the real state and `dp[full-1]` could come out as the bogus `1e9`. So `INF` must be larger than any reachable total (`> 2*10^10`) yet safe to add to without overflowing `long long`. I pick `INF = 4e18`. Check the headroom: I never *add* `c[...]` to a state I have not skipped — I guard `if (dp[mask] == INF) continue;` before using `dp[mask]` — so `INF` is never an operand of `+`, and even if it were, `4e18 + 10^9 < 9.2*10^18` stays in range. Safe. (I also drop the `INF == 1e9` idea entirely; the only role of INF is "unreached", and a value above every reachable total plays it correctly.)

**Fixing and re-verifying.** The corrected core, with the `courier == n` guard, `long long` candidate, and a large safe INF:

```
const long long INF = (long long)4e18;
int full = (1 << n);
vector<long long> dp(full, INF);
dp[0] = 0;
for (int mask = 0; mask < full; mask++) {
    if (dp[mask] == INF) continue;
    int courier = __builtin_popcount(mask);
    if (courier == n) continue;
    for (int z = 0; z < n; z++) {
        if (mask & (1 << z)) continue;
        int nmask = mask | (1 << z);
        long long cand = dp[mask] + c[courier][z];   // long long: no 32-bit wrap
        if (cand < dp[nmask]) dp[nmask] = cand;
    }
}
cout << dp[full - 1] << "\n";
```

Re-run the all-`8*10^8`, `n=3` case end to end in my head along the surviving path: `dp[0]=0`; courier0 to some zone -> `8*10^8`; courier1 -> `1.6*10^9`; courier2 -> `2.4*10^9`. Every operand is `long long`, the sum `2400000000` is stored exactly, and `dp[111] = 2400000000`. That is the correct answer, and crucially it is *positive*, unlike the `-1894967296` the `int` version produced. The overflow is gone for the reason I diagnosed. I also re-confirmed the sample `12` and the greedy-breaker `3` are unaffected, since those use the same corrected code.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `full = 1<<0 = 1`. `dp = [0]` (size 1, `dp[0]=0`). The mask loop runs only `mask=0`, where `courier = popcount(0) = 0 == n`, so `if (courier == n) continue;` skips the body. Output `dp[full-1] = dp[0] = 0`. The empty assignment costs `0` — correct, and the `courier == n` guard is what keeps `c[0][...]` from being touched when there are no rows.
- `n = 1`, `c = [[0]]`: `dp[0]=0`; courier0 -> `dp[1] = 0 + 0 = 0`; output `0`. Correct.
- All entries equal, `n = 4`, every `c = 5`: every permutation costs `4*5 = 20`; the DP fills every `dp[mask]` for popcount `k` with `5*k` and outputs `20`. Correct (ties are handled by `min` naturally).
- Large `n = 20`, entries near `10^9`: worst total about `2*10^10`, well inside `long long`; `2^20 = 1048576` masks times `20` zones is about `2.1*10^7` transitions — fast inside 2 s. Confirmed feasible.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the row-by-row matrix parses regardless of exact spacing.

**Numeric self-check of the complexity bound.** I claimed `O(2^n * n)` is comfortable for `n = 20`. Let me put numbers on it rather than assert it. Masks: `2^20 = 1048576`. For each non-INF mask the inner loop runs `n = 20` iterations. So the transition count is at most `1048576 * 20 = 20971520 ≈ 2.1*10^7`. At a few hundred million simple operations per second, that is well under `0.1` s of pure work, and even with `__builtin_popcount` and vector indexing the constant factor leaves wide margin against the 2 s limit. Memory: one `long long` per mask, `1048576 * 8 bytes = 8388608 ≈ 8 MB`, far under 256 MB. The bound checks out concretely.

**Final solution.** I disproved greedy with a traced two-zone conflict, hand-verified the bitmask recurrence on the sample to `12`, and — the heart of this problem — caught a silent 32-bit overflow by tracing a deliberately large case (`3 * 8*10^8 = 2.4*10^9` wrapping to `-1894967296` in `int`), then fixed it by keeping every cost, state, and intermediate sum in `long long` with an INF safely above every reachable total. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> c(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> c[i][j];

    const long long INF = (long long)4e18;
    // dp[mask] = minimum cost to have assigned couriers 0..popcount(mask)-1
    // to exactly the set of zones in `mask`.
    int full = (1 << n);
    vector<long long> dp(full, INF);
    dp[0] = 0;
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == INF) continue;
        int courier = __builtin_popcount(mask); // next courier to place
        if (courier == n) continue;
        for (int z = 0; z < n; z++) {
            if (mask & (1 << z)) continue;       // zone z already used
            int nmask = mask | (1 << z);
            long long cand = dp[mask] + c[courier][z];
            if (cand < dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[full - 1] << "\n";
    return 0;
}
```

**Causal recap.** Greedy-per-courier looked plausible but a traced two-zone conflict (`[[1,2],[1,9]]`: greedy 10 vs the reachable 3) showed local cheap-edge grabs create illegal or expensive global assignments, so I moved to the bitmask DP and verified its `dp[mask|(1<<z)] = min(dp[mask] + c[popcount(mask)][z])` recurrence on the sample to `12`; the small traces all passed, which masked the real defect — accumulating edge costs in a 32-bit `int`, exposed only by tracing a large case where the third addition `1.6*10^9 + 8*10^8 = 2.4*10^9` overflows and wraps to `-1894967296`; promoting every cost, the `dp` array, and the intermediate `cand` to `long long`, with `INF = 4*10^18` set safely above the `2*10^10` worst-case total (and never used as an addition operand), removes the wrap; and the `courier == n` guard plus `max`-free `min` accumulation close out the `n = 0`, `n = 1`, all-equal, and large-`n` corners while the `2.1*10^7`-transition count keeps it well inside the time limit.
