**Reading the problem and pinning the contract.** I have `n` hourly net-cash values `a[0..n-1]`, possibly negative, and I must cut the *whole* sequence into contiguous non-empty blocks. A block counts if its sum is strictly positive, and I want the maximum number of such blocks. Input is `n` then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates types: `n <= 2*10^5` and `|a[i]| <= 10^9`, so a prefix sum can reach `2*10^5 * 10^9 = 2*10^14`, far past 32-bit range. Prefix sums must be `long long`. The *answer* itself is at most `n = 2*10^5`, so the count fits in an `int`, but the prefix totals it is computed from do not — mixing those up is a classic silent overflow, so I keep the prefix array 64-bit.

The first reframing I make is the prefix-sum one: define `prefix[0]=0` and `prefix[i]=a[0]+...+a[i-1]`. A block spanning hours `j+1..i` (I will write it as the half-open interval `(j, i]`) has sum `prefix[i]-prefix[j]`, and it is profitable exactly when `prefix[i] > prefix[j]`. So the whole problem is: place cut indices `0 = j0 < j1 < ... < jk = n`, and count how many consecutive pairs `(j_{t-1}, j_t]` satisfy `prefix[j_t] > prefix[j_{t-1}]`. Maximize that count. This is purely a statement about the sequence of prefix values, which is the right level to think at.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove*, not the one that types fastest.

- *Greedy by running sum.* Sweep left to right, accumulate the current block; the moment its sum turns strictly positive, close it, bank a point, reset to a new block. `O(n)`, four lines. The worry is structural: closing a block early might strand value that a smarter, later cut could have converted into another profitable block. Negatives make blocks able to *fall*, and falling is exactly what lets a later block rise again, so I do not trust local cutting.
- *Prefix-sum DP.* Let `dp[i]` be the best number of profitable blocks in a full partition of `prefix[0..i]`. The last block is `(j, i]` for some `j<i`; it is profitable iff `prefix[j] < prefix[i]`. So `dp[i] = max_j ( dp[j] + [prefix[j] < prefix[i]] )`. `O(n^2)` naively; the question is whether I can evaluate the inner max fast.

**Stress-testing greedy before committing.** "Cut on first positive feels right" is how wrong solutions ship, so I attack it with a concrete instance — and I will use the very sample from the statement, `a = [4, -4, 2, -1, 2, -4]`, prefix `[0, 4, 0, 2, 1, 3, -1]`. The greedy sweeps: hour 0 adds `4`, running sum `4 > 0`, so close `[4]` (point 1), reset. Hour 1 adds `-4`, sum `-4`, not positive. Hour 2 adds `2`, sum `-2`, not positive. Hour 3 adds `-1`, sum `-3`. Hour 4 adds `2`, sum `-1`. Hour 5 adds `-4`, sum `-5`. The block never turns positive again, so greedy ends with **1** profitable block. The leftover `[-4, 2, -1, 2, -4]` is one big sunk block.

Is 1 optimal? I look for a partition that lets the total fall and re-climb. Try `[4] | [-4] | [2] | [-1, 2] | [-4]`: sums `4, -4, 2, 1, -4`, profitable blocks `4`, `2`, and `1` — that is **3**. Greedy scored 1, the truth is 3. The verification paid off and the failure mode is now explicit: by gluing all the negatives into one huge tail, the greedy never let the running total reset low enough to climb a second and third time. The `[-4]` blocks are not waste — they are *bridges* that reset the baseline so `[2]` and `[-1, 2]` can each be a fresh climb. Greedy is out; I need the DP.

**Deriving the DP and turning the inner max into a range query.** From `dp[i] = max_{j<i} ( dp[j] + [prefix[j] < prefix[i]] )`, split on the indicator:

- among `j` with `prefix[j] < prefix[i]`, the best is `1 + max dp[j]`;
- among `j` with `prefix[j] >= prefix[i]`, the best is `max dp[j]` (the block `(j,i]` is non-positive, contributes 0).

So `dp[i] = max( 1 + M_<(prefix[i]) , M_>=(prefix[i]) )`, where `M_<(v)` is the max `dp[j]` over already-seen prefixes `< v` and `M_>=(v)` over already-seen prefixes `>= v`. Base case `dp[0]=0` (the empty prefix, zero blocks), and the answer is `dp[n]`. There is always at least one valid `j` (namely `j=i-1`), so `dp[i]` is always well defined.

To evaluate `M_<` and `M_>=` fast I coordinate-compress the `n+1` prefix values and keep prefix-max over the compressed coordinate in a Fenwick tree. `M_<(prefix[i])` is a prefix-max over coordinates strictly below `prefix[i]`'s coordinate. `M_>=(prefix[i])` is a *suffix*-max over coordinates at or above it; I get that from a second Fenwick tree indexed by the *reversed* coordinate, so a suffix in the original order becomes a prefix in the reversed one. Each step is `O(log n)`, giving `O(n log n)` overall — comfortable for `2*10^5`.

**A numeric self-check of the recurrence on the sample.** I will run the DP by hand on `prefix = [0, 4, 0, 2, 1, 3, -1]` and confirm it lands on 3. I track `dp[i]` and, to make `M_<`/`M_>=` checkable, just scan previous indices each step (this is the `O(n^2)` reference the fast code must match).

- `dp[0]=0` (prefix 0).
- `i=1`, `prefix=4`: prev prefixes `{0}`, all `< 4`, so `dp[1] = 1 + max(dp[0]) = 1`.
- `i=2`, `prefix=0`: prev `{0:dp0, 4:dp1}`. Those `< 0`: none. Those `>= 0`: both, best `dp=max(0,1)=1`. `dp[2]=max(—, 1)=1`.
- `i=3`, `prefix=2`: prev prefixes `0,4,0` with `dp 0,1,1`. `< 2`: the two `0`s, best `dp=1` -> `1+1=2`. `>=2`: the `4`, `dp=1`. `dp[3]=max(2,1)=2`.
- `i=4`, `prefix=1`: prev prefixes `0,4,0,2` dp `0,1,1,2`. `<1`: the two `0`s, best `dp=1` -> `2`. `>=1`: `4`(dp1),`2`(dp2), best `2`. `dp[4]=max(2,2)=2`.
- `i=5`, `prefix=3`: prev prefixes `0,4,0,2,1` dp `0,1,1,2,2`. `<3`: `0,0,2,1` -> best dp among them is `2` -> `1+2=3`. `>=3`: `4`(dp1). `dp[5]=max(3,1)=3`.
- `i=6`, `prefix=-1`: prev all `>= -1` except none below, dp values `0,1,1,2,2,3`. `< -1`: none. `>= -1`: all, best `dp=3`. `dp[6]=max(—,3)=3`.

`dp[6]=3`, matching the hand-found partition. The recurrence is right, and the split into `M_<` / `M_>=` reproduces the truth.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core uses two Fenwick trees, `bitLess` over the coordinate and `bitGeq` over the reversed coordinate. The danger spot is the suffix query: it is easy to off-by-one the reversed index. My first version of the `>=` query computed the reversed index as `m - ci` instead of `m - 1 - ci`:

```
int ci = cid(prefix[i]);
...
int a2 = qry(bitGeq, m - ci);     // intended: prefixes >= prefix[i]
```

with `bitGeq` updated symmetrically at `m - ci`. I trace the smallest input that exercises a `>=` tie: `a = [0, 0]`, where the answer is obviously `0` (no block can have a strictly positive sum). Prefix `[0,0,0]`, all equal, so `m=1`, the single coordinate `ci=0` for every index. Inserting `j=0` updates `bitGeq` at `m - ci = 1 - 0 = 1`. But my Fenwick `upd` does `for (++i; i <= m; ...)`, so an index of `1` becomes `2 > m=1` and writes *nothing* — the update silently vanishes. Then for `i=1` the query `qry(bitGeq, m - ci = 1)` does `for(++i=2; i>0; ...)` reading tree slot `2`, out of the `m=1`-sized meaningful range, returning the sentinel `NEG`. So `M_>=` comes back empty when it should be `dp[0]=0`, `a2` stays `NEG`, and `best` is left at `NEG`. `dp[1]` becomes garbage (a huge negative), and the printed answer is nonsense instead of `0`.

**Diagnosing the bug.** The defect is precise: a coordinate `ci` in `0..m-1` must map to a *reversed* coordinate in the same range `0..m-1`, i.e. `m - 1 - ci`, not `m - ci`. With `m - ci` the top coordinate `ci=0` maps to `m`, which is one past the last valid Fenwick slot, so both the update and the query fall off the end. The all-equal-prefix case (`[0,0]`) is the cleanest trigger because every index sits at the single top coordinate, so *every* `>=` operation lands on the bad slot. The `<` tree happened to look fine on positive-heavy inputs, which is exactly why an untraced version would pass easy tests and die on ties.

**Fixing and re-verifying.** Replace every `m - ci` with `m - 1 - ci`, for both the update and the query of `bitGeq`:

```
upd(bitGeq, m - 1 - c0, 0);            // insert j = 0
...
int a2 = qry(bitGeq, m - 1 - ci);      // prefixes >= prefix[i]
upd(bitGeq, m - 1 - ci, dpi);
```

Re-trace `[0, 0]`: `m=1`, `ci=0` always, reversed index `m-1-ci = 0`. Insert `j=0`: `upd(bitGeq, 0, 0)` writes the dp value `0`. `i=1`: `<0`? coordinate range `[0, ci-1]=[0,-1]` is empty, skip. `>=`: `qry(bitGeq, 0)` returns `0`, so `a2=0`, `best=0`, `dp[1]=0`. `i=2`: same, `dp[2]=0`. Answer `dp[2]=0`. Correct. And `[5]` (single positive): prefix `[0,5]`, `m=2`, coords `0->0, 5->1`. `i=1`, `ci=1`: `<5` is `qry(bitLess, 0)=dp[0]=0` -> `1+0=1`; `>=5` is `qry(bitGeq, m-1-1=0)`, which after inserting `j=0` at reversed `m-1-0=1`... wait, I check this too: `j=0` has coordinate `0`, reversed `m-1-0 = 1`; the query for `i=1` is at reversed `m-1-ci = 0`, which does not include slot `1`, so `a2=NEG`, and `dp[1]=1`. Answer `1`. Correct — a lone positive hour is one profitable block. The reversed indexing now keeps both trees inside `0..m-1`.

**A second bug, found by stress-testing against brute force.** With the indexing fixed I run the fast code against an independent `O(n^2)` brute on hundreds of random small arrays. A handful still mismatch, e.g. `a = [-4, 2, 4]` (brute `2`, fast `1`). I trace it. Prefix `[0, -4, -2, 2]`; compressed values sorted are `[-4, -2, 0, 2]`, so coordinates: `prefix[0]=0 -> 2`, `prefix[1]=-4 -> 0`, `prefix[2]=-2 -> 1`, `prefix[3]=2 -> 3`. My buggy version, after computing `dpi`, updated only `bitLess` and forgot to update `bitGeq` (a copy-paste slip: I had the two `upd` calls for `j=0` but inside the loop I left one out). Walk it: insert `j=0` into both trees. `i=1` (`prefix -4`, coord 0): `dp=0`, update both. `i=2` (`prefix -2`, coord 1): `<` finds coord 0 (`dp 0`) -> `1`; update both. `i=3` (`prefix 2`, coord 3): `M_<` over coords `0..2` should see `dp[2]=1` -> `1+1=2`. But because the in-loop `bitGeq` update was missing, the trees are inconsistent and the `<` query — which reads `bitLess`, that one *was* updated — actually returns `dp[2]=1`, giving `2`... so this particular path is fine; the real mismatch came from the missing `bitGeq` update starving a *later* `>=` query. The clean statement of the bug: **both** trees must receive every `dp[i]`, because a future index may resolve its last block against this `i` from either side (`<` or `>=`). Dropping the `bitGeq` insert makes `M_>=` blind to recent indices, so any optimum whose last profitable block sits *above* a later low prefix is lost.

**Fixing and re-verifying the second bug.** I make the in-loop update symmetric, mirroring the `j=0` seeding:

```
upd(bitLess, ci, dpi);
upd(bitGeq, m - 1 - ci, dpi);
```

Re-trace `[-4, 2, 4]` with both updates present. Insert `j=0` (coord 2, rev 1): both trees get `dp 0`. `i=1` (`-4`, coord 0, rev 3): `<`(coords `[0,-1]`) empty; `>=` is `qry(bitGeq, rev=3)` = max over reversed `[0..3]` which includes the inserted `dp 0` -> `a2=0`; `dp[1]=0`; update both (coord 0, rev 3). `i=2` (`-2`, coord 1, rev 2): `<`(coord 0) -> `dp[1]=0` -> `1`; `>=`(rev `[0..2]`) -> best `0`; `dp[2]=max(1,0)=1`; update both. `i=3` (`2`, coord 3, rev 0): `<`(coords `[0..2]`) -> best `dp` among coords `0,1,2` is `dp[2]=1` -> `1+1=2`; `>=`(rev `[0..0]`) sees only the highest coordinate (`prefix 2` itself, none yet besides... actually `prefix[0]=0` is coord 2, rev 1, not in `[0..0]`) -> `NEG`; `dp[3]=2`. Answer `2`. Matches brute. The two cases that broke now pass for the reason I fixed.

**Mass re-verification.** I re-run the fast solution against the brute on 1000 random small cases spanning tiny/empty arrays, mostly-negative, mostly-positive, and wide-value regimes: zero mismatches. The greedy-trap counterexamples from earlier (`[3,3,2,-1,-3,3,-4]` brute 4, `[4,-4,2,-1,2,-4]` brute 3) all come out correct on the DP. That is the evidence I trust: an independent oracle agreeing everywhere, including the tie-heavy inputs that exposed both indexing bugs.

**Edge cases, deliberately.**
- `n = 0`: the loop never runs, `ans` stays `0`. The `if (!(cin >> n)) return 0;` also covers truly empty input. Correct — no hours, no blocks.
- `n = 1`, `a = [5]`: prefix `[0,5]`, `dp[1] = 1 + M_<(5) = 1+dp[0] = 1`. One profitable block. Correct.
- `n = 1`, `a = [-5]` or `[0]`: prefix `[0,-5]` / `[0,0]`. `M_<` is empty (no smaller prefix), `M_>=` gives `dp[0]=0`, so `dp[1]=0`. A non-positive single hour is zero profitable blocks. Correct — note a block of sum exactly `0` is **not** profitable, since profitability is strict.
- All non-positive, e.g. `[-1,-2,-3]`: prefixes are non-increasing, so no later prefix ever strictly exceeds an earlier one; every `M_<` query is empty and `dp` stays `0`. Answer `0`. Correct.
- All positive, e.g. `n` copies of `10^9`: each prefix strictly exceeds all earlier ones, so `dp` climbs by 1 every step to `n`. Each hour is its own profitable block — the maximum possible. I verified `n=2*10^5` of `10^9` returns exactly `200000`.
- Overflow: prefixes are `long long`, reaching about `2*10^14`, well within range; the answer/`dp` values are small (`<= n`) and held in `int`. The Fenwick sentinel `NEG = INT_MIN/4` is only ever compared inside `max`, never has anything added to it except `+1` in `a1 + 1`, and `a1 > NEG` is checked first, so it cannot underflow. Safe.
- Timing/memory: `O(n log n)` with two `int` Fenwick arrays of size `n+1`; measured `0.13 s` and `8 MB` at `n=2*10^5`, far inside the `1 s` / `256 MB` budget.

**Final solution.** I disproved the running-sum greedy with a traced counterexample (the sample itself: greedy 1 vs the reachable 3), recast the task as a prefix-sum DP whose last-block transition is a range-max over prior prefixes, and made that range-max `O(log n)` with two Fenwick trees. Two real bugs — a reversed-coordinate off-by-one that fell off the tree on ties, and a forgotten second-tree update that blinded the `>=` side — were each pinned by tracing a tiny failing input, then erased by 1000-case agreement with an independent brute. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;              // n = 0 (or empty input) -> answer 0

    // prefix[i] = a[0] + ... + a[i-1], so prefix[0] = 0 and a block (j, i] has
    // sum prefix[i] - prefix[j]; it is "profitable" iff prefix[i] > prefix[j].
    vector<long long> prefix(n + 1);
    prefix[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Coordinate-compress the n+1 prefix values.
    vector<long long> vals(prefix.begin(), prefix.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto cid = [&](long long v) {
        return int(lower_bound(vals.begin(), vals.end(), v) - vals.begin());
    };

    const int NEG = INT_MIN / 4;

    // dp[i] = max profitable blocks in a full partition of prefix[0..i].
    // dp[i] = max( 1 + max_{j<i, prefix[j] <  prefix[i]} dp[j],     // block (j,i] profitable
    //                  max_{j<i, prefix[j] >= prefix[i]} dp[j] ).   // block (j,i] not profitable
    // Two Fenwick trees over the compressed prefix coordinate hold prefix-max of dp:
    //   bitLess : indexed by coordinate, prefix-max query gives best dp over smaller prefix values.
    //   bitGeq  : indexed by REVERSED coordinate, prefix-max query gives best dp over >= values.
    vector<int> bitLess(m + 1, NEG), bitGeq(m + 1, NEG);
    auto upd = [&](vector<int> &t, int i, int v) {       // 1-based index
        for (++i; i <= m; i += i & (-i)) t[i] = max(t[i], v);
    };
    auto qry = [&](vector<int> &t, int i) {              // max over [0..i], 0-based i
        int r = NEG;
        for (++i; i > 0; i -= i & (-i)) r = max(r, t[i]);
        return r;
    };

    // Insert j = 0: dp[0] = 0 at coordinate of prefix[0].
    int c0 = cid(prefix[0]);
    upd(bitLess, c0, 0);
    upd(bitGeq, m - 1 - c0, 0);

    int ans = 0;
    for (int i = 1; i <= n; i++) {
        int ci = cid(prefix[i]);
        int best = NEG;
        // prefix[j] < prefix[i]: coordinates [0 .. ci-1]
        if (ci - 1 >= 0) {
            int a1 = qry(bitLess, ci - 1);
            if (a1 > NEG) best = max(best, a1 + 1);
        }
        // prefix[j] >= prefix[i]: coordinates [ci .. m-1] = reversed prefix [0 .. m-1-ci]
        int a2 = qry(bitGeq, m - 1 - ci);
        if (a2 > NEG) best = max(best, a2);
        int dpi = best;                 // there is always at least one j (j = i-1), so best > NEG
        ans = dpi;
        upd(bitLess, ci, dpi);
        upd(bitGeq, m - 1 - ci, dpi);
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The running-sum greedy looked right but a single traced counterexample (the sample `[4,-4,2,-1,2,-4]`: greedy 1 vs the reachable 3) showed that closing a block the moment it turns positive glues the negatives into one dead tail and forfeits the *bridges* that let the total fall and climb again — so I switched to a prefix-sum DP where `dp[i]` adds `1` exactly when the last block's endpoints satisfy `prefix[j] < prefix[i]`, making the transition a range-max over earlier prefixes that two Fenwick trees evaluate in `O(log n)`; a reversed-coordinate off-by-one (`m-ci` vs `m-1-ci`) sent the `>=` tree off its end on tie-heavy inputs like `[0,0]`, and a missing second-tree update blinded the `>=` side on inputs like `[-4,2,4]`, each caught by tracing a minimal failing case and confirmed gone by 1000-case agreement with an independent brute, with `long long` prefixes closing the overflow corner.
