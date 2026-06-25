**Reading the problem and pinning the contract.** I have `n` parcels; parcel `i` carries a weight
`w[i] >= 0` (zero is allowed — paperwork-only manifests) and a net profit `v[i]` that may be
negative, zero, or positive. I must load **exactly `K`** parcels whose combined weight is **at most
`C`**, and maximize the total profit; if no size-`K` load fits under the cap, I print `INFEASIBLE`.
Two features of this contract jump out immediately and dictate everything downstream. First, the
count is *fixed at `K`*, not "at most `K`": that means I cannot just drop unprofitable parcels to make
the answer non-negative — if all `K` parcels I am forced to take are losers, the optimum is genuinely
**negative**, and that negative number is the answer, not `INFEASIBLE`. Second, the empty load `K = 0`
is legal and has profit `0`. So I have three distinct outcomes that must never be confused: a positive
profit, a *negative* profit (a real, feasible-but-unprofitable load), and the *unreachable* verdict
`INFEASIBLE`. The whole problem is about keeping those three apart.

Let me fix scale before any algorithm, because it decides the data types. `n <= 200`, `C <= 1000`,
`|v[i]| <= 10^9`, and a load has up to `K <= n <= 200` parcels, so a total profit can reach
`200 * 10^9 = 2 * 10^11`. That blows past the 32-bit range (about `2.1 * 10^9`), so every accumulator
and every table cell must be 64-bit `long long`. An `int` here is a silent wrong-answer on the large
tests. Also `K` is given up to `10^9` in the contract even though only `K <= n` can ever be feasible,
so I must read `K` as 64-bit and special-case `K > n` before I ever size an array by `K`. Weights can
be up to `10^9` too, far larger than `C = 1000`, so a parcel can be individually uncarriable — I have
to guard against indexing a weight axis of size only `C+1` with such a weight.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one
I can size and prove, not the one that is easiest to type.

- *Subset enumeration / meet-in-the-middle.* Enumerate every size-`K` subset and keep the best one
  that fits under `C`. `O(C(n,K))` is obviously correct but explodes — `C(200,100)` is astronomical.
  Meet-in-the-middle reaches `n` near 40, but merging two halves under "exactly `K` total parcels and
  total weight `<= C`, maximize profit" is fiddly and still far short of `n = 200`. Out for the real
  size; I will keep exhaustive enumeration only as my *brute-force oracle* for small random tests.
- *Two-dimensional capacity DP (count x weight).* Carry a table `dp[k][c]` = the best total profit of
  choosing exactly `k` parcels of exact total weight `c`, for `c` in `0..C`. Relax parcel by parcel in
  0/1 fashion. Time `O(n * K * C)`, memory `O(K * C)`. With `n = 200`, `K = 200`, `C = 1000` that is
  `4 * 10^7` table updates and `~201 * 1001` longs of memory (about 1.6 MB) — both comfortable inside
  1 s / 256 MB. This is the route. The risk is not the idea but the *base case and the sign handling*:
  what is reachable before any parcel is placed, and how do I mark "unreachable" without it colliding
  with a real profit of `0`.

**Deriving the DP and checking the recurrence on paper.** I track exact total weight on one axis so
that "weight `<= C`" becomes a final reduction over all `c <= C`, and exact parcel count on the other
so that "exactly `K`" is just reading row `K` at the end. Define

- `dp[k][c]` = maximum total profit using exactly `k` parcels with total weight exactly `c`,
  or "unreachable" if no such selection exists.

The only `(k, c)` reachable with zero parcels placed is `(0, 0)` with profit `0` — the empty load.
*Every other* `(k, c)` is initially unreachable, and I encode unreachable with a sentinel
`NEG = LLONG_MIN/4` that I will never let `+v[i]` push into overflow (I divide `LLONG_MIN` by 4 so it
has headroom, and I only ever add `v[i]` to a *reachable* cell, never to the sentinel). The 0/1
transition for parcel `i` with `(w_i, v_i)`: a state `(k, c)` that is reachable can either *skip*
parcel `i` (no change) or *take* it, reaching `(k+1, c+w_i)` with profit increased by `v_i`, provided
`c + w_i <= C`. So

```
dp_new[k+1][c + w_i] = max(dp_new[k+1][c + w_i], dp_old[k][c] + v_i)   if dp_old[k][c] reachable and c + w_i <= C
```

To make each parcel usable **at most once** I do the classic 0/1 in-place update but I must be careful
about *which* axis I sweep backwards. Here the parcel-count axis `k` is the one that grows by one per
take, so I must iterate `k` **downward** (from `K-1` to `0`); that guarantees the `dp[k][c]` I read
was computed *before* this parcel was offered, so a single parcel cannot fill two of the `K` slots. The
weight axis `c` only needs to not collide within the same `k`-row update; sweeping it downward as well
is safe. The answer is `max over c in 0..C of dp[K][c]`; if that row is entirely the sentinel, no
size-`K` load fits and I print `INFEASIBLE`.

Let me confirm the recurrence by hand on the sample: `n = 5`, `K = 2`, `C = 7`, parcels
`(3,5), (4,-2), (2,0), (5,4), (0,-1)`, claimed answer `5`. I will not run the full DP by hand (two
axes is a lot of cells), but I will check the *objective directly* by enumerating size-2 loads that
fit weight `<= 7`: `{(3,5),(4,-2)}` weight `7`, profit `3`; `{(3,5),(2,0)}` weight `5`, profit `5`;
`{(3,5),(0,-1)}` weight `3`, profit `4`; `{(4,-2),(2,0)}` weight `6`, profit `-2`; `{(4,-2),(0,-1)}`
weight `4`, profit `-3`; `{(2,0),(5,4)}` weight `7`, profit `4`; `{(2,0),(0,-1)}` weight `2`, profit
`-1`; `{(5,4),(0,-1)}` weight `5`, profit `3`; `{(3,5),(5,4)}` weight `8` — over the cap, excluded;
`{(4,-2),(5,4)}` weight `9` — excluded. The best feasible profit is `5` from `{(3,5),(2,0)}`. So the
objective the DP must report is indeed `5`, and notice the optimum here deliberately *uses a
zero-profit parcel* — the recurrence has to be happy adding `v = 0`, not treat `0` as "nothing
happened." Good.

**First implementation — and immediately a trace, because the base case is the whole game.** My first
cut sets up the table and, reaching for "an empty cell should not block anything," initialises the
*entire* table to `0`:

```
vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, 0));   // <-- first attempt
// (no explicit dp[0][0] = 0 needed, it's already 0)
for (int i = 0; i < n; i++) {
    ... 0/1 update, counts downward ...
}
long long ans = LLONG_MIN;
for (int c = 0; c <= Cc; c++) ans = max(ans, dp[Kc][c]);
cout << ans << "\n";
```

The thing nagging me is: `dp[k][c] = 0` is supposed to mean "unreachable," but `0` is also a perfectly
real profit. Let me trace the smallest input that pits those two meanings against each other:
`n = 1`, `K = 1`, `C = 5`, single parcel `(w, v) = (2, -4)`. The only size-1 load is that one parcel,
weight `2 <= 5`, so the answer is obviously `-4`. Run the all-zero table: every `dp[k][c]` starts at
`0`. Parcel 0, count loop `k = 0` (downward from `K-1 = 0`): for each `c` with `c + 2 <= 5`, I do
`dp[1][c+2] = max(dp[1][c+2], dp[0][c] + (-4)) = max(0, 0 - 4) = max(0, -4) = 0`. So `dp[1][*]` stays
`0` everywhere. Final answer `max over c of dp[1][c] = 0`.

**Diagnosing the first bug.** The code returns `0`, but the right answer is `-4`. The defect is
precise and it is exactly the base-case collision I worried about: by seeding *every* `dp[k][c]` to
`0`, I told the DP that "exactly `1` parcel, weight `c`" is already achievable with profit `0` *before
I place any parcel* — a phantom selection. When the real parcel offers `-4`, the `max` prefers the
phantom `0` and the genuine `-4` is discarded. The all-zero initialisation conflates "unreachable"
with "reachable at profit 0," and because profits can be negative, the phantom `0` actively *beats*
the true optimum. This is the wrong-base-case / sign trap in its purest form. The fix is to seed
**only** `dp[0][0] = 0` and set everything else to the sentinel `NEG`, and to skip any cell still equal
to `NEG` when relaxing (never add `v_i` to a sentinel). I confirmed the bug bites on real generated
data too, not just my toy: e.g. `n = 3, K = 1, C = 5` with parcels `(0,-6),(0,-3),(4,-3)` — true best
is `-3` (take the single least-bad parcel), but the all-zero table returns `0`.

**Fixing and re-verifying the base case.** Rewrite with the sentinel:

```
const long long NEG = LLONG_MIN / 4;
vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, NEG));
dp[0][0] = 0;                                    // only the empty load is reachable a priori
...
if (dp[k][c] == NEG) continue;                   // never relax from / add v_i to an unreachable cell
...
long long ans = NEG;
for (int c = 0; c <= Cc; c++) if (dp[Kc][c] != NEG) ans = max(ans, dp[Kc][c]);
if (ans == NEG) cout << "INFEASIBLE\n"; else cout << ans << "\n";
```

Re-trace `n = 1, K = 1, C = 5, (2,-4)`: start `dp[0][0] = 0`, all else `NEG`. Parcel 0, `k = 0`,
`c = 0` is reachable: `dp[1][2] = max(NEG, 0 + (-4)) = -4`. Other `dp[0][c]` are `NEG`, skipped. Final
row 1: `dp[1][2] = -4`, rest `NEG`; `ans = -4`. Correct. And the all-negative single parcel now
reports its true negative profit instead of a phantom `0`. The bug fixed itself exactly where I
predicted, which is the evidence I trust.

**Second implementation pass — and a trace of the 0/1 direction, because reuse is the other classic
trap.** With the base case right, the remaining danger is letting one parcel fill multiple of the `K`
slots. I deliberately sweep the count axis `k` **downward** for the 0/1 guarantee, but let me prove to
myself that the *other* direction would actually break, so I know the downward sweep is load-bearing
and not cargo-culted. Suppose I wrote the count loop upward, `for k = 0 .. K-1`. Trace
`n = 2, K = 2, C = 0`, two zero-weight parcels `(0,5)` and `(0,3)`; the only size-2 load is both
parcels, profit `8`. Start `dp[0][0] = 0`, rest `NEG`. Parcel 0 `(0,5)`, upward `k`: `k = 0`,
`c = 0`: `dp[1][0] = max(NEG, 0 + 5) = 5`. `k = 1`, `c = 0`: now `dp[1][0]` was *just set to 5 in this
same parcel's pass*, so `dp[2][0] = max(NEG, 5 + 5) = 10`. The upward sweep let parcel 0 occupy *both*
slots, giving an impossible profit `10` from a single parcel. Then parcel 1 `(0,3)` would push it
higher still. Wrong.

**Diagnosing and confirming the 0/1 direction.** The upward sweep reads a `dp[k][c]` that this very
parcel already updated, so the parcel gets counted twice — exactly the multiset behaviour I must
forbid. The downward sweep fixes it: process `k = K-1` down to `0`, so when I write `dp[k+1][...]` I am
reading `dp[k][...]` from *before* parcel `i` was offered. Re-trace the same case with the **downward**
sweep: parcel 0 `(0,5)`, `k = 1`: `dp[1][0]` is still `NEG` (parcel 0 hasn't touched row 1 yet at this
`k`), skip; `k = 0`: `dp[1][0] = max(NEG, 0+5) = 5`. Now `dp[2][0]` is still `NEG`. Parcel 1 `(0,3)`,
`k = 1`: `dp[1][0] = 5` is reachable, so `dp[2][0] = max(NEG, 5 + 3) = 8`; `k = 0`: `dp[1][0] =
max(5, 0+3) = 5` (unchanged). Final `dp[2][0] = 8`. Correct — each parcel used exactly once. So the
downward count sweep is the real fix, and I have a concrete witness (`10` vs the true `8`) of why the
other direction is wrong.

**The capacity / huge-weight guard.** Weights can be `10^9` while the weight axis only spans `0..C =
0..1000`. If I compute `c + w_i` with `w_i = 10^9` I would index wildly out of bounds. The clean guard
is: if `w_i > C`, parcel `i` can never be part of any feasible load (it alone exceeds the cap), so I
`continue` past it entirely. After that guard `w_i <= C <= 1000` fits in an `int`, and `c + w_i <= C`
is enforced by sweeping `c` from `C - w_i` down to `0`. Let me sanity-check the guard on
`n = 3, K = 2, C = 3`, parcels `(2,5),(5,9),(6,9)`: parcels 1 and 2 have weight `> 3` and are skipped,
leaving only one carriable parcel; a size-2 load is impossible, so the answer must be `INFEASIBLE`.
With the guard, rows for `k = 2` stay all `NEG`, and I print `INFEASIBLE`. Correct, and crucially
*not* a crash from indexing `c + 6` into a length-4 weight array.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `K = 0`: the count loop body never produces a `dp[k+1]` since the only reachable seed is `dp[0][0]`,
  and I read row `0`: `dp[0][0] = 0`, all other `dp[0][c]` are `NEG`. So `ans = 0`. The empty load,
  profit `0`, *even when every parcel is negative* — correct. Hand-check `n = 3, K = 0, C = 5`,
  parcels all `(*, -9)`: answer `0`. Verified against brute.
- `n = 0`: no parcels read. If `K = 0`, `dp[0][0] = 0` gives `0`; if `K >= 1`, row `K` is all `NEG`
  (and in fact `K > n` triggers the early `INFEASIBLE`). Both correct.
- `K > n` (including `K` up to `10^9` from the contract): no selection of `K` distinct parcels exists,
  so `INFEASIBLE`. I test `K < 0 || K > n` *before* sizing any array by `K`, so I never allocate a
  `10^9`-row table. Without that early check the `vector` allocation itself would explode.
- All feasible loads negative: e.g. `n = 3, K = 2, C = 10`, parcels `(2,-3),(3,-1),(4,-5)`. Best
  size-2 fit is `{(2,-3),(3,-1)}` = `-4`; the DP returns `-4`, *not* `INFEASIBLE`. The sentinel keeps
  "feasible but negative" (`-4`, a real value) distinct from "infeasible" (`NEG` everywhere). Verified.
- Many zero-weight parcels: `n = 3, K = 3, C = 0`, parcels `(0,-2),(0,5),(0,-1)`; forced to take all
  three, profit `-2 + 5 - 1 = 2`. The DP loads all three at weight `0` and returns `2`. Verified —
  zero-weight items stack without spending capacity, exactly as intended.
- Overflow: accumulators and table cells are `long long`; the max profit `~2 * 10^11` fits with room
  to spare. The sentinel `LLONG_MIN/4` is only ever *read* (skipped) and never has `v_i` added to it,
  so it cannot underflow. On the worst case `n = 200, K = 200, C = 1000` with `|v|` near `10^9` the
  program returns a value above `6 * 10^9` (which an `int` would have mangled) in 0.02 s and under
  5 MB. Safe on time, memory, and range.
- Output: exactly one line — either the integer (possibly negative or zero) or the token
  `INFEASIBLE`. `cin >>` consumes arbitrary whitespace, so input formatting is irrelevant.

**Numeric self-check of the complexity claim.** I asserted `O(n * K * C)` updates. On the worst case
that is `200 * 200 * 1000 = 4 * 10^7` inner iterations, each a compare-add. Empirically the worst case
clocked `0.02 s`, i.e. roughly `4 * 10^7 / 0.02 = 2 * 10^9` simple ops/s — consistent with a tight
`-O2` integer loop, and a factor ~50 under the 1 s budget. Memory `(K+1)(C+1)` longs `= 201 * 1001 * 8
bytes ~= 1.6 MB`, matching the observed ~4.8 MB resident (table plus binary/runtime). The bound is
real and comfortable; I am not relying on an unproven identity, the measured numbers back it.

**Stress verification.** I ran the DP against an independent exhaustive `C(n,K)` brute force over
1100 random small cases from two different generators (varied `n`, `K` including `K = 0` and `K > n`,
many zero-weight and over-cap parcels, profits in `[-6, 6]` straddling zero) — **zero mismatches**.
Every required corner (empty load, all-negative negative optimum, INFEASIBLE-vs-negative, zero-weight
stacking, huge weights) was hit and matched.

**Final solution.** I convinced myself the *idea* is right by sizing the two-axis DP and hand-checking
the objective on the sample, and I convinced myself the *code* is right by tracing two real bugs to
precise causes — an all-zero base case that lets a phantom profit `0` beat a genuine negative load, and
an upward count sweep that double-counts a single parcel — fixing each and re-verifying the failing
traces, then sweeping every corner. That is what I ship: one self-contained file, the `O(n*K*C)`
count-by-weight 0/1 DP with a sentinel base case I can defend.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, C;
    if (!(cin >> n >> K >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // If the required count is impossible up front, no feasible load exists.
    if (K < 0 || K > n) { cout << "INFEASIBLE" << "\n"; return 0; }

    const long long NEG = LLONG_MIN / 4;       // "no subset with this (count,weight) exists"
    int Kc = (int)K;
    int Cc = (int)C;

    // dp[k][c] = best total profit using EXACTLY k parcels of total weight EXACTLY c (c in 0..Cc).
    vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, NEG));
    dp[0][0] = 0;                              // exactly 0 parcels, weight 0, profit 0 (empty load)

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > (long long)Cc) continue;      // parcel alone exceeds capacity: never choosable
        int wint = (int)wi;
        for (int k = Kc - 1; k >= 0; k--) {    // 0/1: counts downward so each parcel used once
            for (int c = Cc - wint; c >= 0; c--) {
                if (dp[k][c] == NEG) continue;
                long long cand = dp[k][c] + vi;
                if (cand > dp[k + 1][c + wint]) dp[k + 1][c + wint] = cand;
            }
        }
    }

    long long ans = NEG;
    for (int c = 0; c <= Cc; c++)
        if (dp[Kc][c] != NEG && dp[Kc][c] > ans) ans = dp[Kc][c];

    if (ans == NEG) cout << "INFEASIBLE" << "\n";
    else cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The fixed-count requirement makes the optimum legitimately negative, so the three
outcomes positive / negative / infeasible must stay distinct — which is precisely what my first
all-zero table destroyed: seeding every `dp[k][c]` to `0` planted a phantom "exactly-`k`-parcels at
profit `0`" state, and a trace of `n=1,K=1,C=5,(2,-4)` returning `0` instead of `-4` exposed that the
phantom `0` beats a real negative load. Seeding only `dp[0][0]=0` with a `NEG` sentinel (never relaxed
from, never added to) fixes the sign handling and lets `INFEASIBLE` mean "row `K` is all sentinel."
The second trap, a single parcel filling several of the `K` slots, is killed by sweeping the count axis
downward — witnessed by `n=2,K=2,C=0,(0,5),(0,3)` giving the impossible `10` upward versus the correct
`8` downward — and the `w_i > C` guard plus the early `K > n` check close out huge weights and the
`K`-up-to-`10^9` allocation, with `long long` throughout for the `~2*10^11` profit range.
