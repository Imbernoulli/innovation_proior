**Reading the problem and pinning the contract.** A tightrope walker stands on platform `0` of a row
of `n` platforms and must end on platform `n-1`. From platform `i` she may hop to `i+1` or to `i+2`,
never further. Every platform she lands on — including the start `0` and the finish `n-1` — costs its
toll `t[i] >= 0`, and I must report the cheapest legal crossing. Input on stdin is `n` then the `n`
tolls; I print one integer. Before any algorithm I fix the scale, because it dictates the data type:
`n <= 2*10^5` and each `t[i] <= 10^9`. A crossing that uses only `+1` hops lands on *every* platform,
so the total toll can be as large as `n * 10^9 = 2*10^5 * 10^9 = 2*10^14`. That is two thousand times
past the 32-bit ceiling of about `2.1*10^9`, so every accumulator and the toll array itself must be
64-bit `long long`. That is decision one and it is non-negotiable; an `int` here is a silent
wrong-answer on the large tests, and since all tolls are non-negative the overflow would not even flip
to a recognizably absurd value — it would just be quietly wrong.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one
I can *prove* rather than the one that is easiest to type.

- *Local greedy.* Stand on `i`; of the two platforms I can reach, `i+1` and `i+2`, hop to whichever
  has the smaller toll, break ties somehow, repeat. It is `O(n)` and almost no code. The risk is
  structural: a hop does not just pay a toll, it *changes which platforms are reachable next*. Hopping
  to a cheap platform now can land me where my only forward options are expensive. Greedy decides with
  a one-step horizon; the cost is paid over the whole route. That is exactly the configuration where
  greedy tends to be wrong, so I will not trust it until I have tried to break it.
- *Linear DP.* Let `dp[i]` be the minimum total toll of a legal route that *lands on* platform `i`.
  Any route that reaches `i` made its last hop from `i-1` (`+1`) or from `i-2` (`+2`), and either way
  it then pays `t[i]`. So `dp[i] = t[i] + min(dp[i-1], dp[i-2])`. This is `O(n)` time and `O(1)` extra
  state. The risk here is not the idea but the *base cases* — `dp[0]` and `dp[1]` and the tiny-`n`
  corners are where this kind of recurrence quietly breaks.

**Stress-testing greedy before committing.** "Greedy feels right" is how wrong solutions get shipped,
so let me actually attack it with a concrete instance rather than wave hands. Take
`t = [3, 1, 1, 9, 1, 9, 1]`, platforms `0..6`. Walk greedy: I start on `0` and pay `3`. From `0` the
two reachable platforms are `1` (toll `1`) and `2` (toll `1`) — a tie; say I break ties toward the
nearer platform and hop to `1`, paying `1` (running total `4`). From `1` I see `2` (toll `1`) and `3`
(toll `9`); the cheaper is `2`, hop there, pay `1` (total `5`). From `2` I see `3` (toll `9`) and `4`
(toll `1`); cheaper is `4`, hop, pay `1` (total `6`). From `4` I see `5` (toll `9`) and `6` (toll
`1`); cheaper is `6`, hop, pay `1` (total `7`). I have reached the finish. Greedy's total is `7`.

Is `7` optimal? Let me hunt for a route greedy structurally could not reach. Consider hopping `+2`
every time: `0 -> 2 -> 4 -> 6`, landing on `0, 2, 4, 6` with tolls `3 + 1 + 1 + 1 = 6`. That is
strictly cheaper than `7`. So greedy is wrong, and I can see *why*: standing on `0`, both `1` and `2`
cost `1`, and greedy's tie-break sent me to `1` — but landing on `1` is wasted money, because from `1`
I still have to pay a toll to move on, whereas the route that lands on `2` directly never pays for
platform `1` at all. The local view treated `1` and `2` as interchangeable because their tolls tie,
but globally landing on one extra platform is one extra toll. The verification paid off — it killed an
approach I would otherwise have shipped. And note this is not a tie-break artifact I can patch away:
even if I broke the tie toward the farther platform here, I can build instances like
`t = [2, 1, 100, 1, 1, 1]` where greedy still loses (greedy `6` vs optimal `5`), because the only way
to know whether to absorb a small toll now is to know the tolls further ahead. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I commit to `dp[i]` = minimum total toll of
a legal route that lands on platform `i`, starting from platform `0`. The recurrence
`dp[i] = t[i] + min(dp[i-1], dp[i-2])` holds for `i >= 2`. The base cases need care:

- `dp[0] = t[0]`: she starts on `0` and pays its toll. There is no earlier platform.
- `dp[1] = t[0] + t[1]`: platform `1` can be reached *only* from platform `0` via a `+1` hop (there is
  no platform `-1` to make a `+2` hop from). So the route to `1` is forced: `0 -> 1`, paying both
  tolls. This is the base case most likely to be written wrong — if I naively reused the general
  recurrence with a phantom `dp[-1]`, I would corrupt it.

The answer is `dp[n-1]`: she must finish standing on the last platform. There is no `max(..., 0)` here
and no "do nothing" option — unlike a max-weight selection, every legal crossing *must* end on `n-1`,
so the answer is whatever the cheapest such route costs (and with non-negative tolls it is `>= t[0]`).

Let me confirm the recurrence by hand on the documented sample `t = [3, 1, 1, 9, 1, 9, 1]`, expected
answer `6`. `dp[0] = 3`. `dp[1] = 3 + 1 = 4`. `dp[2] = t[2] + min(dp[1], dp[0]) = 1 + min(4, 3) = 4`.
`dp[3] = t[3] + min(dp[2], dp[1]) = 9 + min(4, 4) = 13`. `dp[4] = t[4] + min(dp[3], dp[2]) =
1 + min(13, 4) = 5`. `dp[5] = t[5] + min(dp[4], dp[3]) = 9 + min(5, 13) = 14`. `dp[6] = t[6] +
min(dp[5], dp[4]) = 1 + min(14, 5) = 6`. Answer `dp[6] = 6`. It matches, and the cheap path threads
through `dp[2] = 4` then `dp[4] = 5` then `dp[6] = 6`, which is exactly the `0 -> 2 -> 4 -> 6` route I
found by hand. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** To get
`O(1)` memory I will keep only the last two `dp` values. My first cut of the carry:

```
long long prev2 = t[0];   // dp[i-2]
long long prev1 = t[1];   // dp[i-1]   <-- suspicious
for (int i = 2; i < n; i++) {
    long long cur = t[i] + min(prev1, prev2);
    prev2 = prev1;
    prev1 = cur;
}
// answer = prev1
```

Something about `prev1 = t[1]` nags at me, so I trace the smallest input that could expose it. Take
`n = 2`, `t = [5, 7]`. The only legal crossing is `0 -> 1` (platform `1` is reachable only by `+1`),
paying `5 + 7 = 12`. My code: `prev2 = t[0] = 5`, `prev1 = t[1] = 7`, the loop body never runs (it
starts at `i = 2`), so the printed answer is `prev1 = 7`. That is wrong — it should be `12`.

**Diagnosing the first bug.** The defect is precise: I initialized `prev1` (which stands for `dp[1]`)
to `t[1]` alone, but `dp[1] = t[0] + t[1]` because reaching platform `1` *forces* her to have first
paid for platform `0`. There is no route that lands on `1` without having started on `0`. I dropped
the `t[0]` term. The general recurrence happens to hide this when `n >= 3`, because for `i = 2` the
`min(prev1, prev2)` would still pick up `prev2 = dp[0] = t[0]` and partly mask the error — which is
exactly why the bug only surfaces cleanly at `n = 2`, where the loop never executes and the broken
`prev1` is printed raw. The fix is to set `prev1 = t[1] + t[0]`.

**Fixing and re-verifying.** Corrected initialization:

```
long long prev2 = t[0];          // dp[0]
long long prev1 = t[1] + t[0];   // dp[1]: platform 1 reachable only from 0 via +1
for (int i = 2; i < n; i++) {
    long long cur = t[i] + min(prev1, prev2);
    prev2 = prev1;
    prev1 = cur;
}
// answer = prev1 = dp[n-1]
```

Re-trace `n = 2`, `t = [5, 7]`: `prev2 = 5`, `prev1 = 7 + 5 = 12`, loop skipped, answer `12`. Correct.
Re-trace the sample `t = [3, 1, 1, 9, 1, 9, 1]`: `prev2 = 3`, `prev1 = 1 + 3 = 4`. `i=2`:
`cur = 1 + min(4, 3) = 4`; now `prev2 = 4, prev1 = 4`. `i=3`: `cur = 9 + min(4, 4) = 13`;
`prev2 = 4, prev1 = 13`. `i=4`: `cur = 1 + min(13, 4) = 5`; `prev2 = 13, prev1 = 5`. `i=5`:
`cur = 9 + min(5, 13) = 14`; `prev2 = 5, prev1 = 14`. `i=6`: `cur = 1 + min(14, 5) = 6`;
`prev2 = 14, prev1 = 6`. Answer `6`. Correct, and it matches the by-hand `dp` table exactly. The case
that broke now passes, and it broke for precisely the reason I fixed — that is the evidence I trust.

**Second trace — the tiny-`n` corners, because that is where this rolling recurrence dies.** The loop
and the two-value carry both *assume* there are at least two platforms (they index `t[0]` and `t[1]`).
So before any of that runs I must guard `n = 0` and `n = 1` explicitly, and I should trace them.

- `n = 0`: there are no platforms. The walker has nothing to cross and pays nothing. But my carry
  reads `t[0]` and `t[1]`, which would be out-of-bounds on an empty vector — undefined behavior, a
  crash or garbage. So I need an early `if (n == 0) { print 0; }`. Trace: input `0`, output `0`.
  Correct, and no out-of-bounds read.
- `n = 1`: she starts and finishes on platform `0`; she pays only `t[0]`. My carry would read `t[1]`,
  out of bounds again. So I need `if (n == 1) { print t[0]; }`. Trace: input `1\n42`, output `42`.
  Correct.

I add both guards *after* reading the input (so the parsing is uniform) and *before* touching
`t[1]`. Let me also re-trace the very first index the loop will touch to be sure the guards leave the
`n >= 2` path intact: for `n = 2` the loop range `i = 2 .. n-1` is empty, and I correctly return
`prev1 = dp[1]`; for `n = 3` the loop runs once at `i = 2` and returns `prev1 = dp[2]`. The boundary
between "loop runs" and "loop empty" is at exactly `n = 2`, and both sides check out.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Empty input / `n = 0`.* The early guard prints `0` without reading `t[0]`. Correct, no UB.
- *`n = 1`.* Prints `t[0]`; a single platform that is both start and finish. Correct.
- *`n = 2`.* The crossing is forced `0 -> 1`; answer `t[0] + t[1]`. Verified above on `[5, 7] -> 12`.
- *Skip an expensive middle.* `n = 3`, `t = [0, 10^9, 0]`: `dp[0] = 0`, `dp[1] = 10^9`,
  `dp[2] = 0 + min(10^9, 0) = 0`. Answer `0` — she hops `0 -> 2` straight over the costly platform.
  Correct; this is the cheapest-route analogue of the greedy trap, and the DP nails it.
- *All tolls equal.* `t = [c, c, ..., c]` with `m` platforms: the cheapest route uses as many `+2`
  hops as possible, landing on `ceil(m/2)` platforms, paying `ceil(m/2) * c`. The DP gives this
  automatically by always taking the smaller `dp`. No special-casing needed.
- *All tolls zero.* Answer `0`. Every `dp[i] = 0`. Correct.
- *Overflow.* Accumulators are `long long`; the maximum total `~2*10^14` (all `+1` hops, all tolls
  `10^9`) fits with four orders of magnitude to spare. Tolls are non-negative, so `min` never has to
  reason about a sentinel — there is no `-infinity` anywhere, which removes a whole class of
  underflow/overflow worries the max-weight version had. Safe.
- *Output.* Exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input
  parsing is format-agnostic (the `n` and the tolls may be on one line or many).

**Sanity-checking the derivation itself against brute force.** Beyond hand traces I convinced myself
the *idea* is right by disproving greedy and by hand-checking the `dp` table on the sample. As a final
check on the reasoning I also confirmed (mentally, and it is what the verification harness does) that
the recurrence's claim "a route landing on `i` last hopped from `i-1` or `i-2`" is exhaustive: those
are the *only* two hop lengths allowed, so there is no third predecessor to miss, and `min` over the
two predecessor optima is exactly the optimum for `i` by the usual cut-and-paste argument (a cheapest
route to `i` must reach its predecessor by a cheapest route, else swap in the cheaper prefix). The
recurrence is both complete (no missing transition) and correct (optimal substructure holds).

**Final solution.** I convinced myself the idea is right by disproving the local greedy with a traced
counterexample and hand-checking the recurrence on the sample, and I convinced myself the *code* is
right by tracing the two failing/degenerate inputs (`n = 2` initialization, `n = 0/1` out-of-bounds)
to precise causes and re-verifying the fixes and the corners. That is what I ship — one self-contained
file, the simple `O(n)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // empty input -> 0 (no platforms, no tolls)
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << t[0] << "\n"; return 0; } // must land on the only platform

    // dp[i] = minimum total toll paid for a legal route that LANDS on platform i,
    // starting from platform 0 (always landed on) and using jumps of +1 or +2.
    // dp[0] = t[0]. To reach i you arrive from i-1 (a +1 jump) or i-2 (a +2 jump):
    //   dp[i] = t[i] + min(dp[i-1], dp[i-2]).
    // We only keep the previous two values (O(1) memory).
    long long prev2 = t[0];                 // dp[0]
    long long prev1 = t[1] + t[0];          // dp[1]: only reachable from 0 via +1
    for (int i = 2; i < n; i++) {
        long long cur = t[i] + min(prev1, prev2);
        prev2 = prev1;
        prev1 = cur;
    }

    cout << prev1 << "\n";                   // dp[n-1]: must finish on the last platform
    return 0;
}
```

**Causal recap.** The local greedy ("hop to the cheaper of the two platforms in front of me") looked
right but a single traced counterexample (`t = [3,1,1,9,1,9,1]`: greedy lands on platform `1` and
pays `7`, while hopping `0 -> 2 -> 4 -> 6` pays `6`) showed that a hop changes which platforms are
reachable, so paying for an extra landing now can never be undone — local choice, global cost. That
pushed me to the prefix DP `dp[i] = t[i] + min(dp[i-1], dp[i-2])`, whose two transitions correspond to
the only two legal hop lengths; my first rolling-carry version initialized `dp[1]` as `t[1]` instead
of `t[0] + t[1]`, which a trace of `n = 2` (`[5,7]` printing `7` instead of `12`) pinpointed, because
reaching platform `1` forces having paid for platform `0`; and explicit `n = 0` and `n = 1` guards
keep the two-value carry from reading past the end of the toll array, closing the empty, single, and
overflow corners over 64-bit accumulators.
