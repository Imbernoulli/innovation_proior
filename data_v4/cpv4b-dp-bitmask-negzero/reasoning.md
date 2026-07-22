I have `n` stages and `n` artists, and an `n x n`
reward matrix `p` whose entry `p[i][j]` is the net reward — money minus fees and costs — for booking
artist `i` on stage `j`. Crucially `p[i][j]` may be negative or zero. I must pick a *partial*
matching: a subset of artists, each placed on a distinct stage, every artist at most once and every
stage hosting at most one artist, but with **no obligation** to fill a stage or to book an artist. I
maximize the total reward, and because booking nobody is legal, the answer never drops below `0`.
Input is `n`, then `n` rows of `n` integers; output is one integer.

Before any algorithm I fix the scale, because it decides data types. `n <= 18`, and `|p[i][j]| <=
10^9`. A full booking sums up to `18` entries, so a reward can reach `18 * 10^9 = 1.8 * 10^10`. That
overflows 32-bit (`~2.1 * 10^9`) by an order of magnitude, so every reward and every accumulator must
be 64-bit `long long`. That is non-negotiable: an `int` here is a silent wrong answer on the large
`n = 18` tests. The state space is a subset of artists, `2^18 = 262144` masks, which is small.

**Laying out the candidate approaches.** Two framings are on the table and I want the one whose
*corners* I can defend, not merely the one with the shortest body.

- *Full-assignment bitmask DP.* The classic. Book every stage with exactly one artist (so the booking
  is a permutation of artists to stages), with `dp[mask]` = best reward when the artists in `mask`
  have filled stages `0..popcount(mask)-1`, transition `dp[mask | (1<<a)] = max(dp[mask] +
  p[a][popcount(mask)])`, and answer `dp[full]`. Clean. But it answers a *different* problem: it is
  forced to use every stage and every artist. On a loss-heavy matrix it must absorb negative rewards
  it should have skipped. So as stated it is wrong for the partial problem — I would have to bolt on
  "optional" artists and stages, which is exactly where I expect to slip.
- *Stage sweep, subset over artists.* Process stages `s = 0, 1, ..., n-1` in order. `dp[mask]` = the
  best total reward after deciding stages `0..s-1`, where `mask` is the set of artists already booked.
  At stage `s` I either leave it dark (add `0`, keep the same `mask`) or hand it one still-free artist
  `a` (add `p[a][s]`, set bit `a`). This makes "partial" first-class: skipping is a real transition,
  not an afterthought. I commit to this; the danger is not the idea but the base case and the final
  aggregation, which is precisely where the sign issue will bite.

**Deriving the recurrence.** Let me write it cleanly. I sweep stages with an outer loop and keep a
`dp` array indexed by artist-subset. Before stage `0`, nobody is booked, so the only reachable state
is `mask = 0` with reward `0`; every other mask is unreachable, value `-infinity`. At stage `s`, from
a reachable state `dp[mask]`:

- *Leave stage `s` empty:* `ndp[mask] = max(ndp[mask], dp[mask])`. No artist consumed, `+0` reward.
- *Book free artist `a`:* `ndp[mask | (1<<a)] = max(..., dp[mask] + p[a][s])`, for every `a` with bit
  `a` clear.

After all `n` stages, the answer is `max over reachable mask of dp[mask]`, floored at `0` for the
empty booking (which is already `dp[0] = 0`, but I keep the explicit floor as armour). The "leave
empty" branch is what lets the optimum skip a whole stage; the floor and the `dp[0] = 0` base are
what let an all-loss matrix return `0`.

**A numeric self-check of the recurrence on the sample.** I do not trust a recurrence I have not run
by hand. Sample: `n = 3`, rows `7 1 3 / 2 8 4 / 5 6 9`, claimed answer `24`. I will not expand all
masks, but I will check the claimed optimum is reachable and that no obviously-better one exists. The
diagonal booking artist `i` -> stage `i` gives `p[0][0] + p[1][1] + p[2][2] = 7 + 8 + 9 = 24`. Is any
permutation better? The three columns' maxima are stage 0: `max(7,2,5)=7`, stage 1: `max(1,8,6)=8`,
stage 2: `max(3,4,9)=9`, and those maxima are attained by *distinct* artists (0, 1, 2), so the upper
bound `7+8+9=24` is achievable and is therefore exactly optimal. Could a *partial* booking beat a
full one here? No — every entry used in the optimum is positive, and dropping any positive entry only
lowers the sum. So `24` is right, and my DP, which can reach the diagonal via book-0-at-0, book-1-at-1,
book-2-at-2 (never choosing "empty"), will find it. The recurrence is sound on this case.

**First implementation — and a deliberate trace, because the base case is where this dies.** Here is
my first cut. I am suspicious of myself, so I will write it the way that "feels" natural and then
attack it.

```
const long long NEG = LLONG_MIN / 4;
int full = 1 << n;
vector<long long> dp(full, NEG);
dp[0] = 0;
for (int s = 0; s < n; s++) {
    vector<long long> ndp(full, NEG);
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == NEG) continue;
        // book one free artist a on stage s
        for (int a = 0; a < n; a++) {
            if (mask & (1 << a)) continue;
            int nm = mask | (1 << a);
            long long cand = dp[mask] + p[a][s];
            if (cand > ndp[nm]) ndp[nm] = cand;
        }
    }
    dp.swap(ndp);
}
long long ans = 0;
for (int mask = 0; mask < full; mask++)
    if (dp[mask] != NEG && dp[mask] > ans) ans = dp[mask];
cout << ans << "\n";
```

I trace the smallest input that should expose a flaw. Take `n = 2`, `p = [[-1, 5], [-1, -1]]`. By
hand the best partial booking: leave stage 0 dark, book artist 0 on stage 1 for `+5`, leave artist 1
on the bench. Answer `5`. (Booking anything on stage 0 only loses money; the only positive entry is
`p[0][1] = 5`.)

Now the code. `NEG` sentinel, `dp = [0, NEG, NEG, NEG]` (masks `00, 01, 10, 11`). Stage `s = 0`:
from `mask = 00` (value `0`) the loop books artist 0 -> `ndp[01] = 0 + p[0][0] = -1`, and artist 1 ->
`ndp[10] = 0 + p[1][0] = -1`. No other reachable mask. So after stage 0, `dp = [NEG, -1, -1, NEG]`.

Notice already: `dp[00]` is now `NEG`. The state "stage 0 left empty, nobody booked" has *vanished*.

Stage `s = 1`: reachable masks are `01` (`-1`) and `10` (`-1`). From `01` I can book artist 1 ->
`ndp[11] = -1 + p[1][1] = -1 + (-1) = -2`. From `10` I can book artist 0 -> `ndp[11] = max(-2, -1 +
p[0][1]) = max(-2, -1 + 5) = 4`. There is no state where artist 0 is booked on stage 1 *without*
having already booked someone on stage 0. Final `dp = [NEG, NEG, NEG, 4]`, and `ans = max(0, 4) = 4`.

**The bug.** The code prints `4`, the truth is `5`. The defect is precise and it is a *base/skip*
defect: my transition never offers the option to **leave a stage empty**, so once stage 0 runs, the
empty-mask state `dp[00] = 0` is destroyed (it is not copied into `ndp`). Every booking is therefore
forced to start by putting *some* artist on stage 0 — I silently reintroduced the full-assignment
constraint I swore off. The only way to play artist 0 on stage 1 was to first eat a loss on stage 0
(`-1`), netting `4` instead of the clean `5`. The missing piece is the "skip this stage" carry:
`ndp[mask] = max(ndp[mask], dp[mask])`. Without it, partial bookings on later stages are unreachable
unless padded with earlier losses, and `dp[0]`'s reward-`0` floor cannot propagate forward.

**Fix and re-trace.** I add the skip branch at the top of the per-mask body:

```
if (dp[mask] > ndp[mask]) ndp[mask] = dp[mask];   // leave stage s empty
```

Re-trace `[[-1,5],[-1,-1]]`. `dp = [0, NEG, NEG, NEG]`. Stage 0: from `00` skip -> `ndp[00] = 0`;
book 0 -> `ndp[01] = -1`; book 1 -> `ndp[10] = -1`. `dp = [0, -1, -1, NEG]`. Stage 1: from `00`
(value `0`) skip -> `ndp[00] = 0`; book 0 -> `ndp[01] = 0 + 5 = 5`; book 1 -> `ndp[10] = 0 + (-1) =
-1`. From `01` (`-1`) skip -> `ndp[01] = max(5, -1) = 5`; book 1 -> `ndp[11] = -1 + (-1) = -2`. From
`10` (`-1`) skip -> `ndp[10] = max(-1, -1) = -1`; book 0 -> `ndp[11] = max(-2, -1 + 5) = 4`. Final
`dp = [0, 5, -1, 4]`, `ans = max(0, 5, -1, 4) = 5`. Correct. The case that broke now passes, and it
broke for exactly the reason the fix addresses — the evidence I trust.

**Second debug episode — a sign/aggregation trap on the all-loss corner.** The required corner is an
all-negative matrix, where the answer must be `0` (book nobody). I want to be sure my *aggregation*
honours that and that I did not accidentally seed a wrong base. Consider an intermediate version where,
out of "tidiness", I initialise the answer from the full mask instead of flooring at `0`:

```
long long ans = dp[full - 1];   // WRONG: assumes a full booking is wanted
```

Trace it on `n = 2`, `p = [[-3, -1], [-4, -2]]` (every entry a loss; truth `0`). With the fixed
transition, after both stages `dp[11]` (full booking) is the best *permutation*: either artist0@stage0
+ artist1@stage1 = `-3 + -2 = -5`, or artist1@stage0 + artist0@stage1 = `-4 + -1 = -5`; so `dp[11] =
-5`. The tidy aggregation would print `-5` — a roster that loses money when the organisers could
simply book nobody and net `0`. That is the sign/empty-corner bug in the open. The correct
aggregation maximises over *all* reachable masks and floors at `0`: `dp = [0, ...]` always holds
`dp[0] = 0` (skip every stage), so `ans = max(0, dp[01], dp[10], dp[11]) = max(0, -3, -4, -5) = 0`.
I keep `ans` initialised to `0` and only raise it, never lower it, which is the floor that encodes
the empty booking. With that, the all-loss matrix returns `0`. Bug avoided, and I now have a concrete
witness (`-5` vs `0`) of why the floor is load-bearing rather than decorative.

**Edge cases, deliberately, because this family of code dies in the corners.**

- `n = 0`: `full = 1 << 0 = 1`, `dp = [0]`, the stage loop never runs, `ans = max(0, dp[0]) = 0`. The
  empty festival — correct. (`if (!(cin >> n)) return 0;` also covers truly empty input.)
- `n = 1`, `p = [[-5]]`: `dp = [0, NEG]`. Stage 0: from `00` skip -> `ndp[00] = 0`; book 0 ->
  `ndp[01] = -5`. `dp = [0, -5]`, `ans = max(0, 0, -5) = 0`. Bench the artist rather than lose `5` —
  correct.
- `n = 1`, `p = [[8]]`: stage 0 gives `dp = [0, 8]`, `ans = 8`. Correct.
- All-negative (general): every booked-state value is a sum of negatives, strictly below the `dp[0] =
  0` floor, so `ans` stays `0`. Verified above on the `2 x 2` loss matrix.
- Zeros: `p = [[0,0],[0,5]]`. Booking artist 0 on a zero-stage costs nothing, but the only gain is
  `p[1][1] = 5`. The DP can leave stage 0 dark and book artist 1 on stage 1 for `5`; mixing in a zero
  booking neither helps nor hurts. Expected `5`. (My run confirms `5`.) Zeros behave as the neutral
  element they should — they neither force nor forbid a booking.
- Overflow: `dp` values are `long long`; the largest reachable magnitude is `~1.8 * 10^10`, far inside
  64-bit range. The sentinel `LLONG_MIN / 4` is only ever *read* — guarded by `if (dp[mask] == NEG)
  continue;` so `p[a][s]` is never added to it, and it can never underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the row/column
  layout of the input is irrelevant to parsing.

**Complexity sanity check.** Outer loop over `n` stages, inner over `2^n` masks, innermost over `n`
artists: `O(n^2 * 2^n)`. For `n = 18` that is `18 * 18 * 262144 ~ 8.5 * 10^7` constant-time updates,
plus allocating an `ndp` of `2^18` longs per stage (18 allocations). Comfortable inside 2 seconds and
256 MB. The `dp.swap(ndp)` keeps memory at two arrays.

**Cross-checking against a brute force.** Idea-level confidence is not enough, so I pit the DP against
an independent brute force that recursively enumerates *every* partial matching — for each stage it
tries "empty" or "any free artist" and tracks the best running total, flooring at `0` — over hundreds
of random small matrices, biased toward all-negative, zero-heavy, and mixed-sign distributions and
including `n = 0`. The two agree on every case. The agreement on the all-negative and zero-heavy
families is the part I care about most, since that is where the base case and the sign of the
aggregation decide the answer.

**Final solution.** I disproved the full-assignment framing for the partial problem, derived the
stage-sweep subset DP, and hardened it through two real bugs: the missing skip-branch that secretly
re-imposed full assignment (caught by `[[-1,5],[-1,-1]]` returning `4` instead of `5`), and the
tempting `dp[full-1]` aggregation that returns a money-losing roster instead of the empty `0` (caught
by an all-loss matrix yielding `-5`). The shipped file is the stage-sweep DP with the skip branch and
the `0`-floored max over all masks:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> empty roster -> 0
    vector<vector<long long>> p(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> p[i][j];

    // We sweep stages s = 0..n-1. dp[mask] = best total reward achievable after deciding
    // stages 0..s-1, where `mask` is the set of artists already booked. Each stage may be
    // left empty (booking nobody, +0) or given one still-free artist (+p[artist][stage]).
    // Partial rosters are allowed, so the empty roster (dp all skipped) keeps the value 0.
    const long long NEG = LLONG_MIN / 4;
    int full = 1 << n;
    vector<long long> dp(full, NEG);
    dp[0] = 0;                            // before any stage: nobody booked, reward 0

    for (int s = 0; s < n; s++) {
        vector<long long> ndp(full, NEG);
        for (int mask = 0; mask < full; mask++) {
            if (dp[mask] == NEG) continue;
            // Option 1: leave stage s empty.
            if (dp[mask] > ndp[mask]) ndp[mask] = dp[mask];
            // Option 2: book one free artist a on stage s.
            for (int a = 0; a < n; a++) {
                if (mask & (1 << a)) continue;
                int nm = mask | (1 << a);
                long long cand = dp[mask] + p[a][s];
                if (cand > ndp[nm]) ndp[nm] = cand;
            }
        }
        dp.swap(ndp);
    }

    long long ans = 0;                    // empty roster is always allowed
    for (int mask = 0; mask < full; mask++)
        if (dp[mask] != NEG && dp[mask] > ans) ans = dp[mask];

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The full-assignment bitmask DP looked clean but answers the wrong question for a
partial, signed problem because it cannot leave stages dark; I switched to a stage sweep with a subset
state over artists. My first transition omitted the "leave stage empty" carry, which destroyed the
reward-`0` empty state after stage 0 and secretly re-imposed full assignment — a trace of
`[[-1,5],[-1,-1]]` returning `4` instead of `5` pinpointed it, and adding `ndp[mask] = max(ndp[mask],
dp[mask])` fixed it. Then the tempting `ans = dp[full-1]` aggregation would report a money-losing
full roster (`-5` on an all-loss matrix) instead of the legal empty `0`; maximising over all masks
with `ans` floored at `0` fixes the sign/empty corner. With 64-bit accumulators and the guarded `NEG`
sentinel, the empty (`n = 0`), single-negative, all-negative, and zero-heavy cases all return the
correct values, matching an independent brute force over hundreds of random tests.
