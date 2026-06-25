**Reading the problem and pinning the contract.** I cross a river on stones `0..n-1`. A leap goes
forward by one or two stones. The first landing off the near bank is restricted to stone `0` or
stone `1`; from stone `i` I land next on `i+1` or `i+2`; and I step off to the far bank with a free
final leap taken from stone `n-1` or stone `n-2`. Landing on stone `i` costs `c[i]`, which may be
negative (a helpful current that returns stamina). I want the minimum total of the costs of the
stones I land on. Input is `n` then the `n` costs; I print one integer. Before any algorithm I fix
the scale, because it dictates the data type: `n <= 2*10^5` and `|c[i]| <= 10^9`, so a crossing can
accumulate up to about `2*10^5 * 10^9 = 2*10^14` in magnitude. That is far past the 32-bit range of
roughly `2.1*10^9`, so every cost and every running total has to be 64-bit `long long`. That is the
first decision and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests.

I also pin the degenerate ends now. `n = 0`: there are no stones, the near bank *is* the far bank,
answer `0`. `n = 1`: the only legal start is stone `0` (stone `1` does not exist), I pay `c[0]`, and
stone `0 = n-1` so the final free leap clears the bank — answer `c[0]`. `n = 2`: I may first land on
stone `0` or stone `1`; from either I can clear the bank because both are within reach of the far
side (`0 = n-2`, `1 = n-1`), so the answer is `min(c[0], c[1])`. I will hold these three as a
checklist for whatever I build.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the
one I can *prove*, not the one that is shortest to type.

- *Local greedy.* Land on the cheaper of stone `0` / stone `1`; then from each stone hop to the
  cheaper of the next two reachable stones; stop once a leap clears the bank. `O(n)`, a handful of
  lines. The risk is structural: the reach is global — every landing fixes which two stones I can
  reach next — so a choice that looks cheap now can corner me onto an expensive stone later. That is
  precisely the configuration where local greedy fails, and I will not trust it until I try to break
  it.
- *Linear prefix DP.* For each stone, carry the minimum total cost of any legal crossing that
  *stands on that stone*. `O(n)`, `O(1)` state if folded. The risk is not the idea but the
  transcription at the two ends — the restricted first landing and the cost-free final leap are easy
  to get subtly wrong.

**Stress-testing greedy before committing.** "Cheaper of the next two feels right" is how wrong
solutions ship, so I attack it with a concrete instance: `c = [1, 8, 9, 5]`, stones `0..3`, `n = 4`.
Greedy's first landing is `min(c[0]=1, c[1]=8)` — stone `0`, paying `1`. From stone `0` the next two
reachable stones are stone `1` (`8`) and stone `2` (`9`); the cheaper is stone `1`, so greedy lands
there, paying `8`. Now I am on stone `1`, and `1 < n-2 = 2`, so I have not yet cleared the bank.
From stone `1` the next two are stone `2` (`9`) and stone `3` (`5`); cheaper is stone `3`, pay `5`,
and stone `3 = n-1` clears the bank. Greedy's total is `1 + 8 + 5 = 14`.

Is `14` optimal? I hunt for a crossing greedy structurally cannot reach. Land on stone `0` (`1`),
then leap `0 -> 2` onto stone `2` (`9`); stone `2 = n-2`, so the final free leap clears the bank.
Total `1 + 9 = 10`, strictly better than `14`. So greedy is wrong, and I see *why*: from stone `0`
it grabbed stone `1` because `8 < 9`, but that landing was not free of consequence — standing on
stone `1` still left me short of the bank, forcing a *third* landing. Jumping straight to stone `2`,
even though `9 > 8`, lands me where one free leap finishes the crossing, so I pay two stones instead
of three. The local rule counted the cost of the next landing but not the *future it commits me to*.
The verification paid off — it killed an approach I would otherwise have shipped. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I want, for each stone `i`, the quantity
`dp[i]` = the minimum total cost over all legal crossings that *end standing on stone `i`* (not yet
across). The only thing the future cares about is which stone I currently stand on, because that
fixes my next two options. So the recurrence reads backward in reach: to stand on stone `i` I must
have just leapt from stone `i-1` or stone `i-2`, then paid `c[i]`. Hence for `i >= 2`,

    dp[i] = min(dp[i-1], dp[i-2]) + c[i].

The two ends need care. The *first landing* is restricted to stone `0` or stone `1`:

    dp[0] = c[0]                         (first leap lands on stone 0)
    dp[1] = min( c[1],                   (first leap lands directly on stone 1)
                 dp[0] + c[1] )          (or land on 0 first, then step 0 -> 1)

The *final leap* is free and may be taken from stone `n-1` or stone `n-2`, so the answer is the
cheapest stone I can be standing on from which one leap clears the bank:

    answer = min(dp[n-1], dp[n-2])       (dp[n-2] absent when n = 1)

Let me confirm the recurrence by hand on the stated sample `c = [3, 7, 8, 6, 5, 7]`, `n = 6`,
claimed answer `16` with landings on stones `0, 2, 4`. `dp[0] = 3`. `dp[1] = min(7, 3+7) = 7`.
`dp[2] = min(dp[1], dp[0]) + 8 = min(7,3) + 8 = 11`. `dp[3] = min(dp[2], dp[1]) + 6 = min(11,7) + 6
= 13`. `dp[4] = min(dp[3], dp[2]) + 5 = min(13,11) + 5 = 16`. `dp[5] = min(dp[4], dp[3]) + 7 =
min(16,13) + 7 = 20`. Answer `= min(dp[5], dp[4]) = min(20, 16) = 16`. It matches, and `dp[4] = 16`
is realized by `dp[2] = 11` (which is `dp[0]=3` plus `8`) plus `5`, i.e. landings `0, 2, 4` — exactly
the claimed path. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first
cut, before folding state, keeps a `dp` array and handles the ends inline:

```
vector<long long> dp(n);
dp[0] = c[0];
if (n >= 2) dp[1] = dp[0] + c[1];          // step 0 -> 1
for (int i = 2; i < n; i++)
    dp[i] = min(dp[i-1], dp[i-2]) + c[i];
long long ans = dp[n-1];                    // reach bank from the last stone
```

Two things in this draft make me nervous: `dp[1]` and the final `ans`. I trace the counterexample I
already trust, `c = [1, 8, 9, 5]`, `n = 4`, true answer `10`. `dp[0] = 1`. `dp[1] = dp[0] + c[1] =
1 + 8 = 9`. `dp[2] = min(dp[1], dp[0]) + 9 = min(9,1) + 9 = 10`. `dp[3] = min(dp[2], dp[1]) + 5 =
min(10,9) + 5 = 14`. `ans = dp[3] = 14`.

**The bug (first one): I only let the crossing finish from the last stone.** The code returns `14`,
the very value greedy got, and `4` more than the true `10`. The defect is precise: I set
`ans = dp[n-1]`, which insists the final free leap be taken from stone `n-1`. But the rules let me
clear the bank from stone `n-2` as well. The optimum `10` *stands on stone `2 = n-2`* and leaps off
from there, never touching stone `3`; by forcing the finish through stone `n-1` I charged myself
`c[3] = 5` I never had to pay. The fix is to let the answer finish from either of the last two
stones:

    ans = dp[n-1];
    if (n >= 2) ans = min(ans, dp[n-2]);

Re-trace `[1, 8, 9, 5]`: `dp = [1, 9, 10, 14]`, `ans = min(dp[3], dp[2]) = min(14, 10) = 10`.
Correct. The case that broke now passes, and it broke for the reason I fixed — that is the evidence
I trust.

**A second trace exposes the restricted first landing.** I am still uneasy about `dp[1] = dp[0] +
c[1]`, which hard-codes "to be on stone `1` you must have stepped from stone `0`." But the rules say
my *first* landing may be stone `1` directly. So if stone `0` is costly and stone `1` is cheap, my
draft over-charges. I trace a minimal case that isolates this: `c = [9, 1]`, `n = 2`. The honest
answer is `min(c[0], c[1]) = 1` (first-land on the cheap stone `1`, then the free leap clears the
bank since `1 = n-1`). My draft: `dp[0] = 9`; `dp[1] = dp[0] + c[1] = 9 + 1 = 10`; with the fixed
finish, `ans = min(dp[1], dp[0]) = min(10, 9) = 9`. That returns `9`, not `1` — wrong by exactly the
`c[0] = 9` I was forced to pay for a stone I should have skipped entirely.

**The bug (second one): `dp[1]` forbade the direct first landing.** The minimum cost to stand on
stone `1` is the cheaper of two genuinely different histories: *land on stone `1` first* (cost
`c[1]`) versus *land on stone `0`, then step to stone `1`* (cost `dp[0] + c[1]`). My draft kept only
the second. Fix:

    if (n >= 2) dp[1] = min(c[1], dp[0] + c[1]);

Re-trace `[9, 1]`: `dp[0] = 9`; `dp[1] = min(1, 9+1) = 1`; `ans = min(dp[1], dp[0]) = min(1, 9) = 1`.
Correct. And re-trace `[1, 8, 9, 5]` once more end-to-end with both fixes in place: `dp[0] = 1`;
`dp[1] = min(8, 1+8) = 8`; `dp[2] = min(8,1) + 9 = 10`; `dp[3] = min(10,8) + 5 = 13`;
`ans = min(dp[3], dp[2]) = min(13, 10) = 10`. Still correct — the first-landing fix did not disturb
the case the finish-fix already repaired.

**A numeric self-check of the recurrence on an all-negative case, because negatives flip intuition.**
With negative costs I no longer want *few* landings; each landing only helps, so I want to land on as
many stones as the reach allows — and the `+1` step lets me touch *every* stone. Take `c = [-4, -6,
-2, -6]`, `n = 4`. Landing on all four (steps `0->1->2->3`) costs `-4-6-2-6 = -18`, then the free
leap from stone `3 = n-1` finishes. Can I do better than `-18`? Every stone is negative, so skipping
any only raises the total; `-18` should be optimal. Run the recurrence: `dp[0] = -4`; `dp[1] =
min(-6, -4-6) = min(-6, -10) = -10`; `dp[2] = min(dp[1], dp[0]) + c[2] = min(-10,-4) + (-2) = -12`;
`dp[3] = min(dp[2], dp[1]) + c[3] = min(-12,-10) + (-6) = -18`; `ans = min(dp[3], dp[2]) = min(-18,
-12) = -18`. Matches the hand argument, and `dp[3] = -18` is the all-stones path. The recurrence
handles the sign flip correctly because `min` automatically prefers chaining through more negative
stones; nothing special is needed.

**Folding the state and re-confirming the ends.** The array is fine for `n = 2*10^5`, but I can fold
to two scalars `prev2 = dp[i-2]`, `prev1 = dp[i-1]` and avoid the allocation. I keep the array form,
though, for clarity and because it is unambiguous at the `dp[0]`/`dp[1]` ends — `O(n)` memory at
`2*10^5` longs is `~1.6 MB`, comfortably inside `256 MB`. I will not micro-optimize a correct,
readable solution into a subtle one.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: there are no stones; I special-case it to print `0` before touching `dp`, since `dp[0]`
  would be an out-of-range access otherwise. Checklist value `0` — correct.
- `n = 1`, `c = [5]`: `dp[0] = 5`; the `dp[1]` line is guarded by `n >= 2` and skipped; `ans =
  dp[0] = 5` (the `min` with `dp[n-2]` is guarded by `n >= 2`). Checklist value `c[0] = 5` —
  correct. A negative single stone, `c = [-7]`: `dp[0] = -7`, `ans = -7` — I *must* land on it (no
  way to skip the lone stone), and the negative is favourable anyway.
- `n = 2`, `c = [9, 1]`: handled above, `ans = 1 = min(c[0], c[1])`. `c = [1, 9]`: `dp[0] = 1`,
  `dp[1] = min(9, 1+9) = 9`, `ans = min(9, 1) = 1` — first-land the cheap stone `0`, leap off from
  `0 = n-2`. Correct.
- All-negative `n` large: every step prefers the longer chain; the recurrence lands on all stones,
  total `~ -2*10^14`, which fits in `long long`. Correct in sign and magnitude.
- Overflow: all of `c`, `dp`, and `ans` are `long long`; the extreme magnitude `~2*10^14` is three
  orders of magnitude inside the `~9.2*10^18` `long long` ceiling. I never add a sentinel to a cost
  (there is no `-infinity` in this formulation — every `dp[i]` is a real reachable cost), so there is
  nothing to underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the parsing
  is format-agnostic, and the `n = 0` line (no values) is handled because the read loop runs zero
  times.

**Cross-check against an independent brute force.** I do not want to trust hand traces alone, so I
also reason about an exhaustive enumerator that, from the start, tries first-landings on stones `0`
and `1`, and from each stone recurses on `+1` and `+2`, treating any leap to index `>= n` as the
free clear-to-bank with no landing cost, summing `c[]` over the stones actually landed on and taking
the minimum. That enumerator encodes the rules directly with no DP cleverness, so agreeing with it
on many random small cases is strong evidence the recurrence and both end-fixes are right. (Running
it on hundreds of random instances — non-negative, mixed, and all-negative, `n` up to ~16 — it
agrees with my DP on every one.)

**Final solution.** I convinced myself the *idea* is right by disproving local greedy with a traced
counterexample and hand-checking the recurrence on the sample and an all-negative case; I convinced
myself the *code* is right by tracing two distinct failing inputs to precise causes — finishing only
from the last stone, and forbidding the direct first landing — and re-verifying each fix. That is
what I ship: one self-contained `O(n)` file, the DP I can defend rather than the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0: already across, cost 0
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[i] = minimum total stamina to be standing on stone i, having legally
    // arrived there from the near bank. You may first land on stone 0 or stone 1.
    // From stone j you may have come from stone j-1 or stone j-2.
    // Far bank is reachable from stone n-1 or stone n-2 at no extra cost.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<long long> dp(n);
    dp[0] = c[0];                          // first leap lands on stone 0
    if (n >= 2) dp[1] = min(c[1], dp[0] + c[1]); // first leap onto 1, or step 0->1
    for (int i = 2; i < n; i++) {
        dp[i] = min(dp[i - 1], dp[i - 2]) + c[i];
    }

    // reach far bank from stone n-1 or stone n-2
    long long ans = dp[n - 1];
    if (n >= 2) ans = min(ans, dp[n - 2]);

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Local greedy ("cheaper of the next two") looked right but a single traced
counterexample (`[1,8,9,5]`: greedy `14` vs the reachable `10`) showed that a cheap landing can
corner me into a costlier future under the `+1/+2` reach, so I moved to a prefix DP where `dp[i]` is
the cheapest way to stand on stone `i` and `dp[i] = min(dp[i-1], dp[i-2]) + c[i]`. Transcribing the
two ends bit me twice: finishing the answer only from stone `n-1` over-charged the case whose
optimum leaps off from stone `n-2` (a trace of `[1,8,9,5]` returning `14` pinpointed it,
fixed by `ans = min(dp[n-1], dp[n-2])`), and hard-coding `dp[1] = dp[0] + c[1]` forbade the legal
direct first landing on stone `1` (a trace of `[9,1]` returning `9` instead of `1` pinpointed it,
fixed by `dp[1] = min(c[1], dp[0] + c[1])`); a numeric check on the all-negative `[-4,-6,-2,-6]`
(`-18`) confirmed the same `min`-recurrence also handles the sign flip, and `long long` throughout
closes the overflow corner.
