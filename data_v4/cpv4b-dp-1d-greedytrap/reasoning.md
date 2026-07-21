I cross a river on stones `0..n-1`, leaping forward one or two stones at a time and paying `c[i]` —
possibly negative — each time I land. Three features of the contract shape everything. The first
landing off the bank is restricted to stone `0` or stone `1`. The last leap onto the far bank is
*free* and may be taken from stone `n-1` or stone `n-2`. And the reach is `+1/+2`, so each landing
fixes exactly which two stones I can reach next. That last property is the trap: it makes a purely
local rule tempting, because at every step there seem to be only two choices, yet the choice ripples
forward.

Scale first, because it fixes the data type. `n <= 2*10^5` and `|c[i]| <= 10^9`, so a single
crossing can accumulate about `2*10^14` in magnitude — three orders past the 32-bit `~2.1*10^9`
ceiling. Every cost and every running total must be 64-bit; an `int` here is a silent wrong answer
on the large tests, so I fix the data type before touching the algorithm.

The degenerate ends I pin as a checklist. `n = 0`: the near bank is the far bank, answer `0`.
`n = 1`: the only start is stone `0`, and `0 = n-1`, so the free leap finishes — answer `c[0]`.
`n = 2`: I first-land on stone `0` or `1`, and either clears the bank (`0 = n-2`, `1 = n-1`), so the
answer is `min(c[0], c[1])`.

The obvious solution is greedy: first-land on the cheaper of stone `0`/`1`, then always hop to the
cheaper of the next two reachable stones. It is `O(n)` and a few lines, but the global reach is
exactly the setting where a cheap-now landing can corner me, so I try to break it. Take
`c = [1, 8, 9, 5]`, `n = 4`. Greedy lands on stone `0` (`1`), then from `0` picks stone `1`
over stone `2` because `8 < 9`, then from `1` picks stone `3` over stone `2` because `5 < 9`, and
`3 = n-1` finishes: total `1 + 8 + 5 = 14`. But `0 -> 2` pays `1 + 9 = 10` and lands on stone
`2 = n-2`, from which a single free leap clears the bank — two landings instead of three. Greedy
grabbed stone `1` for being cheap, but standing on stone `1` still left me short of the bank and
forced a third landing; the local rule priced the next stone but not the future it commits to.
Greedy is out.

So I carry state. Let `dp[i]` be the minimum total cost over all legal crossings that end *standing
on stone `i`*. The future depends only on which stone I currently stand on, since that fixes my next
two options, so the recurrence reads backward in reach: to stand on stone `i` (`i >= 2`) I must have
just leapt from stone `i-1` or stone `i-2`, then paid `c[i]`:

    dp[i] = min(dp[i-1], dp[i-2]) + c[i].

The body is clearly right; the two ends are where transcription bites, because they encode the two
special rules of the contract. My first cut writes them the naive way — to be on stone `1` you
stepped from stone `0`, and you finish from the last stone:

```
vector<long long> dp(n);
dp[0] = c[0];
if (n >= 2) dp[1] = dp[0] + c[1];          // step 0 -> 1
for (int i = 2; i < n; i++)
    dp[i] = min(dp[i-1], dp[i-2]) + c[i];
long long ans = dp[n-1];                    // reach bank from the last stone
```

Trace the counterexample I already trust, `c = [1, 8, 9, 5]`, true answer `10`. `dp[0] = 1`;
`dp[1] = 1 + 8 = 9`; `dp[2] = min(9, 1) + 9 = 10`; `dp[3] = min(10, 9) + 5 = 14`; `ans = dp[3] =
14`. That is greedy's wrong value, `4` too high. The defect is `ans = dp[n-1]`, which insists the
free leap be taken from stone `n-1`; but the optimum stands on stone `2 = n-2` and leaps off from
there, never touching stone `3`, so I charged myself `c[3] = 5` I never had to pay. Let the finish
come from either of the last two stones:

    ans = dp[n-1];
    if (n >= 2) ans = min(ans, dp[n-2]);

Re-trace: `ans = min(14, 10) = 10`, correct.

The second end is still wrong: `dp[1] = dp[0] + c[1]` hard-codes "to be on stone `1` you stepped
from stone `0`", but the rules let stone `1` be the *first* landing. A minimal case isolates it:
`c = [9, 1]`, `n = 2`, honest answer `min(c[0], c[1]) = 1`. The draft gives `dp[0] = 9`,
`dp[1] = 9 + 1 = 10`, `ans = min(10, 9) = 9` — wrong by exactly the `c[0] = 9` I was forced to pay
for a stone I should have skipped. The minimum cost to stand on stone `1` is the cheaper of two
genuinely different histories — land on stone `1` first (`c[1]`) or land on stone `0` then step
(`dp[0] + c[1]`):

    if (n >= 2) dp[1] = min(c[1], dp[0] + c[1]);

Re-trace `[9, 1]`: `dp[1] = min(1, 10) = 1`, `ans = min(1, 9) = 1`, correct. And `[1, 8, 9, 5]` end
to end with both fixes: `dp = [1, 8, 10, 13]`, `ans = min(13, 10) = 10` — the first-landing fix did
not disturb the case the finish fix repaired.

With both ends fixed I run the given sample `c = [3, 7, 8, 6, 5, 7]`: `dp = [3, 7, 11, 13, 16, 20]`,
`ans = min(20, 16) = 16`, realized by landings `0, 2, 4` — matching the stated crossing.

Negative costs invert the objective: I no longer want *few* landings,
and since a `+1` step touches every stone I can in principle land on all of them. Nothing special is
needed, because `min` already prefers chaining through more-negative stones. On `c = [-4, -6, -2,
-6]` the recurrence gives `dp = [-4, -10, -12, -18]`, `ans = -18`, which is the land-on-everything
path, and every stone being negative makes skipping any one strictly worse. The sign flip is handled
by the same recurrence.

The array is `O(n)` — about `1.6 MB` of `long long` at `n = 2*10^5`, comfortably inside `256 MB`.
It folds to two scalars, but I keep the array: it is unambiguous at the `dp[0]`/`dp[1]` ends and
readability is worth more than a saved allocation here. `n = 0` needs its own branch before touching
`dp`, since `dp[0]` would otherwise be an out-of-range access; it prints `0`. Overflow is closed by
`long long` throughout — the extreme `~2*10^14` sits three orders inside the `~9.2*10^18` ceiling,
and since every `dp[i]` is a real reachable cost there is no `-infinity` sentinel to underflow.

To back the hand traces I also test against an exhaustive enumerator that encodes the rules directly
— first-land on stone `0` or `1`, recurse on `+1`/`+2`, treat any leap to index `>= n` as the free
clear-to-bank, and minimize the sum over landed stones. On hundreds of random instances (non-
negative, mixed, and all-negative, `n` up to ~16) it agrees with the DP on every one.

What I ship is this prefix DP, one self-contained `O(n)` file; the full module is in the answer.
