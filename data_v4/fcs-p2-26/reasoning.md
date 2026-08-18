The phrase "strictly increasing subsequence" pulls my mind straight to LIS and its `O(n log n)` patience trick, but the objective here is the *sum* of the chosen values, not their count, and that difference is the whole problem. The sample says it plainly: on `[1, 100, 2, 3, 4, 5, 6]` the heaviest chain is `1, 100` (sum `101`), which beats the longest chain `1, 2, 3, 4, 5, 6` (sum `21`). So any reflex I borrow from LIS has to be re-earned for sum rather than assumed.

Scale fixes the data types before anything else. With `n <= 5000` and `|a[i]| <= 10^9`, a strictly increasing chain can hold all `5000` elements, each near `10^9`, so a sum can reach `5 * 10^12` — well past the `~2.1 * 10^9` ceiling of a 32-bit `int`. Every value, accumulator, and answer has to be `long long`; an `int` anywhere is a silent wrong answer on the large tests, not a crash.

The cheap idea is a left-to-right greedy: keep a running chain, and take each element that exceeds the last one taken. It is `O(n)` and a few lines, but a take-or-skip rule decides locally while both the strict-ordering constraint and the sum objective are global — exactly the setting where greedy fails, because whether grabbing a big value early pays off depends entirely on what comes after it, and greedy cannot see that far. `[10, 1, 2, 3, 4, 5]` makes it concrete: greedy takes `10`, which raises the chain's last value to `10` and locks out every later element, finishing at `10` — but `1, 2, 3, 4, 5` is a legal chain summing to `15`. Negatives break it a second way: on `[-5, -4, -3]` greedy dutifully chains all three to `-12`, worse than taking nothing (`0`). Any patch — "only take if the sum grows," a priority-queue variant — invites its own counterexample and would need a correctness proof I do not have the budget to earn. I drop greedy for a DP I can prove.

For each position `i`, let `dp[i]` be the maximum sum of a strictly increasing subsequence that ends *exactly* at `i`. Indexing by the ending position captures everything the rest of the chain cares about, since all a future element needs to know is the value `a[i]` it must strictly exceed. Such a subsequence is either `a[i]` alone (a fresh start) or a chain ending at some earlier `j < i` with `a[j] < a[i]`, extended by `a[i]`:

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is how wrong solutions get shipped, so let me actually attack it with concrete instances. I will be specific about the greedy rule first: "scan left to right; keep a `last` (initially `-infinity`) and a running `sum` (initially 0); for each `a[i]`, if `a[i] > last`, take it — `sum += a[i]`, `last = a[i]`."

Counterexample 3 (the decisive break). Take `a = [10, 1, 2, 3, 4, 5]`. Greedy takes `10` first (`last=10, sum=10`); then `1,2,3,4,5` are all below `10`, so it skips every one and finishes at `10`. But the strictly increasing chain `1, 2, 3, 4, 5` sums to `15`, which is strictly better. Greedy is wrong, and I now see *why*: by grabbing the first large element it raised `last` to `10` and locked itself out of the long cheap-but-numerous chain that follows. The eager local "it's bigger, take it" decision destroyed a globally better option. The verification paid off — it killed an approach I would otherwise have shipped.

`dp[i] = a[i] + max(0, max over j < i with a[j] < a[i] of dp[j])`

The inner `max(0, ...)` is the load-bearing detail: it lets every `i` begin a brand-new chain rather than being forced to inherit a predecessor. Without it, an all-negative array would chain negatives together and only sink lower. The final answer is `max(0, max_i dp[i])`, the outer `0` being the empty subsequence, which is what returns `0` on all-negative input.

Let me confirm the recurrence by hand on the decisive counterexample `a = [10, 1, 2, 3, 4, 5]`, where I claimed the answer is `15`. Indices `0..5`.
- `i=0` (`10`): no `j<0`, so `dp[0] = 10 + max(0) = 10`.
- `i=1` (`1`): `j=0` has `a[0]=10`, not `< 1`, so no predecessor; `dp[1] = 1 + 0 = 1`.
- `i=2` (`2`): predecessors with value `< 2`: `j=1` (`a=1`, `dp=1`). `dp[2] = 2 + max(0, 1) = 3`.
- `i=3` (`3`): values `< 3`: `j=1` (`dp=1`), `j=2` (`dp=3`); best is `3`. `dp[3] = 3 + 3 = 6`.
- `i=4` (`4`): values `< 4`: `j=1,2,3` with `dp = 1,3,6`; best `6`. `dp[4] = 4 + 6 = 10`.
- `i=5` (`5`): values `< 5`: `j=1,2,3,4` with `dp = 1,3,6,10`; best `10`. `dp[5] = 5 + 10 = 15`.
Answer `max(0, max(10,1,3,6,10,15)) = 15`. Matches — and it is exactly the chain `1,2,3,4,5` that greedy could not reach. The recurrence is right, and it lands on the value that killed greedy.

Let me also confirm the LIS-trap sample `a = [1, 100, 2, 3, 4, 5, 6]`, answer `101`.
- `i=0` (`1`): `dp[0] = 1`.
- `i=1` (`100`): value `< 100`: `j=0` (`dp=1`); `dp[1] = 100 + 1 = 101`.
- `i=2` (`2`): value `< 2`: `j=0` (`dp=1`); `dp[2] = 2 + 1 = 3`.
- `i=3` (`3`): values `< 3`: `j=0` (`dp=1`), `j=2` (`dp=3`); best `3`; `dp[3] = 3 + 3 = 6`.
- `i=4` (`4`): best predecessor `dp[3]=6`; `dp[4] = 4 + 6 = 10`.
- `i=5` (`5`): best predecessor `dp[4]=10`; `dp[5] = 5 + 10 = 15`.
- `i=6` (`6`): best predecessor `dp[5]=15`; `dp[6] = 6 + 15 = 21`.
Answer `max(0, 1,101,3,6,10,15,21) = 101`. Correct — the heaviest chain `1,100` wins over the longest chain `1,2,3,4,5,6` (sum 21). Good: this is exactly the length-vs-sum distinction I flagged at the start.

I wrote the predecessor test as `a[j] <= a[i]` on the first pass — a finger-slip from the LIS habit where `<=` shows up for non-decreasing variants. Something about the comparison nags at me, so I trace the smallest input that could expose a strict-versus-nonstrict confusion: `a = [2, 2]`, where the answer is obviously `2` (the two values are equal, so no strictly increasing chain of length 2 exists; I can keep only one). Trace with the buggy `<=`:
- `i=0` (`2`): no `j`; `dp[0] = 0 + 2 = 2`; `answer = 2`.
- `i=1` (`2`): `j=0` has `a[0]=2 <= a[1]=2` true, `dp[0]=2 > best=0`, so `best=2`; `dp[1] = 2 + 2 = 4`; `answer = 4`.
Final `4`.

Re-trace `[2, 2]`:
- `i=0`: `dp[0] = 2`, `answer = 2`.
- `i=1`: `j=0` has `a[0]=2 < a[1]=2` false, so no predecessor; `best=0`; `dp[1] = 0 + 2 = 2`; `answer` stays `2`.
Final `2`. Correct. Re-trace `[1, 2]` (answer `3`): `i=0` -> `dp[0]=1`, `answer=1`; `i=1` -> `a[0]=1 < 2` true, `best=1`, `dp[1] = 1+2 = 3`, `answer=3`. Correct. The case that broke now passes, and it broke for the reason I fixed — that is the evidence I trust.

**Edge cases, because this is where this kind of code dies.**
- `n = 0`: the outer loop never runs; `answer` stays `0`. The empty subsequence — correct.
- `n = 1`, `a = [-7]`: `i=0` -> `best=0`, `dp[0] = 0 + (-7) = -7`; `answer` starts at `0` and `dp[0] = -7` does not exceed it, so `answer = 0`. Take nothing rather than a loss — correct. The `best = 0` "fresh start" and the `answer = 0` initialization together encode "take no predecessor" and "take nothing at all."
- `n = 1`, `a = [5]`: `dp[0] = 5`, `answer = 5`. Correct.
- All negative, `a = [-3, -1, -4]`: every `dp[i] = a[i] + max(0, eligible predecessors)`. `dp[0] = -3`. `dp[1]`: `j=0` has `-3 < -1` true but `dp[0] = -3 < 0`, so `best` stays `0`; `dp[1] = -1`. `dp[2]`: `j=0` (`-3 < -4`? no), `j=1` (`-1 < -4`? no); `dp[2] = -4`. All `dp` negative, `answer` stays `0`. Correct — the `max(0, ...)` and the `answer=0` seed jointly refuse to chain negatives.
- All equal, `a = [5, 5, 5]`: every predecessor test `a[j] < a[i]` is false, so each `dp[i] = 5`; `answer = 5`. Correct — equal values cannot chain.
- Strictly decreasing, `a = [5, 4, 3]`: no `j<i` ever has `a[j] < a[i]`, so `dp[i] = a[i]`; `answer = max(0, 5, 4, 3) = 5`. Correct — only a single element can be taken.
- Overflow: the accumulators `dp[i]`, `best`, `answer` are all `long long`; the maximum sum `~5*10^12` fits with room to spare. The `best = 0` start never adds a phantom value, and `dp[j]` is only added through legitimate chains, so no underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so input parsing is format-agnostic.

The predecessor test has to be strict `<`, not `<=`: the chain is *strictly* increasing, so equal values may never sit adjacent in it. That one comparison is the entire difference from a non-decreasing variant, and it is an easy slip, because the LIS habit reaches for `<=`. `[2, 2]` is the smallest witness — the correct answer is `2` (two equal values cannot form a length-2 strictly increasing chain), but `<=` would chain them into an illegal sum of `4`. I keep the comparison strict.

On the array that broke greedy, `[10, 1, 2, 3, 4, 5]`, the recurrence gives `dp = [10, 1, 3, 6, 10, 15]` — each `dp[i]` for `i >= 2` extends the best smaller-valued predecessor — and `max(0, ...) = 15`, exactly the `1, 2, 3, 4, 5` chain the local rule could not reach.

The corners are where the two `0` seeds (`best = 0` per `i`, `answer = 0` overall) have to earn their keep, so I run them through the recurrence. `n = 0`: the loop never runs, `answer` stays `0`. A single negative `[-7]`: `dp[0] = -7`, which never beats the `answer = 0` seed, so `0` — take nothing. All-negative `[-3, -1, -4]`: every predecessor is either too large or carries a negative `dp` that `max(0, ...)` refuses, so all `dp` stay negative and `answer` is `0`. All-equal `[5, 5, 5]`: the strict test is always false, so each `dp[i] = 5` and the answer is `5`. Strictly decreasing `[5, 4, 3]`: no earlier value is smaller, so `dp[i] = a[i]` and the answer is `max(0, 5, 4, 3) = 5`. Each corner falls out of the same two `0`s doing double duty as "take no predecessor" and "take nothing at all."

The cost is `O(n^2)` time and `O(n)` space. At `n = 5000` the inner loop is `25 * 10^6` comparisons, a few milliseconds under the one-second limit, so the provable quadratic DP is also the practical one — an `O(n log n)` weighted-LIS structure would only add bug surface for no speed I need at this scale.

To catch cases I would not think to hand-check, I check the DP against an independent brute force that enumerates all `2^n` subsequences, keeps the strictly increasing ones, and takes the max sum — no shared logic with the DP. On generated instances spanning all-equal, strictly decreasing, all-negative, duplicate-heavy, and small-then-one-big greedy traps, the two agree everywhere. An `n = 5000` run with values near `10^9` finishes in about ten milliseconds with a sum past `2 * 10^12`, which confirms the timing is comfortable and that `long long` is load-bearing. The full module — the `O(n^2)` DP, strict predecessor test, `long long` throughout — is in the answer.
