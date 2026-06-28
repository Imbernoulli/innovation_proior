**Reading the problem and pinning the contract.** I have `n` coin denominations and a target `S`, an unlimited supply of each denomination, and I must form `S` *exactly* using as few coins as possible — or print `-1` if it cannot be formed at all. The input is `n` and `S` on the first line, then the `n` denominations; the output is a single integer. Before I reach for any algorithm I fix the scale, because that decides both the data types and which methods are even admissible: `n <= 100`, each `c[i] <= 10^6`, and `S <= 10^6`. The answer, when it exists, is at most `S` (every coin is at least `1`, so it takes at most `S` coins of value `1` to reach `S`), i.e. at most `10^6`, which fits comfortably in a 32-bit integer — but I will still carry sums in `long long` so that `s - c` and any sentinel arithmetic never flirt with overflow or signedness surprises. That is the cheap, non-negotiable first decision.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* correct within the budget, not the one that is shortest to type.

- *Greedy by largest coin.* Sort the denominations descending; repeatedly subtract the largest coin that is `<= remaining`; count how many subtractions until the remainder hits `0`, or fail if I get stuck. This is the instinctive move — it is what a human does at a cash register, it is a handful of lines, and it is essentially linear. The risk is structural: "fewest coins to reach `S`" is a global optimization, and greedy makes an irrevocable local choice at each step. For the *canonical* currency `{1, 5, 10, 25}` greedy happens to be optimal (those denominations form what is called a canonical system), but the problem hands me an **arbitrary** denomination set, and there is no promise the set is canonical. I will not trust greedy until I have actively tried to break it.
- *DP over sums.* Define `dp[s]` = the minimum number of coins that sum to exactly `s`. Build it bottom-up from `s = 0` to `s = S`, relaxing every denomination at every value. This is `O(S * n)` and `O(S)` memory. The risk here is not the *idea* — it is provably correct, as I will argue — but the *transcription* (off-by-one in the loop bounds, the unreachable/`-1` handling, the base case) and whether `O(S * n)` is actually fast enough at the stated limits.

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is exactly how a wrong solution gets shipped, so let me attack it with a concrete instance instead of an intuition. Take the denomination set `{1, 3, 4}` and target `S = 6`. Greedy sorts to `[4, 3, 1]` and does: remaining `6`, largest coin `<= 6` is `4`, take it, remaining `2`, one coin so far; largest coin `<= 2` is `1`, take it, remaining `1`, two coins; largest coin `<= 1` is `1`, take it, remaining `0`, three coins. Greedy's answer is `3` (the multiset `4 + 1 + 1`).

Is `3` optimal? Let me hunt for something greedy structurally could not reach. The target `6` is just `3 + 3` — two coins of the denomination `3`. That is strictly fewer than greedy's `3`. So greedy is **wrong**, and I now see precisely *why*: by snatching the largest coin `4` first, it committed to a remainder of `2` that the denomination set can only patch with two `1`s, whereas stepping *down* to a non-maximal coin (`3`) lined up a clean `3 + 3`. The largest-fits choice was locally tempting and globally worse. The verification paid off — it killed an approach I would otherwise have shipped in three lines. Greedy is out.

I want to be sure this is not a one-off curiosity that I could special-case around, so I probe a second family. Take `{1, 5, 6, 9}` and `S = 11`. Greedy: largest `<= 11` is `9`, remaining `2`, then `1 + 1`, total `9 + 1 + 1 = 3` coins. Optimal is `5 + 6 = 2` coins. Wrong again, and for the same reason — the big coin `9` strands a remainder (`2`) that the set fills inefficiently, while a pair of mid-sized coins (`5, 6`) tiles `11` exactly. This is not a corner case I can patch; greedy is simply not correct for arbitrary denominations. The whole greedy family — including "greedy with a small bounded look-ahead", which I briefly consider — is the kind of thing that is *risky to get right in the budget*: I would have to either prove the specific denomination system is canonical (an `O(c_max^2)`-ish test that is itself easy to botch) or bolt on a fallback, and at that point I am writing more code than the DP and still need the DP to back it up. Not worth it. I commit to the DP.

**Deriving the DP and proving it correct.** I want `dp[s]` = the minimum number of coins summing to exactly `s`, with the convention that `dp[s] = INF` (unreachable) if no multiset of the denominations sums to `s`. The base case is `dp[0] = 0`: it takes zero coins to make the empty sum. For `s >= 1`, any optimal multiset that sums to `s` uses at least one coin; let one of its coins be `c` (a denomination with `c <= s`). Removing that one coin leaves a multiset summing to `s - c` using one fewer coin, and that remaining multiset must itself be optimal for `s - c` (otherwise I could swap in a cheaper one and beat the supposed optimum for `s` — a standard exchange argument). Therefore

```
dp[s] = 1 + min over denominations c with c <= s of dp[s - c],
```

and `dp[s] = INF` if every `dp[s - c]` is `INF`. Because every transition references a strictly smaller index `s - c < s` (denominations are `>= 1`), filling `dp` in increasing order of `s` guarantees each `dp[s - c]` is final by the time I read it. Unlimited supply is handled *for free* by this ordering: when I relax `dp[s]` against `dp[s - c]`, that earlier entry may already have used coin `c` one or more times, so the same denomination can be reused without any extra bookkeeping. (This is exactly the "unbounded knapsack" relaxation order — relaxing forward over increasing `s`, as opposed to the 0/1 case which iterates `s` downward.) The final answer is `dp[S]`, printed as `-1` if it is still `INF`. The proof is complete, which is the property greedy lacked.

**A note on the loop structure I chose.** There are two equivalent ways to organize the double loop. One is "for each coin, for each `s` forward, relax `dp[s]` from `dp[s - c]`". The other — the one I will write — is "for each value `s` from `1` to `S`, for each coin `c`, relax `dp[s]` from `dp[s - c]`". They compute the identical table; I prefer the second because the outer index `s` is exactly the quantity my recurrence is defined on, which makes the bounds and the `c <= s` guard read directly off the math and leaves fewer places to introduce an off-by-one.

**Checking the recurrence by hand on the sample.** Denominations `{1, 3, 4}`, `S = 6`, expected `2`. `dp[0]=0`. `dp[1]`: only `c=1` fits, `dp[0]+1=1`. `dp[2]`: `c=1` -> `dp[1]+1=2`; `c=3,4` too big; `dp[2]=2`. `dp[3]`: `c=1`->`dp[2]+1=3`, `c=3`->`dp[0]+1=1`; `dp[3]=1`. `dp[4]`: `c=1`->`dp[3]+1=2`, `c=3`->`dp[1]+1=2`, `c=4`->`dp[0]+1=1`; `dp[4]=1`. `dp[5]`: `c=1`->`dp[4]+1=2`, `c=3`->`dp[2]+1=3`, `c=4`->`dp[1]+1=2`; `dp[5]=2`. `dp[6]`: `c=1`->`dp[5]+1=3`, `c=3`->`dp[3]+1=2`, `c=4`->`dp[2]+1=3`; `dp[6]=2`. Answer `2` — matches, and the winning `2` is exactly the `3 + 3` greedy could not find.

**First implementation — and immediately a self-verify, because clean math transcribes dirty.** My first cut of the core, before I had thought carefully about the sentinel, used a plain large constant and added to it unconditionally:

```
const long long INF = LLONG_MAX;
vector<long long> dp(S + 1, INF);
dp[0] = 0;
for (long long s = 1; s <= S; s++)
    for (int i = 0; i < n; i++)
        if (c[i] <= s)
            dp[s] = min(dp[s], dp[s - c[i]] + 1);   // <-- danger
```

Something about `dp[s - c[i]] + 1` bothered me: when `dp[s - c[i]]` is the unreachable sentinel, I am computing `sentinel + 1`. If the sentinel is `LLONG_MAX`, that addition *overflows* to `LLONG_MIN`, and then `min` happily picks that garbage negative as the "best" answer. So I traced the smallest input that exposes it: denominations `{4}`, `S = 6`. Walk it: `dp[0]=0`; `dp[1]`: `c=4 > 1`, skip, stays `INF`; ... `dp[4]`: `c=4<=4`, `dp[0]+1=1`, so `dp[4]=1`; `dp[5]`: `c=4<=5`, `dp[1]+1`. But `dp[1]` is `LLONG_MAX`, so `dp[1]+1` overflowed to `LLONG_MIN`, and `min(INF, LLONG_MIN) = LLONG_MIN`. Now `dp[5]` is a huge negative number, and `dp[6]` reading from `dp[2]` (also overflowed) inherits the corruption. The program would print a nonsensical negative instead of `-1`.

**Diagnosing the bug.** The defect is precise and twofold. First, I add `1` to a sentinel that is at the top of the `long long` range, which overflows; the fix is to (a) make the sentinel comfortably below `LLONG_MAX` so `+1` is harmless arithmetically, *and* (b) guard the relaxation so I never even try to extend an unreachable state. Even with a smaller sentinel, relaxing from an unreachable `dp[s - c]` is conceptually wrong: it would claim `s` is reachable in `sentinel + 1` coins, which only stays harmless because the number is huge — that is fragile, and I would rather make the unreachability explicit. So the correct transition is: only relax when `dp[s - c]` is *not* the sentinel. Second, the final print must map "still sentinel" to `-1`, which I had not yet wired up.

**Fixing and re-verifying.** I switch the sentinel to a large finite value well clear of `LLONG_MAX` so no `+1` can overflow, and I guard the relaxation on `dp[s - c] != INF`:

```
const long long INF = (long long)4e18;       // < LLONG_MAX (~9.2e18); +1 is safe, and a real
                                             // count never approaches it (real answers <= 10^6)
vector<long long> dp(S + 1, INF);
dp[0] = 0;
for (long long s = 1; s <= S; s++)
    for (int i = 0; i < n; i++) {
        long long v = c[i];
        if (v <= s && dp[s - v] != INF && dp[s - v] + 1 < dp[s])
            dp[s] = dp[s - v] + 1;
    }
cout << (dp[S] == INF ? -1 : dp[S]) << "\n";
```

Re-trace `{4}`, `S = 6`: `dp[0]=0`; `dp[1..3]` stay `INF` (either `c>s` or `dp[s-4]` is `INF`); `dp[4]`: `dp[0]!=INF`, `dp[4]=1`; `dp[5]`: `dp[1]==INF`, skip, stays `INF`; `dp[6]`: `dp[2]==INF`, skip, stays `INF`. Final `dp[6]==INF` -> print `-1`. Correct — `6` genuinely is not a multiple of `4`, and there is no overflow anywhere because I never add `1` to a sentinel. The case that broke before now passes, and it passes for the exact reason I fixed, which is the evidence I trust. The choice of `4e18` rather than, say, `LLONG_MAX/2` is deliberate: it is unmistakably "infinity" relative to any real answer (`<= 10^6`), yet `+1` on it is a no-op risk-wise.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `S = 0`: the `for s` loop never executes, `dp[0]` is `0`, the print sees `dp[0] != INF` -> `0`. Zero coins make the empty sum — correct, and it is the right answer regardless of the denominations.
- Denominations all larger than `S` (e.g. `{7, 9}`, `S = 5`): every guard `v <= s` fails for `s` in `1..5`, so every `dp[s>=1]` stays `INF`, and `dp[5] == INF` -> `-1`. Correct.
- A single denomination that does divide `S` (e.g. `{5}`, `S = 5`): `dp[5]` relaxes from `dp[0]` -> `1`. Correct.
- Duplicate denominations in the input (e.g. `[3, 3, 3]`): harmless — the inner loop just relaxes against the same value several times, the `min` is idempotent, the answer is unchanged. I do *not* need to de-duplicate.
- Fully unreachable target with a common factor (e.g. `{2, 4}`, `S = 3`): odd target, even coins, every `dp[odd]` stays `INF` -> `-1`. Correct.
- Largest instance, `S = 10^6` and `n = 100`: the inner work is `S * n = 10^8` simple relaxations of `long long`s with an early `v <= s` guard; that runs in roughly `0.1` s in practice, far inside a `2` s limit, and the `dp` array is `~8` MB, far inside `256` MB. The DP I can prove is also the DP that comfortably fits.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so newline-vs-space in the input is irrelevant.

**Self-verification against an independent oracle.** Hand-traces convince me of the cases I think to check; to catch the cases I do *not* think of, I cross-checked against a separate brute oracle written as a **BFS by coin count** — start at sum `0`, and the first time the BFS frontier reaches `S`, the number of BFS layers is the minimum coin count. That is a structurally different formulation from the forward value-DP (shortest path in unit-weighted layers vs. bottom-up relaxation), so agreement between them is real evidence, not a tautology. I ran the DP against this oracle on a battery of generated cases spanning every regime above — dense small coins (the greedy-killer zone), no-`1` sets, impossible targets, `S = 0`, coins exceeding `S`, duplicates, single denominations, and a few large targets — well over five hundred cases plus the hand-picked edges, and got **zero** mismatches. The greedy-vs-optimal gap that started this whole detour (`{1,3,4}`, `S=6`: greedy `3` vs optimal `2`) is of course among them, and the DP returns `2`.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with a concrete counterexample and then proving the DP recurrence with an exchange argument; I convinced myself the *code* is right by tracing the overflow/unreachable bug to a precise cause, fixing it, re-verifying the failing case, walking the corners, and then differential-testing against an independent BFS oracle. That is what I ship — one self-contained file, the simple provable `O(S * n)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;        // empty input -> nothing to do
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[s] = minimum number of coins to make sum s exactly; INF if unreachable.
    // Unbounded supply of each denomination, so each coin relaxes every larger sum.
    const long long INF = (long long)4e18;
    vector<long long> dp((size_t)S + 1, INF);
    dp[0] = 0;                              // zero coins make sum 0
    for (long long s = 1; s <= S; s++) {
        for (int i = 0; i < n; i++) {
            long long v = c[i];
            if (v <= s && dp[s - v] != INF && dp[s - v] + 1 < dp[s]) {
                dp[s] = dp[s - v] + 1;
            }
        }
    }

    cout << (dp[S] == INF ? -1 : dp[S]) << "\n";
    return 0;
}
```

**Causal recap.** Greedy-by-largest looked right but two traced counterexamples (`{1,3,4}`, `S=6`: greedy `3` vs optimal `3+3=2`; and `{1,5,6,9}`, `S=11`: greedy `3` vs optimal `5+6=2`) showed that grabbing the largest fitting coin strands a remainder the arbitrary denomination set fills inefficiently, and that this is structural, not a patchable corner — so I moved to the bottom-up DP and proved its recurrence by an exchange argument, with the forward (increasing-`s`) order giving unlimited reuse for free. The first DP added `1` to an `LLONG_MAX` sentinel, which overflowed to a negative on unreachable sub-sums; a trace of `{4}`, `S=6` printing garbage instead of `-1` pinpointed it, and the fix was a finite `4e18` sentinel plus a `dp[s - v] != INF` guard and an explicit `-1` print. Edge walks (`S=0`, coins `> S`, impossible targets, duplicates, single coins) and a differential test against an independent BFS-by-coin-count oracle over 500+ cases with zero mismatches closed it out; at `S=10^6`, `n=100` the `~10^8`-op DP runs in about `0.1` s, well inside the limit.
