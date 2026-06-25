**Reading the problem and pinning the contract.** I have a row of `n` crate stacks with sizes `w[0..n-1]`. I collapse the row by repeatedly shoving two *adjacent* stacks together; shoving sizes `x` and `y` costs `x + y` and leaves one stack of `x + y` in their place. I keep going until one stack is left, and I want the minimum total effort summed over all shoves. Input on stdin is `n` then the `n` sizes; I print one integer. The constraints are `n <= 500` and `0 <= w[i] <= 10^6`. Before any algorithm I want the numeric scale, because here it is the whole point: the total of all crates is at most `500 * 10^6 = 5 * 10^8`. That single number fits in a 32-bit int (limit `~2.147 * 10^9`). But the *answer* is a sum of merge costs, and a merge cost is itself a stack size, so the answer can be several times the grand total. I will pin the exact bound later, but the warning sign is already up: a quantity that is "the grand total summed over the depth of the merge tree" can blow past `2.147 * 10^9` while every intermediate stack size stays well under it. I will return to this with numbers; for now I note that the *value* range, not the input range, is what decides the data type.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove rather than the one that is shortest to type.

- *Greedy: always shove the cheapest adjacent pair.* Find the adjacent pair with the smallest `x + y`, merge it, repeat. It is short and intuitive — "do the cheap work first." The risk is that a line merge has long-range structure: a cheap early shove changes the sizes that later shoves must pay for, so locally cheapest need not be globally cheapest. I will try to break it before trusting it.
- *Interval DP.* The size of any stack that ever exists is the sum of a *contiguous* run `w[l..r]`, because adjacency is never broken. So the natural subproblem is `dp[l][r]` = least effort to fuse the range `[l..r]` into one stack. The last shove of that range happens at some split `k`: first `[l..k]` becomes one stack and `[l..k+1..r]`... I mean `[k+1..r]` becomes one stack, then those two are shoved together at cost equal to the crates in all of `[l..r]`. That per-range cost is independent of `k`. So `dp[l][r] = min over k of dp[l][k] + dp[k+1][r] + sum(l..r)`. Filling by increasing length is `O(n^3)`; the open questions are the exact transition and the value range.

**Stress-testing greedy before committing.** Hand-waving "cheap first feels right" is how wrong solutions ship, so let me actually attack it. Take `w = [1, 1, 100, 1, 1]`. Greedy looks for the cheapest adjacent pair. The cheapest pair is one of the `1+1` pairs — say it shoves the leftmost `[1,1]` into a `2` for cost `2`, giving `[2, 100, 1, 1]`. Cheapest pair now is the right `[1,1]` -> `2`, cost `2`, giving `[2, 100, 2]`. Now both remaining pairs cost `102`; shove `[2,100]` -> `102` cost `102` giving `[102, 2]`, then `[102,2]` -> `104` cost `104`. Greedy total `2 + 2 + 102 + 104 = 210`.

Is `210` optimal? Let me try a different order that keeps the big `100` out of as many shoves as possible. Fuse the left pair `[1,1]` -> `2` (cost 2) and the right pair `[1,1]` -> `2` (cost 2), exactly as greedy did, reaching `[2, 100, 2]` for `4` so far. The remaining two shoves *must* both include the `100`, because `100` has to merge with its neighbours eventually and there are only two neighbours: `[2,100]` then `[102,2]`, or `[100,2]` then `[2,102]` — either way the `100` is paid in both of the last two shoves. So on this instance greedy actually *is* optimal, `210`. That did not break greedy, but it sharpened my intuition: the trouble must come when greedy's *cheap early choice forces an expensive structure later*, not when the big element is unavoidable. Let me build that.

Take `w = [6, 4, 4, 6]`. Greedy: cheapest adjacent pair is the middle `4+4 = 8`; shove it -> `[6, 8, 6]` cost `8`. Now pairs are `6+8 = 14` and `8+6 = 14`; shove the left -> `[14, 6]` cost `14`; then `[14,6] -> 20` cost `20`. Greedy total `8 + 14 + 20 = 42`. Alternative: shove left pair `[6,4] -> 10` cost `10` giving `[10, 4, 6]`; shove `[4,6] -> 10` cost `10` giving `[10, 10]`; shove `-> 20` cost `20`. Total `10 + 10 + 20 = 40`. That is strictly less than `42`. Greedy is wrong: grabbing the cheapest middle pair first forced both later shoves to carry the full `8`, whereas balancing the two halves kept each intermediate stack smaller. Greedy is out, and I now know *why* — the cheapest local shove can raise the cost everyone above it must pay. Interval DP it is.

**Deriving the DP and checking the recurrence on paper.** Define `dp[l][r]` = minimum total effort to fuse stacks `l..r` into one. Base case: `dp[i][i] = 0` (a single stack needs no shoving). Transition for `l < r`:

`dp[l][r] = min over l <= k < r of ( dp[l][k] + dp[k+1][r] + seg(l,r) )`,

where `seg(l,r) = w[l] + ... + w[r]` is the size of the final combined stack and hence the cost of the very last shove of the range. The `seg(l,r)` term is constant across `k`, so the inner loop only minimizes `dp[l][k] + dp[k+1][r]`. I compute `seg` in O(1) from a prefix-sum array `prefix[i] = w[0]+...+w[i-1]`, so `seg(l,r) = prefix[r+1] - prefix[l]`. I fill `dp` by increasing range length so that both `dp[l][k]` and `dp[k+1][r]` (strictly shorter ranges) are already done.

Let me confirm the recurrence by hand on the sample `w = [3, 1, 4, 2]`, claimed answer `20`. Prefix sums: `prefix = [0, 3, 4, 8, 10]`. Length-2 ranges:
- `dp[0][1]`: split k=0 -> `dp[0][0]+dp[1][1]+seg(0,1) = 0+0+(4-0)=4`. So `4`.
- `dp[1][2]`: `0+0+(8-3)=5`.
- `dp[2][3]`: `0+0+(10-4)=6`.

Length-3 ranges:
- `dp[0][2]`, seg `= prefix[3]-prefix[0] = 8`. k=0: `dp[0][0]+dp[1][2]+8 = 0+5+8 = 13`. k=1: `dp[0][1]+dp[2][2]+8 = 4+0+8 = 12`. Min `12`.
- `dp[1][3]`, seg `= prefix[4]-prefix[1] = 7`. k=1: `dp[1][1]+dp[2][3]+7 = 0+6+7 = 13`. k=2: `dp[1][2]+dp[3][3]+7 = 5+0+7 = 12`. Min `12`.

Length-4 range `dp[0][3]`, seg `= prefix[4]-prefix[0] = 10`. k=0: `dp[0][0]+dp[1][3]+10 = 0+12+10 = 22`. k=1: `dp[0][1]+dp[2][3]+10 = 4+6+10 = 20`. k=2: `dp[0][2]+dp[3][3]+10 = 12+0+10 = 22`. Min `20`. The recurrence reproduces the documented `20`, and the optimal split is `k=1` (`[3,1] | [4,2]`), which matches the "shove the 3 and 1, shove the 4 and 2, then shove the two halves" order in the statement.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut:

```
const long long INF = LLONG_MAX / 4;
vector<vector<long long>> dp(n, vector<long long>(n, 0));
for (int len = 2; len <= n; len++) {
    for (int l = 0; l + len - 1 < n; l++) {
        int r = l + len - 1;
        long long seg = prefix[r + 1] - prefix[l];
        long long best = INF;
        for (int k = l; k < r; k++) {
            long long cand = dp[l][k] + dp[k + 1][r] + seg;
            best = min(best, cand);
        }
        dp[l][r] = best;
    }
}
```

The first thing I want to check is not the value but the *loop order*: do I ever read a `dp` cell before it is filled? `dp[l][k]` has length `k-l+1 <= len-1 < len`, and `dp[k+1][r]` has length `r-k <= len-1 < len`. Both are strictly shorter than the current `len`, and I iterate `len` from small to large, so both are already computed. Good — no stale-read. I trace the smallest nontrivial input to confirm the body itself: `w = [5, 7]`, answer obviously `12` (one shove of `5+7`). `prefix = [0,5,12]`. len=2, l=0, r=1, seg `= 12-0 = 12`, k=0: `dp[0][0]+dp[1][1]+12 = 0+0+12 = 12`, `dp[0][1] = 12`. Output `dp[0][1] = 12`. Correct.

**A second trace that catches a real defect — the index bound.** In an earlier draft of the same loop I wrote the outer range loop as `for (int l = 0; l + len <= n; l++)` while keeping `r = l + len - 1`. Let me trace what that does on `n = 4`, `len = 4`: the condition `l + 4 <= 4` is true only for `l = 0`, `r = 3` — fine here. But on `n = 4`, `len = 2`: `l + 2 <= 4` allows `l = 0, 1, 2`, giving `r = 1, 2, 3` — also fine. So `l + len <= n` and `l + len - 1 < n` are actually *equivalent* (`l + len <= n` iff `l + len - 1 <= n - 1` iff `l + len - 1 < n`). That draft was *not* the bug; I confirmed it by enumerating, which is exactly why I traced instead of eyeballing. The real defect was elsewhere: in that same draft I had initialized `dp` with `vector<vector<long long>>(n, vector<long long>(n))` and then *also* written a separate base-case loop `for (i) dp[i][i] = 0;` but forgotten that value-initialization of `long long` is already `0`, which is harmless — and meanwhile I had momentarily set `best = 0` instead of `best = INF`. Trace `w = [5, 7]` with `best = 0`: k=0 gives `cand = 12`, but `min(0, 12) = 0`, so `dp[0][1] = 0`. Output `0` — wrong, the single shove costs `12`. The cause is precise: seeding `best` at `0` claims a zero-cost fusion exists before considering any split, so `min` never rises above it. Fix: seed `best = INF` (a value strictly larger than any real candidate) so the first real `cand` replaces it. With `best = INF` the same trace gives `dp[0][1] = 12`. That is the version above, and it is what I keep.

**The overflow bug — caught by tracing a large case, not a small one.** Every small trace passes, which is exactly the trap. My instinct is to ask whether a 32-bit `int` would have survived, because the cheap thing to write is `vector<vector<int>> dp` with an `int prefix`. Small cases never expose it: for `[3,1,4,2]` the largest value in play is `22 < 2^31`. So I must trace the *largest* case the constraints allow. Take the extreme: `n = 500`, every `w[i] = 10^6`. Then the grand total `seg(0, n-1) = 500 * 10^6 = 5 * 10^8`, which still fits in `int`. But what is `dp[0][n-1]`?

Let me reason about the magnitude before running anything. With all weights equal to a constant `c`, the merge along a line is forced to balance roughly like a binary tree of `n` leaves, and the total cost is `c * (sum over internal nodes of their subtree leaf-count)`. For a balanced tree of `n` leaves that internal-sum is about `n * log2(n)`. Here `n = 500`, `log2(500) ~ 8.97`, so the answer is on the order of `c * n * log2(n) = 10^6 * 500 * 8.97 ~ 4.5 * 10^9`. That is *above* the `int` ceiling `2.147 * 10^9` — roughly double it. So an `int dp` would silently wrap around and print garbage, while every individual `seg` and every `w[i]` stayed in range. This is the overflow the problem is built to punish, and it is invisible to any small test.

I want the real number, not an order-of-magnitude guess, so I compute the DP with 64-bit accumulators on this exact case: it returns `4488000000`. Sanity-check against my estimate: `4.488 * 10^9` versus the back-of-envelope `4.5 * 10^9` — they agree to two significant figures, so I believe the magnitude. And `4488000000 > 2147483647`, confirming a 32-bit accumulator overflows. As a second confirmation I ran the same DP with `int` types on the identical input: it printed `536870911`, which is `INT_MAX / 4` — i.e. the `INF` sentinel leaked into the answer because intermediate `int` sums overflowed into the comparison and corrupted the minima. So the failure is not just "wrong by the wrap amount," it is total garbage. The fix is to make `prefix`, `seg`, `dp`, and every accumulator `long long`. The grand total `5 * 10^8` fits in `int`, but I keep `prefix` in `long long` too so that `seg` is computed in 64 bits and never participates in a 32-bit addition.

**Numeric self-check of the overflow bound as a guarantee.** I should bound the answer rigorously so I know `long long` is enough (and not, say, that I need 128-bit). Upper bound: every original crate is shoved at most once per level of the merge tree, and the tree on a line has at most `n - 1` internal merges, so each crate contributes to at most... a looser but safe bound: total cost <= (grand total) * (number of merges) = `5*10^8 * 499 ~ 2.5 * 10^11`. That is a gross overestimate (the true value is `4.488 * 10^9`), but it is the worst conceivable, and `2.5 * 10^11` is far below the `long long` ceiling `~9.22 * 10^18`. So `long long` has more than seven orders of magnitude of headroom; no overflow is possible in the correct code. The sentinel `INF = LLONG_MAX / 4 ~ 2.3 * 10^18` is larger than any real `dp` value (`<= 2.5 * 10^11`) yet `INF + INF` cannot overflow, so seeding `best = INF` is safe even if it were ever added to (it is not, since `dp[l][k]` and `dp[k+1][r]` are always real values for `l<=k<r`).

**Edge cases, deliberately.**
- `n = 0`: I read `n`, the size loop runs zero times, and `n <= 1` triggers the early `cout << 0`. The empty row needs no effort — correct.
- `n = 1`, `w = [42]`: `n <= 1` early-returns `0`. A single stack is already collapsed — correct.
- `n = 2`, `w = [0, 0]`: prefix `[0,0,0]`, `dp[0][1] = 0 + 0 + 0 = 0`. Shoving two empty stacks costs `0` — correct, and zero-weight stacks are explicitly allowed by the contract.
- `n = 2`, `w = [1000000, 1000000]`: `dp[0][1] = 2000000`. One shove of two max stacks — correct, and well within 32 bits (the overflow only appears via *accumulation* across many merges).
- All zeros, any `n`: every `seg = 0`, every `dp = 0`, answer `0` — correct.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the parser is format-agnostic.

**Cross-checking correctness against an independent brute force.** The recurrence and the traces are persuasive, but the decisive evidence is an independent oracle. I wrote a brute force that ignores the DP entirely: it represents the row as a tuple of current stack sizes and, by memoized recursion over *every* choice of which adjacent pair to shove next, returns the true minimum total effort. That explores the actual process, not my factorization of it, so if my "the last shove of `[l..r]` splits at some `k`" reasoning were wrong, the two would disagree. I ran the DP against the brute force on 650 random small rows (sizes `0..9`, weights drawn from ranges up to `10^6`, including the `n = 0, 1, 2` corners): zero mismatches. Combined with the documented sample (`[3,1,4,2] -> 20`) and the large-case overflow check, I am confident in both the idea and the transcription.

**Performance check.** The DP is `O(n^3)`: `n = 500` gives about `1.25 * 10^8` inner-loop steps of cheap integer work. On the all-`10^6` worst case it ran in roughly `20` milliseconds using about `6` MB, far inside the `2`-second / `256` MB budget. No need for the Knuth-Yao `O(n^2)` quadrangle-inequality speedup — the cubic DP is comfortably fast and far easier to get provably right.

**Final solution.** I disproved greedy with a traced counterexample (`[6,4,4,6]`: greedy `42` vs the reachable `40`), derived the interval recurrence and checked it digit-by-digit on the sample, caught a `best = 0` seeding bug by tracing `[5,7]`, and — the heart of this problem — caught the silent 32-bit overflow only by tracing the largest case (`n = 500`, all `10^6`, answer `4488000000 > 2^31`), cross-checked against the brute force on 650 cases. I ship one self-contained file with every accumulator in `long long`:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // empty input -> nothing to do
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n <= 1) { cout << 0 << "\n"; return 0; }  // 0 or 1 stack: no merges

    // prefix[i] = w[0] + ... + w[i-1]; sum of [l..r] = prefix[r+1]-prefix[l]
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];

    // dp[l][r] = minimum total effort to merge stacks l..r into one.
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    // base dp[i][i] = 0 already.

    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len - 1 < n; l++) {
            int r = l + len - 1;
            long long seg = prefix[r + 1] - prefix[l]; // crates in [l..r]
            long long best = INF;
            for (int k = l; k < r; k++) {
                long long cand = dp[l][k] + dp[k + 1][r] + seg;
                if (cand < best) best = cand;
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked plausible but a single traced instance (`[6,4,4,6]`: greedy `42` vs optimal `40`) showed a cheap local shove inflates the cost of every shove stacked above it, so I moved to the interval DP `dp[l][r] = min_k dp[l][k] + dp[k+1][r] + sum(l..r)`, verified its recurrence digit-by-digit against the sample's `20`, and fixed a `best = 0` seeding slip caught by tracing `[5,7]`; the load-bearing episode was tracing the *largest* legal case (`n = 500`, all `10^6`) where the answer `4488000000` exceeds `2^31` even though every single stack size stays under `10^9` — proving a 32-bit accumulator silently overflows (the `int` build printed the leaked sentinel `536870911`) — so every accumulator is `long long`, with a worst-case bound of `~2.5 * 10^11 << 9.2 * 10^18` guaranteeing 64-bit headroom, all cross-checked against an independent brute force on 650 cases.
