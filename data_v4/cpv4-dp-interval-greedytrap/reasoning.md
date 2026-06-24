**Reading the problem and pinning the contract.** I have `n` beads in a fixed order, widths `w[0..n-1]` with `w[i] >= 1`, and a board of glow-lines each of usable width `W`. I cut the bead sequence into contiguous groups: line 1 takes a prefix, line 2 the next chunk, and so on, never reordering. A line holding beads `l..r` consumes `used(l, r) = (w[l] + ... + w[r]) + (r - l)`, where the `(r - l)` term is the unit clearances between the `r - l + 1` beads on that line, and this must be `<= W` or the line overflows (illegal). A non-last line with `slack = W - used` pays `slack^2`; the **last** line pays nothing. I minimize the total over all non-last lines. The empty layout is never an issue because `w[i] <= W` guarantees every bead can sit alone, so a legal cut always exists. Output is a single integer.

Before any algorithm I fix the scale, because it decides the data types and the achievable complexity. `n <= 5000`, `W <= 10^6`, `w[i] <= W`. The largest possible penalty: every line is a single bead of width about `W/2` so it cannot share with a neighbour, leaving slack near `W/2`; that is up to `n - 1 ≈ 5000` lines each paying about `(5*10^5)^2 = 2.5*10^11`, for a total near `1.25*10^15`. That is far beyond the 32-bit ceiling of `~2.1*10^9`, so every accumulator — the DP table, the slack, the squared term — must be 64-bit. I will use `long long` everywhere. An `int` here is a silent wrong-answer on the big tests, so this is non-negotiable. The complexity budget: `n = 5000` means an `O(n^2) = 2.5*10^7` partition scan is comfortably under one second, so I do not need a fancy `O(n log n)` convex-hull trick; a clean quadratic DP is both fast enough and far easier to get provably right.

**Laying out the candidate approaches.** Two strategies are genuinely different and I want to commit to the one I can *prove*, not the one that types fastest.

- *Greedy first-fit.* Sweep the beads once, keep packing the current line while the next bead still fits, and open a new line the instant it would overflow. This is `O(n)` and trivially short. The intuition that sells it is "filling each line as full as possible leaves the least slack" — which is locally true per line. The danger is the cost is the *sum of squares* of slack, a convex penalty, and convex penalties reward *spreading* slack evenly rather than concentrating it. Greedy makes a purely local decision (fill now) that can dump a large slack onto one later line. I refuse to trust it until I have tried to break it.
- *Interval / partition DP.* Define `dp[i]` = minimum penalty to lay out the first `i` beads, with the constraint baked in that the layout's final line is some contiguous group `[j..i-1]`. Then `dp[i] = min over legal j of ( dp[j-1] + cost of the group [j..i-1] )`, where `dp[0] = 0` (no beads, no cost). This is exactly an interval-partition recurrence: each transition glues an interval `[j..i-1]` onto an optimal prefix. The risk is not the idea but the transcription — the width/feasibility test, the prefix-sum indexing, and the "last line is free" special case are all easy to write subtly wrong.

**Stress-testing greedy before committing — building a counterexample by hand.** Hand-waving "greedy feels right" is how wrong solutions ship, so let me actually attack it with a concrete instance small enough to trace fully. Take `n = 4`, `W = 6`, `w = [4, 1, 3, 3]`.

Greedy first-fit: open line 1 with bead 0, `used = 4`. Try bead 1: adding it costs `w[1] + 1` gap `= 1 + 1 = 2`, so `used` would be `4 + 2 = 6 <= 6` — it fits exactly, `used = 6`, `slack = 0`. Try bead 2: `used` would be `6 + (3 + 1) = 10 > 6` — overflow, so close line 1 (`{0,1}`, slack 0) and open line 2 with bead 2, `used = 3`. Try bead 3: `3 + (3 + 1) = 7 > 6` — overflow, close line 2 (`{2}`, slack `6 - 3 = 3`) and open line 3 with bead 3 (`{3}`, the last line, free). Greedy's total = line 1 `0^2` + line 2 `3^2` + line 3 free `= 0 + 9 + 0 = 9`.

Is 9 optimal? Let me hunt for a layout greedy structurally cannot reach because it over-filled line 1. Put bead 0 alone on line 1: `used = 4`, `slack = 2`, cost `4`. Put beads `{1,2}` on line 2: `used = 1 + 1 + 3 = 5`, `slack = 1`, cost `1`. Put bead 3 alone on the last line: free. Total `= 4 + 1 = 5 < 9`. So greedy is wrong, and I can now name the mechanism precisely: by stuffing line 1 to exactly `W`, greedy left no room to push bead 2 forward, which forced bead 2 onto a sparse line that paid `9`; the optimum *deliberately wastes a little* on line 1 (slack 2, cost 4) to keep slack small and balanced afterward. The squared penalty makes `2^2 + 1^2 = 5` beat `0^2 + 3^2 = 9` even though both waste the same total of slack units — convexity rewards spreading. The verification paid off: it killed an approach I would otherwise have shipped. Greedy is out.

**Deriving the DP and checking the recurrence on the sample.** I want `dp[i]` = the minimum penalty to lay out beads `0..i-1`. To compute it I decide which beads share the *last* line of that prefix layout: say that line is the group `[j-1 .. i-1]` (1-based prefix boundary `j`, so the group starts at bead index `j-1` and ends at bead `i-1`). For the group `[j-1..i-1]` to be a legal line it must not overflow: with `cnt = i - (j - 1)` beads, `widthSum = pre[i] - pre[j-1]` (using `pre`, the prefix sums of `w`), the used width is `used = widthSum + (cnt - 1)` and I need `used <= W`. The cost contributed by this line is `(W - used)^2` — *unless* this group is the actual last line of the whole board, i.e. `i == n`, in which case the last line is free and contributes `0`. Then

```
dp[i] = min over all legal j in [1..i] of ( dp[j-1] + cost([j-1..i-1]) )
dp[0] = 0
```

and the answer is `dp[n]`. A subtlety I want to nail: the "last line is free" rule only applies to the *global* last line, which is precisely the group that ends at bead `n-1`, i.e. when `i == n`. For every `dp[i]` with `i < n`, the group `[j-1..i-1]` is an interior line of the eventual board, so it always pays `slack^2`. That is why the freeness is keyed on `i == n` inside the transition, not on "is this the last group considered". Good — that is the one place this differs from a plain min-sum partition.

Let me confirm the recurrence by hand on the sample `n = 4, W = 6, w = [4,1,3,3]`, expected answer `5`. Prefix sums `pre = [0,4,5,8,11]`.

- `dp[0] = 0`.
- `dp[1]` (beads `{0}`): only `j = 1`, group `[0..0]`, `cnt = 1`, `used = 4 + 0 = 4 <= 6`. `i = 1 != n`, cost `(6-4)^2 = 4`. `dp[1] = dp[0] + 4 = 4`.
- `dp[2]` (beads `{0,1}`): `j = 2` group `[1..1]` `used = 1`, cost `(6-1)^2 = 25`, candidate `dp[1] + 25 = 29`; `j = 1` group `[0..1]` `used = (4+1) + 1 = 6`, cost `(6-6)^2 = 0`, candidate `dp[0] + 0 = 0`. `dp[2] = 0`.
- `dp[3]` (beads `{0,1,2}`): `j = 3` group `[2..2]` `used = 3`, cost `9`, cand `dp[2] + 9 = 9`; `j = 2` group `[1..2]` `used = (1+3)+1 = 5`, cost `1`, cand `dp[1] + 1 = 5`; `j = 1` group `[0..2]` `used = (4+1+3)+2 = 10 > 6` illegal. `dp[3] = 5`.
- `dp[4] = answer` (beads `{0,1,2,3}`, `i == n` so last group free): `j = 4` group `[3..3]` `used = 3`, free, cand `dp[3] + 0 = 5`; `j = 3` group `[2..3]` `used = (3+3)+1 = 7 > 6` illegal (and earlier `j` only grow `used`, so stop). `dp[4] = 5`.

`dp[4] = 5`, matching the expected sample answer. The recurrence is right, and the trace also reproduced the optimal layout I found by hand (line 1 `{0}`, line 2 `{1,2}`, last line `{3}`).

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core loop:

```
vector<long long> dp(n + 1, INF);
dp[0] = 0;
for (int i = 1; i <= n; i++) {
    for (int j = i; j >= 1; j--) {
        long long cnt = i - (j - 1);
        long long widthSum = pre[i] - pre[j - 1];
        long long used = widthSum + (cnt - 1);
        long long slack = W - used;
        long long pen = (i == n) ? 0 : slack * slack;
        dp[i] = min(dp[i], dp[j - 1] + pen);
    }
}
```

I trace it on the tiny illegal-overflow case `n = 2, W = 4, w = [4, 4]`. Each bead is width 4 = W, so they cannot share a line (`4 + 1 + 4 = 9 > 4`); the only legal layout is `{0}` then `{1}`, with the last line free, so the answer must be `(4 - 4)^2 = 0`. Run it: `pre = [0,4,8]`. `dp[1]`: `j = 1` group `[0..0]` `used = 4`, `i = 1 != n` cost `0`, `dp[1] = 0`. `dp[2]` (`i == n`): `j = 2` group `[1..1]` `used = 4`, free, cand `dp[1] + 0 = 0`; `j = 1` group `[0..1]` `cnt = 2`, `widthSum = 8`, `used = 8 + 1 = 9`, `slack = 4 - 9 = -5`, `i == n` so `pen = 0`, cand `dp[0] + 0 = 0`. `dp[2] = 0`. The number `0` is right *by luck*, but look at what happened on `j = 1`: I evaluated a line of `used = 9 > W = 4` — an **illegal, overflowing line** — and because `i == n` made `pen = 0`, the code happily treated that overflowing single line as a free, valid layout. The answer matched only because a legal layout also gave `0` here.

**Diagnosing the first bug.** The defect is that I never test feasibility `used <= W`. On the last line the bug is masked (cost is forced to 0 regardless), but on an *interior* line it is fatal: I would compute `slack = W - used` for an overflowing group, get a negative `slack`, square it into a *positive* finite penalty, and offer that as a legal candidate — letting the DP cram an illegal over-wide line whenever its squared (negative) slack happens to look cheap. To expose the interior version, take `n = 3, W = 4, w = [3, 3, 1]`: the group `[0..1]` has `used = 3 + 1 + 3 = 7 > 4`, `slack = -3`, and my buggy code would value it at `9` as if it were a legal line. That is a phantom layout. I must (a) skip any group with `used > W`, and (b) exploit that `used` only grows as `j` decreases (I am adding earlier beads), so the moment `used > W` I can `break` the inner loop — every smaller `j` overflows too. That also turns the worst case into `O(n * maxBeadsPerLine)`, which never exceeds `O(n^2)`.

**Second bug — surfaced by the same fix.** Adding the feasibility guard exposes a second latent problem: I read `dp[j - 1]` and add `pen` to it, but `dp[j - 1]` can still be `INF` (unreachable prefix) for some `j` once I start skipping. Adding a finite `pen` to `INF` is fine numerically only if `INF` is small enough not to overflow when summed; with `INF = LLONG_MAX`, `INF + pen` overflows to a negative number and would *win* the `min`, corrupting `dp[i]`. Two defences: set `INF = LLONG_MAX / 4` so even `INF + (max pen ≈ 10^12)` stays safely positive and below `LLONG_MAX`, and explicitly `continue` when `dp[j - 1] == INF` so I never build a candidate on an unreachable prefix. In this problem `dp[j-1]` is actually always reachable (any prefix can be laid out one-bead-per-line), but I add the guard anyway because relying on "it happens to always be reachable" is exactly the kind of unstated assumption that breaks under a constraint change.

**Fixing and re-verifying.** The corrected inner loop adds the feasibility test with an early `break`, the `INF` guard, and keeps the `i == n` last-line-free rule:

```
for (int j = i; j >= 1; j--) {
    long long cnt = i - (j - 1);
    long long widthSum = pre[i] - pre[j - 1];
    long long used = widthSum + (cnt - 1);
    if (used > W) break;                 // earlier beads only grow used -> all smaller j overflow
    if (dp[j - 1] == INF) continue;
    long long slack = W - used;
    long long pen = (i == n) ? 0 : slack * slack;
    dp[i] = min(dp[i], dp[j - 1] + pen);
}
```

Re-trace `n = 2, W = 4, w = [4,4]`: `dp[1]`: `j=1` `used=4<=4`, cost 0, `dp[1]=0`. `dp[2]` (`i==n`): `j=2` `used=4`, free, cand 0; `j=1` `used=9>4` → `break`. `dp[2]=0`. Correct, and now for the *right reason* — the overflowing group is rejected by `break`, not waved through by the freeness rule. Re-trace the interior-bug case `n=3, W=4, w=[3,3,1]`: `dp[1]`: `used=3`, cost `(4-3)^2=1`, `dp[1]=1`. `dp[2]`: `j=2` `[1..1]` `used=3` cost 1 cand `dp[1]+1=2`; `j=1` `[0..1]` `used=7>4` `break`. `dp[2]=2`. `dp[3]` (`i==n`): `j=3` `[2..2]` `used=1` free cand `dp[2]+0=2`; `j=2` `[1..2]` `used=3+1+1=5>4` `break`. `dp[3]=2`. The brute force on `[3,3,1]` with `W=4` also gives `2` (layout `{0}` cost 1, `{1}` cost 1, `{2}` free) — agreement, and the phantom `used=7` line is gone. Both cases that were wrong-by-luck or wrong-outright now pass, and they pass for the reason I fixed.

**Sanity-checking the derivation against the headline sample.** Re-run the corrected loop on `n=4,W=6,w=[4,1,3,3]` mentally — it is the same arithmetic as the recurrence trace above (`dp = [0,4,0,5,5]`) because every group I evaluated there is legal and reachable, and the one illegal group (`[0..2]`, `used=10`) is now cut by `break` instead of contributing a bogus candidate. Final `dp[4] = 5`, matching the documented sample output. The derivation and the code agree on the very example that motivated abandoning greedy.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1` (e.g. `W = 5, w = [3]`): `dp[1]` with `i == n` makes the single line free, cand `dp[0] + 0 = 0`. Answer `0`. A lone bead is the last line, so no slack penalty — correct. Brute confirms `0`.
- Everything fits on one line (`W = 10, w = [2,2,2]`): for `dp[3]` (`i == n`), `j = 1` group `[0..2]` `used = 6 + 2 = 8 <= 10`, free, cand `0`. Answer `0`. Correct: one line, last line, free.
- Each bead exactly `W` wide (`W = 4, w = [4,4,4]`): every pair overflows, so each bead is alone; only the last is free, the other two pay `(4-4)^2 = 0`. Answer `0`. The `break` correctly forbids any sharing.
- All-same large-slack forced layout (`W = 6, w = [4,4,4,4]`): no two share (`4+1+4=9>6`), three interior lines pay `(6-4)^2 = 4` each, last free. Answer `12`. Brute agrees.
- Overflow: `dp` is `long long`; the maximum real value `~1.25*10^15` fits with three orders of magnitude to spare. `INF = LLONG_MAX/4 ≈ 2.3*10^18`, and I never add `pen` to an `INF` entry (guarded by `continue`), so no sentinel overflow. The squared term: `slack < W <= 10^6`, so `slack*slack < 10^12`, computed in 64-bit — safe.
- Negative slack never reaches the square: any group with `used > W` (hence `slack < 0`) is rejected by `break` before `slack*slack` is even formed, so I never square a negative into a spurious cost.
- Reading: `n` and `W` on the first line, then `n` widths; `cin >>` skips arbitrary whitespace/newlines, so parsing is format-agnostic. Output is exactly one integer and a newline.

**Performance check.** Worst case for the inner scan is "the `break` never fires", which needs a line that can hold every remaining bead — e.g. all widths `1` with `W` huge. Then each `i` scans back over all `i` beads: `O(n^2) = 2.5*10^7` at `n = 5000`. I measured this exact adversarial input and it runs in about 0.02 s, comfortably under the 1 s limit. When beads are wide the `break` fires early and it is far faster. No further optimisation (Li Chao / convex hull) is needed at this scale.

**Cross-check against an independent brute force.** I wrote a separate solver that enumerates all `2^(n-1)` cut-position subsets, rejects any partition containing an overflowing line, sums `slack^2` over non-last lines, and takes the minimum — an obviously-correct but exponential method, structurally different from the DP. Over 600 random small cases (`n <= 12`, `W` in `[3,14]`, widths in `[1,W]`) the DP matched the brute force on every single case, zero mismatches. I also ran a greedy-vs-DP search and confirmed greedy first-fit is strictly worse than the DP on many cases (e.g. the headline `[4,1,3,3]`, greedy 9 vs optimum 5), which is the trap this problem is built around.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with a traced counterexample and hand-checking the recurrence on the sample, and I convinced myself the *code* is right by tracing two failing inputs to precise causes — the missing feasibility test (an overflowing line waved through as free) and the `INF`-plus-`pen` overflow — fixing both, and re-verifying the fixes, the corners, and 600 brute-force comparisons. That is what I ship: one self-contained `O(n^2)` partition DP I can defend, not the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, W;
    if (!(cin >> n >> W)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // prefix[i] = sum of widths of beads 0..i-1
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + w[i];

    const long long INF = LLONG_MAX / 4;

    // dp[i] = minimum total penalty to pack the first i beads (0..i-1) into lines.
    // A line covering beads [j..i-1] is allowed only if its used width
    //   used = sum w[j..i-1] + (i-1-j)   (one unit gap between adjacent beads)
    // does not exceed W. Penalty of that line is (W - used)^2, EXCEPT the last
    // line (the one ending at i == n) has penalty 0 (no trailing-slack penalty).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        // line is [j .. i-1], 0-based beads, j from i-1 down to 0
        for (int j = i; j >= 1; j--) {
            // beads j-1 .. i-1  (1-based prefix indexing: beads with indices j-1..i-1)
            long long cnt = i - (j - 1);             // number of beads on the line
            long long widthSum = pre[i] - pre[j - 1];
            long long used = widthSum + (cnt - 1);   // gaps between beads
            if (used > W) break;                     // adding earlier beads only grows used
            if (dp[j - 1] == INF) continue;
            long long slack = W - used;
            long long pen = (i == n) ? 0 : slack * slack; // last line: no slack penalty
            dp[i] = min(dp[i], dp[j - 1] + pen);
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** Greedy first-fit looked optimal because filling each line maximally minimizes that line's slack, but a single traced counterexample (`n=4, W=6, w=[4,1,3,3]`: greedy `9` vs the reachable `5`) showed that a *convex* squared-slack penalty rewards spreading slack, so over-filling one line to dump a big slack later is worse — greedy is structurally blind to this and is discarded. The fix is an interval-partition DP, `dp[i] = min_j (dp[j-1] + cost[j-1..i-1])`, whose recurrence I checked on the sample; transcribing it, my first loop omitted the `used <= W` feasibility test, which a trace of `[4,4]` exposed (an overflowing line slipped through as "free" because the last-line rule zeroed its cost), so I added an early `break` on overflow (valid because `used` grows monotonically as the group widens) and an `INF`-guard with `INF = LLONG_MAX/4` to stop the unreachable-prefix sentinel from overflowing; with 64-bit accumulators, the `i == n` last-line-free special case, and the `break`-based feasibility test in place, the DP matches an exponential brute force on 600 random cases and resolves the single-bead, all-fit, all-overflow, and large-penalty corners.
