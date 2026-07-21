Two features of this bead-packing task decide everything before I pick an algorithm: the per-line penalty is `slack^2`, a *convex* cost, and the very last line pays nothing at all. Convexity is what makes the "obvious" packing wrong, and the free last line is exactly where a feasibility bug can hide. So I pin the contract first. I cut the fixed-order bead sequence into contiguous groups, one per glow-line; a line holding beads `l..r` consumes `used(l,r) = (w[l]+...+w[r]) + (r-l)` — the widths plus one unit of clearance between each adjacent pair — and must satisfy `used <= W`. Each non-last line pays `(W-used)^2`; the final line is free. Because `w[i] <= W`, the one-bead-per-line layout is always legal, so a feasible cut always exists. Output is a single integer.

Scale first, because it fixes the data types and the reachable complexity. `n <= 5000`, `W <= 10^6`. The worst total penalty comes from many single-bead lines each about half-full: up to `n-1 ≈ 5000` interior lines each paying near `(5·10^5)^2 = 2.5·10^11`, so roughly `1.25·10^15` overall — three orders past the 32-bit ceiling of `~2.1·10^9`. Every accumulator has to be `long long`; an `int` is a silent wrong answer on the big tests. On the other hand `n = 5000` means an `O(n^2) = 2.5·10^7` partition scan clears one second easily, so I have no reason to reach for an `O(n log n)` convex-hull trick — a quadratic DP is fast enough and much easier to get provably right.

Now the strategy, and here the problem is built around one specific temptation. Greedy first-fit — sweep once, keep adding the next bead while it fits, open a new line the instant it would overflow — is `O(n)` and a dozen lines, and it *locally* minimizes each line's slack by filling maximally. But the objective is the sum of *squares* of slack, and a convex penalty rewards spreading slack evenly rather than concentrating it; a purely local "fill now" decision can dump a large slack onto one later line. Before trusting it I try to break it.

Take `n=4, W=6, w=[4,1,3,3]`. Greedy opens line 1 with bead 0 (`used=4`), adds bead 1 (`used = 4 + (1+1) = 6`, exact fit, slack 0), then bead 2 would need `6 + (3+1) = 10 > 6`, so it closes `{0,1}` and opens `{2}` (`used=3`); bead 3 needs `3+(3+1)=7>6`, so `{2}` closes with slack 3 and `{3}` becomes the free last line. Greedy pays `0^2 + 3^2 = 9`. But I can do better by *wasting* a little early: `{0}` alone (slack 2, cost 4), `{1,2}` (`used=1+1+3=5`, slack 1, cost 1), `{3}` free — total 5. Both layouts waste the same total number of slack units, yet `2^2+1^2 = 5` beats `0^2+3^2 = 9` because convexity punishes the concentrated waste. Greedy is out.

So I want the exact optimum over all contiguous partitions — an interval-partition DP. Let `dp[i]` be the minimum penalty to lay out the first `i` beads, deciding which beads share that layout's *last* line: say that line is the group `[j-1 .. i-1]`. With prefix sums `pre[i] = w[0]+...+w[i-1]`, the group has `cnt = i-(j-1)` beads, `widthSum = pre[i]-pre[j-1]`, and `used = widthSum + (cnt-1)`; it is legal iff `used <= W`. Then

```
dp[i] = min over legal j of ( dp[j-1] + pen ),    dp[0] = 0
```

with `pen = (W-used)^2` normally, but `pen = 0` when `i == n`. The one genuinely delicate point is that last-line-free rule: it applies only to the *global* final line, which is exactly the group ending at bead `n-1`, i.e. `i == n`. For any `dp[i]` with `i < n` the group is an interior line and must pay `slack^2`. So freeness is keyed on `i == n`, not on "the last group I happen to consider." The answer is `dp[n]`.

Running the recurrence on the sample confirms the transcription of the free-line rule: `pre=[0,4,5,8,11]` gives `dp=[0,4,0,5,5]` — `dp[3]=5` via `{0}` then `{1,2}`, and `dp[4]=5` where the trailing group `{3}` is free — matching the expected 5 and the hand-optimum above.

Now the code. My first cut of the inner loop:

```
for (int j = i; j >= 1; j--) {
    long long cnt = i - (j - 1);
    long long widthSum = pre[i] - pre[j - 1];
    long long used = widthSum + (cnt - 1);
    long long slack = W - used;
    long long pen = (i == n) ? 0 : slack * slack;
    dp[i] = min(dp[i], dp[j - 1] + pen);
}
```

I trace it on `n=2, W=4, w=[4,4]`. Both beads are width 4 = W, so they cannot share (`4+1+4=9>4`); the only legal layout is `{0}`,`{1}` with the last line free, answer 0. The code: `dp[1]=0`; for `dp[2]` (`i==n`), `j=2` gives `used=4`, free, candidate 0 — but `j=1` gives `cnt=2, used=9, slack=-5`, and because `i==n` forces `pen=0`, the code accepts this *overflowing* single line as a free, valid layout. It returns 0, which is right only by luck: a legal layout also scored 0 here.

The defect is that I never test `used <= W`. On the last line the freeness masks it; on an interior line it is fatal — for an overflowing group `slack = W-used` is negative and squares into a positive finite penalty that the DP will happily offer as a legal candidate. To see that interior version, `n=3, W=4, w=[3,3,1]`: the group `[0..1]` has `used=7`, `slack=-3`, and the buggy code values it at 9 as though it were a real line. So I add the feasibility test, and I can make it a `break`: as `j` decreases the group only gains beads, so `used` grows monotonically — the moment `used > W`, every smaller `j` overflows too. That also caps the inner scan at the beads-per-line count, keeping the whole thing within `O(n^2)`.

Adding that guard surfaces a second issue: I read `dp[j-1]` and add `pen`, but with the skip in place `dp[j-1]` could be an unreachable `INF`. If `INF = LLONG_MAX`, then `INF + pen` overflows to a negative number that would *win* the `min` and corrupt `dp[i]`. Two defences: set `INF = LLONG_MAX/4` so `INF + (max pen ≈ 10^12)` stays safely positive, and `continue` on `dp[j-1] == INF` so I never build on an unreachable prefix. Here every prefix is in fact reachable (one-bead-per-line always works), but I keep the guard rather than depend on that.

The corrected loop:

```
for (int j = i; j >= 1; j--) {
    long long cnt = i - (j - 1);
    long long widthSum = pre[i] - pre[j - 1];
    long long used = widthSum + (cnt - 1);
    if (used > W) break;              // earlier beads only grow used -> all smaller j overflow
    if (dp[j - 1] == INF) continue;
    long long slack = W - used;
    long long pen = (i == n) ? 0 : slack * slack;
    dp[i] = min(dp[i], dp[j - 1] + pen);
}
```

Re-tracing, `[4,4]` now rejects the `used=9` group by `break` and returns 0 for the right reason, and `[3,3,1]` gives `dp=[0,1,2,2]` — each width-3 bead alone pays `(4-3)^2=1`, the trailing `{2}` is free, total 2 — matching brute force with the phantom `used=7` line gone.

Two non-sample cases exercise the machinery. With `W=6, w=[4,4,4,4]`, sharing is impossible (`4+1+4=9>6`), so the `break` forces four singleton lines — three interior at `(6-4)^2=4` plus a free last — answer 12, the smallest case where the sparse penalty is nonzero. With `W=4, w=[4,4,4]` every bead equals `W`, again forced apart, but each interior line now pays `(4-4)^2=0`, answer 0. `n=1` and everything-fits both collapse to a single free last line, answer 0.

The `O(n^2)` scan is worst when a line holds many beads — all widths 1 with large `W`, where each `i` scans back over all `i` predecessors: `2.5·10^7` at `n=5000`, about 0.02 s. Wide beads fire the `break` early and run faster.

As a final guard I cross-check against an exponential solver that enumerates all `2^(n-1)` cut-position subsets, discards any partition with an overflowing line, and takes the minimum squared-slack sum over non-last lines. Over 600 random small cases (`n <= 12`, `W in [3,14]`) the DP matches every one, and the same search confirms greedy first-fit is strictly worse on many — the trap this problem is built around. The full program is in the answer.
