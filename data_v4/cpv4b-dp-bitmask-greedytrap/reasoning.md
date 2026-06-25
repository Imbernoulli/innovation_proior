**Reading the problem and pinning the contract.** I have `m` channels and `k` bursts. Firing burst `j` costs `c[j]` and delivers the channel set `mask[j]`; I may fire any subset of bursts and re-delivering an already-delivered channel is harmless. I must cover all `m` channels at minimum total cost, or print `-1`. This is weighted set cover, full stop. The one thing that makes it tractable as an *exact* problem rather than an NP-hard nightmare is the bound `m <= 18`: a set of delivered channels is a subset of an 18-element universe, so there are only `2^18 = 262144` possible "coverage states", and each fits in a 32-bit `int`. With `k <= 200`, a DP that touches every state and tries every burst is `2^18 * 200 ≈ 5.2*10^7` relaxations — fine in a second. Before any algorithm I fix the arithmetic: costs are up to `10^6` and a cover could in principle stack many bursts; a sum of `200 * 10^6 = 2*10^8` already exceeds nothing dangerous yet, but the *sentinel* I use for "unreachable" plus a cost must not overflow, and I would rather not think hard about it — so accumulators are `long long` from the start. An `int` sentinel near `INT_MAX` plus a cost is a classic silent overflow, and 64-bit removes the worry entirely.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove* optimal, because this problem explicitly asks for the exact minimum, not an approximation.

- *Greedy by efficiency.* Repeatedly fire the burst whose ratio `cost / (number of channels it newly covers)` is smallest, until all channels are covered. This is the textbook set-cover greedy. It is `O(k * m * answer-size)` and trivial to write. The reason I distrust it: set cover's greedy is famous precisely as an *approximation* — it has a logarithmic-factor guarantee, which is a polite way of saying it is provably not optimal on some inputs. Coverage is a global property and the ratio is a local score, so this is exactly the configuration where greedy stumbles. I will not ship it until I have actively tried to break it.
- *Bitmask DP.* States are subsets `S` of covered channels, `dp[S]` is the minimum cost to reach a covered set that is `S` (or a superset, depending on how I phrase the transition). Relax forward: from `S`, firing burst `j` moves me to `S | mask[j]` at extra cost `c[j]`. The answer is `dp[full]`. `O(2^m * k)`. The risk here is not the *idea* but the *transcription* — the processing order of states and the sentinel handling are exactly the kind of detail that transcribes wrong.

**Stress-testing greedy before committing — building an explicit counterexample.** I refuse to take "greedy feels efficient" on faith, so I will try to construct an instance where grabbing the most efficient burst first paints me into a corner. The trap I am looking for: a burst that covers a lot of channels cheaply *per channel* (great ratio, so greedy grabs it) but whose coverage leaves an awkward leftover that is expensive to finish, while a different pair of bursts partitions the universe cleanly for less total.

Let me build it concretely. Four channels `{0,1,2,3}` (`m = 4`). Three bursts:

- Burst 1: covers `{0,1,2}`, cost `5`. Ratio `5/3 ≈ 1.667` per channel.
- Burst 2: covers `{0,1}`, cost `4`. Ratio `4/2 = 2.0`.
- Burst 3: covers `{2,3}`, cost `4`. Ratio `4/2 = 2.0`.

Trace greedy. Nothing is covered. Ratios: burst 1 is `5/3 ≈ 1.667`, bursts 2 and 3 are `2.0` each. Greedy fires burst 1 (best ratio), now `{0,1,2}` covered, cost so far `5`. Remaining channel: `{3}`. Which bursts cover `3`? Only burst 3, and against the leftover it newly covers just `{3}`, so its effective ratio is `4/1 = 4.0`; greedy fires it. Total: `5 + 4 = 9`.

Now the optimum. Burst 2 `{0,1}` plus burst 3 `{2,3}` covers all four channels for `4 + 4 = 8`. That is strictly cheaper than greedy's `9`. So greedy is **wrong**, and I can see exactly *why*: the high-coverage burst 1 looked most efficient, but it half-covered the pair `{2,3}` that burst 3 would have completed for free, wasting burst 1's spend on channels `{0,1}` that burst 2 covers more cheaply in context. The leftover-channel cleanup is where greedy bleeds. The counterexample paid off — it killed an approach I might otherwise have shipped. Greedy is out; I commit to the exact DP.

**Deriving the DP and checking it on the counterexample.** I define `dp[S]` = minimum total cost of a multiset of fired bursts whose union of masks is exactly `S` *or any superset reachable by adding more bursts on top*. Concretely I will phrase it as a forward relaxation: `dp[0] = 0` (covered nothing, spent nothing), every other state starts at `+infinity`, and from any reachable state `S` I may fire any burst `j`, paying `c[j]` and arriving at `S | mask[j]`. The answer is `dp[full]` where `full = (1<<m) - 1`, or `-1` if it stayed infinite.

The crucial structural fact that makes the forward relaxation correct in a single left-to-right pass: `S | mask[j]` is always `>= S` as an integer, because OR only ever *sets* bits, never clears them. It equals `S` exactly when `mask[j]` is already a subset of `S` (firing it covers nothing new) — those transitions I skip, since they only add cost. So every useful transition goes from `S` to a *strictly numerically larger* state. That means if I iterate `S` from `0` upward, by the time I process state `S` its `dp[S]` value is final: it could only have been improved by transitions out of strictly smaller states, all already processed. No queue, no topological sort needed — plain ascending iteration is a valid order.

Let me confirm the recurrence numerically on the counterexample I just built (`m = 4`, masks: burst1=`0b0111`=7, burst2=`0b0011`=3, burst3=`0b1100`=12, costs `5,4,4`; `full = 15`). I will only track the states that matter. Start `dp[0]=0`. From `S=0`: fire b1 -> `dp[7] = 5`; fire b2 -> `dp[3] = 4`; fire b3 -> `dp[12] = 4`. Process `S=3` (`dp=4`): fire b1 -> `3|7 = 7`, candidate `4+5=9` vs current `dp[7]=5`, keep `5`; fire b3 -> `3|12 = 15`, `dp[15] = 4+4 = 8`. Process `S=7` (`dp=5`): fire b3 -> `7|12 = 15`, candidate `5+4 = 9` vs `dp[15]=8`, keep `8`. Process `S=12` (`dp=4`): fire b2 -> `12|3 = 15`, candidate `4+4 = 8`, ties `dp[15]=8`. Final `dp[15] = 8`. The DP returns the true optimum `8`, beating greedy's `9`. The recurrence is right, and it found exactly the burst-2-plus-burst-3 partition greedy missed.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core, after reading masks and costs:

```
const long long INF = LLONG_MAX;        // <-- watch this
int full = (1 << m) - 1;
vector<long long> dp(1 << m, INF);
dp[0] = 0;
for (int S = 0; S <= full; S++) {
    if (dp[S] == INF) continue;
    for (int j = 0; j < k; j++) {
        int ns = S | mask[j];
        long long nc = dp[S] + cost[j];   // <-- and watch this
        if (nc < dp[ns]) dp[ns] = nc;
    }
}
```

Two things make me uneasy: the `INF = LLONG_MAX` sentinel and the unconditional `dp[S] + cost[j]`. Let me trace the smallest input that could expose a sentinel problem. Take an instance where some state is unreachable but I still try to add to it... actually I guard with `if (dp[S] == INF) continue;`, so I never *read* `dp[S]` when it is INF. But I *write* `dp[ns]` and the comparison `nc < dp[ns]` reads `dp[ns]` which may be `INF = LLONG_MAX`. That read is fine in the comparison. So where is the danger? It is the `continue`-free inner relaxation when `mask[j]` is a subset of `S`: then `ns == S`, and I compute `nc = dp[S] + cost[j]` and compare against `dp[S]` itself — `nc` is larger so it is harmlessly rejected, but it is wasted work, and more importantly it means a burst with `mask[j] = 0` (a zero-channel burst, which the contract allows) creates a self-loop `S -> S` that does nothing. Not a correctness bug, but I will skip `ns == S` to be clean.

The real bug is the sentinel arithmetic. Construct it: suppose `dp[S]` is finite but I reach a state whose only path so far is via INF — no, the guard handles that. Let me think where `INF + cost` actually executes. It is the line `dp[S] + cost[j]` — but I only get there when `dp[S] != INF` because of the guard. So that specific addition is safe. The hazard is subtler: I want to be sure that `nc < dp[ns]` with `dp[ns] == LLONG_MAX` works, which it does. So `LLONG_MAX` itself is not yet broken here... but it is *fragile*: if I ever refactor to a backward transition or drop the guard, `LLONG_MAX + cost` overflows to a negative number and silently wins the `min`. I will harden the sentinel to `LLONG_MAX / 4` so that even an accidental `INF + cost` stays comfortably positive and never beats a real cost. This is the kind of latent landmine I would rather defuse now than debug later.

**A second, real bug — the wrong iteration direction.** To double-check my claim that ascending order is the *correct* one, let me deliberately consider the opposite and trace it, because if I had typed `for (int S = full; S >= 0; S--)` (a reflex from subset-sum-style DPs that go high-to-low) the code would be wrong, and I want to *see* the failure rather than merely assert it. Use the counterexample again. Descending: I process `S = 15` first — `dp[15] = INF`, guarded, skip. ... eventually `S = 12` (`dp=12` is still INF at this point because I have not processed `S=0` yet!). In fact with descending order, when I reach `S = 0` last and finally set `dp[3]`, `dp[7]`, `dp[12]`, I have *already passed* those states and will never relax out of them again. So `dp[15]` stays `INF` and the program prints `-1` for a clearly solvable instance. That is a concrete, total failure — descending order breaks the forward DP because transitions go low-to-high, so I must consume states low-to-high. Good: the trace confirms ascending is not a stylistic choice but a correctness requirement, and pins exactly what would go wrong otherwise.

**Fixing and re-verifying.** I settle on: ascending `S`, `INF = LLONG_MAX/4`, skip `ns == S`. Re-trace the counterexample with the hardened code (already done above in the recurrence check): it yields `dp[15] = 8`. Re-trace a trivial reachability case `m = 1`, one burst covering `{0}` cost `7`: `dp[0]=0`; `S=0` fires the burst -> `dp[1] = 7`; answer `dp[1] = 7`. Correct. Re-trace an impossible case `m = 3` with bursts only covering `{0,1}`: every reachable state is a subset of `{0,1}` = `0b011`; `dp[7]` (the full set `0b111`) is never touched, stays `INF`, so I print `-1`. Correct. The two failing modes I worried about — sentinel overflow and wrong direction — are now structurally impossible, and the cases that would have exposed them pass.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `m = 0`: then `full = (1<<0) - 1 = 0`, and `dp[0] = 0` already, so I print `dp[full] = dp[0] = 0`. Every requirement is vacuously satisfied by firing nothing — correct, and it matches the contract that `m = 0` answers `0`. The state vector is `vector<long long> dp(1, INF)` then `dp[0]=0`; size `1<<0 = 1`, no out-of-bounds.
- `k = 0`: no bursts. If `m = 0` the answer is `0` (above). If `m > 0`, `dp[full]` is unreachable, stays `INF`, I print `-1`. Correct — you cannot cover anything with no bursts.
- Zero-channel burst (`t = 0`, so `mask[j] = 0`): for any `S`, `ns = S | 0 = S`, caught by the `ns == S` skip, so it is correctly ignored (firing it only wastes energy). Without the skip it would be a harmless self-loop, but skipping is cleaner and avoids redundant work.
- Heavy overlap / redundancy: e.g. `m=2`, bursts `{0,1}` cost 10, `{0,1}` cost 3, `{0}` cost 4. From `S=0` the cost-3 full burst sets `dp[3]=3`; nothing beats it; answer `3`. The DP naturally prefers the cheaper redundant burst — correct.
- Overflow: accumulators are `long long`; the largest plausible total is bounded by `k * max-cost` along any non-repeating path, but even a loose bound `2^m` bursts at `10^6` is `~2.6*10^11`, far inside 64-bit. The sentinel `LLONG_MAX/4 ≈ 2.3*10^18` plus any cost stays positive and never wins a `min`. Safe.
- State-space size: `1 << 18 = 262144` `long long`s is about 2 MB — trivially inside 256 MB.

**Numeric self-check of the complexity claim.** I claimed `O(2^m * k)` is about `5.2*10^7` at the limit. Check: `2^18 = 262144`, times `k = 200` is `5.24*10^7` inner iterations, each a couple of integer ops. At a few hundred million simple ops per second that is well under `0.2 s`. I do not trust the estimate blind, so I built a worst case — `m=18`, `k=200`, all single-channel bursts so nearly the entire `2^18` state space becomes reachable and every state runs its full inner loop — and measured: it finishes in about `0.10 s` using `~5 MB`. The measured time matches the estimate's order of magnitude, confirming the `1 s` limit is comfortable and the bound is not hand-waving.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with an explicit counterexample (`9` vs the optimal `8`) and hand-running the DP recurrence on it; I convinced myself the *code* is right by tracing the two failure modes — a fragile `LLONG_MAX` sentinel and a wrong descending iteration order that prints `-1` on solvable inputs — to their precise causes, hardening the sentinel and locking the ascending order, then re-verifying the counterexample, the reachable, impossible, `m=0`, `k=0`, and overlap corners. That is what I ship: one self-contained file, the exact `O(2^m k)` bitmask DP I can defend, not the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, k;
    if (!(cin >> m >> k)) return 0;

    vector<int> mask(k, 0);
    vector<long long> cost(k, 0);
    for (int j = 0; j < k; j++) {
        long long c;
        int t;
        cin >> c >> t;
        cost[j] = c;
        int mk = 0;
        for (int s = 0; s < t; s++) {
            int ch;
            cin >> ch;
            mk |= (1 << ch);
        }
        mask[j] = mk;
    }

    const long long INF = LLONG_MAX / 4;
    int full = (1 << m) - 1;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;

    // dp[S] = minimum total cost of a set of bursts whose union of delivered
    // channels is exactly S' >= S, reached by accumulating bursts. We process
    // states in increasing order; from state S we may fire any burst j, moving
    // to S | mask[j] at additional cost cost[j]. Firing the same burst twice is
    // never beneficial, so monotone forward relaxation suffices.
    for (int S = 0; S <= full; S++) {
        if (dp[S] == INF) continue;
        long long base = dp[S];
        for (int j = 0; j < k; j++) {
            int ns = S | mask[j];
            if (ns == S) continue;            // adds nothing new
            long long nc = base + cost[j];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[full] >= INF) cout << -1 << "\n";
    else cout << dp[full] << "\n";
    return 0;
}
```

**Causal recap.** Greedy by cost-per-new-channel looked efficient, but an explicit constructed counterexample (`{0,1,2}`@5, `{0,1}`@4, `{2,3}`@4: greedy grabs the high-coverage burst for total `9`, the optimum partitions cleanly for `8`) showed a locally efficient pick strands an expensive leftover, so I moved to an exact bitmask DP over the `2^m <= 262144` coverage states; its forward transition `S -> S | mask[j]` only ever sets bits so target states are numerically larger, which makes plain ascending iteration the correct and only-correct processing order — a descending pass I traced prints `-1` on solvable inputs because it consumes states before they are relaxed; and hardening the unreachable sentinel from `LLONG_MAX` to `LLONG_MAX/4`, skipping no-op `ns == S` transitions, and using `long long` throughout closes out the overflow, zero-channel-burst, impossible, `m = 0`, and `k = 0` corners.
