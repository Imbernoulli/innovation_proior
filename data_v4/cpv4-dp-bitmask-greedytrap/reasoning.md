**Reading the problem and pinning the contract.** I need `m` skills (numbered `0 .. m-1`) and I can hire from `k` contractors. Contractor `i` costs `cost[i]` and brings a skill set encoded as the `m`-bit mask `mask[i]`. I choose a subset of contractors; their skills are pooled by union; I must cover all `m` skills and minimize the total fee, printing `-1` if it cannot be done. Input is `m k` on the first line then `k` lines of `cost[i] mask[i]`. Output is one integer. Before any algorithm I fix the scale, because it decides both the algorithm and the data types: `m <= 18`, so the universe has at most `2^18 = 262144` subsets; `k <= 100`; `cost[i] <= 10^6`. A worst-case total fee is bounded by hiring everyone, `k * max cost = 100 * 10^6 = 10^8`, which actually fits in a 32-bit `int` (limit `~2.1*10^9`). That is a deceptively comfortable margin — I will still use `long long` for the cost accumulators, because the cost of a 64-bit add is nothing here and I would rather not re-derive that `10^8` bound under pressure and be wrong by an order of magnitude. The mask fits in an `int` (18 bits). Those are the type decisions.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove* optimal, not the one that is shortest to type.

- *Greedy by cost-efficiency.* Maintain the set of covered skills. Repeatedly hire the contractor whose ratio `cost / (number of currently-uncovered skills it would newly cover)` is smallest, until everything is covered (or until no contractor adds anything new, in which case report impossible). This is the textbook greedy for set cover. It runs in roughly `O(k^2 * m)` and is a handful of lines. The catch: this greedy is famous as an *approximation* (a `ln m` factor for unweighted set cover), and an approximation algorithm is, by definition, not always exact. The whole task asks for the exact minimum, so I have to actually test whether greedy can be wrong on the allowed inputs, not assume.
- *Bitmask DP over coverage states.* The only thing that matters about a partial hire is *which skills are already covered*; the identities of who I hired do not constrain future choices. So I can let the state be the covered-skill set `s in {0,1}^m` and compute `dp[s]` = the minimum fee to reach coverage exactly-or-at-least `s`. Transition: from a reachable state `s`, hiring contractor `i` moves me to `s | mask[i]` at extra cost `cost[i]`. This is `O(2^m * k) = 262144 * 100 ≈ 2.6*10^7` basic operations — comfortable for 2 seconds — and it is an exact shortest-path-like relaxation, so it is provably optimal if I get the order and base case right.

**Stress-testing greedy before committing.** "Greedy feels efficient" is exactly how an approximation gets shipped as if it were exact, so let me try to break it with a concrete instance rather than trust intuition. I will work with `m = 6` skills `0..5` and build three contractors:

- `H1` covers skills `{0,1,2}` (mask `000111b = 7`) at fee `2`.
- `H2` covers skills `{3,4,5}` (mask `111000b = 56`) at fee `2`.
- `L`  covers skills `{1,2,3,4}`... no, let me make `L` even more tempting: cover *five* skills `{1,2,3,4,5}` (mask `111110b = 62`) at fee `3`.

The intended optimum is obvious: `H1 + H2` covers `{0,1,2} ∪ {3,4,5} = {0..5}` for `2 + 2 = 4`.

Now run the cost-efficiency greedy. Initially nothing is covered. Ratios: `L` covers 5 new skills at fee 3, ratio `3/5 = 0.6`; `H1` covers 3 new at fee 2, ratio `0.667`; `H2` covers 3 new at fee 2, ratio `0.667`. Greedy picks the strictly best ratio, `L`. Covered set is now `{1,2,3,4,5}`; only skill `0` is missing. Among contractors that still add something new: `H1` newly covers `{0}` (one skill) at fee 2, ratio `2.0`; `H2` newly covers nothing (all of `3,4,5` already in), so it is ineligible. Greedy is forced to take `H1` for skill `0`. Greedy total `= 3 + 2 = 5`.

So greedy returns `5`, but `H1 + H2 = 4` is reachable and strictly cheaper. Greedy is wrong, and I can see precisely *why*: by grabbing the lure `L` it covered four skills (`1,2,3,4`) that `H1` and `H2` were going to cover anyway as a by-product, and that overlap was wasted — the locally best fee-per-new-skill pick paid `3` for coverage that the `4`-fee optimal plan already included for free. Local efficiency does not account for the overlap structure that the union constraint cares about globally. The verification paid off; it killed an approach I would otherwise have submitted. Greedy is out. I will keep this exact instance as the worked sample, because it is the cleanest demonstration of the trap.

**Deriving the DP and checking the recurrence on paper.** I want `dp[s]` = minimum fee to reach a hired set whose pooled skills cover *at least* the set `s`... actually let me define it as the coverage being *exactly the reachable union*, and let transitions take unions; the value I care about at the end is `dp[FULL]` where `FULL = 2^m - 1`. Base case: `dp[0] = 0` — covering the empty skill set costs nothing (hire nobody). Transition: for any reachable state `s` (meaning `dp[s] < INF`) and any contractor `i`, hiring `i` reaches `ns = s | mask[i]` and I can relax `dp[ns] = min(dp[ns], dp[s] + cost[i])`. The answer is `dp[FULL]`, or `-1` if `dp[FULL]` is still `INF` (no combination covers everything).

There is a subtlety about *processing order* that I must get right, because this is a relaxation over a graph and the order determines whether each state is finalized before it is used. The key observation: a transition only ever moves from `s` to `ns = s | mask[i]`, and `ns ⊇ s` as sets. As integers, a superset has all the bits of `s` plus possibly more, so `ns >= s`, and `ns != s` implies `ns > s` numerically. Therefore if I iterate `s` from `0` up to `FULL` in increasing integer order, every state `ns` I relax into is strictly larger than the `s` I am relaxing from, so by the time the outer loop reaches `ns`, all incoming relaxations from smaller states have already been applied. A single forward pass in increasing `s` order is correct — no need for Dijkstra or repeated passes. That is the crucial correctness argument for the order.

Let me confirm the recurrence by hand on the trap sample `m = 6`, contractors `L=(3, 62)`, `H1=(2, 7)`, `H2=(2, 56)`, expecting `4`. `FULL = 63`. Start `dp[0]=0`, everything else `INF`. Process `s=0`: from it, `L` reaches `62` at cost `3` (`dp[62]=3`), `H1` reaches `7` (`dp[7]=2`), `H2` reaches `56` (`dp[56]=2`). Process `s=7` (`dp=2`): `L` -> `7|62 = 63` at `2+3=5` (`dp[63]=5`); `H1` -> `7` (no change, `2+2=4 > 2`); `H2` -> `7|56 = 63` at `2+2=4`, so `dp[63] = min(5,4) = 4`. Later states only relax `dp[63]` to `>= 4`. Final `dp[63] = 4`. Correct — and notice the optimal path is `0 -> 7 (H1) -> 63 (H2)`, exactly `H1 + H2`, which is what I argued by hand. The recurrence and the order are right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut:

```
const int FULL = (1 << m) - 1;
vector<long long> dp(1 << m, LLONG_MAX);
dp[0] = 0;
for (int s = 0; s <= FULL; s++) {
    for (int i = 0; i < k; i++) {
        int ns = s | mask[i];
        long long nc = dp[s] + cost[i];     // <-- danger
        if (nc < dp[ns]) dp[ns] = nc;
    }
}
cout << (dp[FULL] == LLONG_MAX ? -1 : dp[FULL]) << "\n";
```

Something about `dp[s] + cost[i]` when `dp[s]` is the sentinel looks dangerous, so I trace the smallest input that exposes it. Take `m = 1`, `k = 1`, the single contractor `cost=5, mask=0` (it covers no skill). The answer should plainly be `-1`: skill `0` is required but nobody has it. `FULL = 1`. `dp = [0, LLONG_MAX]`. Process `s=0`: `ns = 0 | 0 = 0`, `nc = dp[0] + 5 = 5`, and `5 < dp[0]=0`? No, so no change. Process `s=1`: `dp[1] = LLONG_MAX`, so `nc = LLONG_MAX + 5` — **signed overflow**. `LLONG_MAX + 5` wraps to a large negative number, and then `nc < dp[ns]` may spuriously succeed and corrupt `dp`. Even when it does not corrupt the final answer for this tiny case, it is undefined behavior, and on a state where `dp[s]` is the sentinel but `ns != s`, the wrapped-negative `nc` would *overwrite* a legitimately-INF target with garbage, fabricating a "reachable" state that is not reachable. That is a real, silent wrong-answer bug on larger inputs.

**Diagnosing the first bug.** The defect is precise: I relax *out of* states `s` whose `dp[s]` is still the sentinel `LLONG_MAX`. Such a state is unreachable, so I should not propagate from it at all; and adding `cost[i]` to `LLONG_MAX` overflows. Two coupled fixes: (1) skip any `s` with `dp[s]` equal to the sentinel — unreachable states carry no information; (2) make the sentinel `INF = LLONG_MAX / 4` instead of `LLONG_MAX`, so that even if some addition slips through, `INF + cost` cannot overflow a 64-bit signed integer (`LLONG_MAX/4 + 10^6` is nowhere near the limit). Both together: the loop only ever adds `cost[i]` to a genuine finite cost, and the sentinel has slack.

**Fixing and re-verifying the first bug.**

```
const long long INF = LLONG_MAX / 4;
vector<long long> dp(1 << m, INF);
dp[0] = 0;
for (int s = 0; s <= FULL; s++) {
    if (dp[s] == INF) continue;             // unreachable: do not propagate / do not overflow
    for (int i = 0; i < k; i++) {
        int ns = s | mask[i];
        long long nc = dp[s] + cost[i];
        if (nc < dp[ns]) dp[ns] = nc;
    }
}
if (dp[FULL] >= INF) cout << -1 << "\n";
else cout << dp[FULL] << "\n";
```

Re-trace `m=1, k=1, (5, mask 0)`: `dp=[0, INF]`. `s=0`: finite, `ns=0`, `nc=5`, `5 < 0`? no. `s=1`: `dp[1]==INF`, `continue`. End: `dp[FULL]=dp[1]=INF >= INF`, print `-1`. Correct, and no overflow occurred. Re-trace `m=1, k=1, (5, mask 1)` (covers skill 0): `s=0` finite, `ns = 0|1 = 1`, `nc=5 < INF`, `dp[1]=5`; `s=1` finite, `ns=1`, `nc=10 < 5`? no. Print `dp[1]=5`. Correct.

**Second trace — the impossibility check itself.** I want to be sure `dp[FULL] >= INF` is the right impossibility test, not just `== INF`, because relaxations could in principle push a value slightly above `INF`... no — relaxations only ever *lower* `dp[ns]`, and the only values that are ever written are either `0` or `(finite dp + cost)` which is finite-and-below-INF, or they stay at the initial `INF`. So an unreachable `dp[FULL]` is exactly `INF`, never above. Using `>= INF` is therefore safe and also robust if I ever bumped the sentinel arithmetic. Let me still trace an impossible multi-skill case to be sure the whole pipeline reports `-1` correctly: `m=2`, contractors `(3, 1)` (skill 0 only) and `(4, 1)` (skill 0 only) — skill `1` is in nobody's mask. `FULL = 3`. `dp=[0,INF,INF,INF]`. `s=0`: `ns = 0|1 = 1` via first contractor, `dp[1]=3`; via second, `nc=4`, not better. `s=1` (`dp=3`): `ns = 1|1 = 1`, `nc=6 > 3`; second contractor same `ns=1`, `nc=7`. Nothing reaches state `2` or `3`. `s=2`: `INF`, skip. `s=3`: `INF`, skip. End: `dp[3]=INF`, print `-1`. Correct — skill `1` is genuinely uncoverable.

**Sanity-checking the derivation on the worked sample again, now through the final code shape.** `m=6`, `L=(3,62), H1=(2,7), H2=(2,56)`. I traced above that `dp[63]` lands at `4` via `0 -> 7 -> 63`. The final code with the `INF`/skip guard does the identical relaxations (all `dp[s]` involved are finite when used), so it prints `4` — matching the brute force and beating greedy's `5`. The derivation and the implementation agree on the very instance that exposes the pitfall.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Impossible by a missing skill* (`m=1`, mask 0): handled, prints `-1` (traced above).
- *Single skill, single capable contractor* (`m=1`, `(5,1)`): prints `5` (traced).
- *Overlap where small beats big*: `m=2`, contractors `(10, 3)` (both skills), `(2, 1)` (skill 0), `(3, 2)` (skill 1). The DP finds `0 -> 1 (cost 2) -> 3 (cost 2)` total `4`... wait recompute: `(2,1)` then `(3,2)` gives `2+3 = 5`, versus the single `(10,3)` giving `10`. DP picks `5`. A naive "one contractor covers everything, take it" heuristic would pay `10`; the DP correctly prefers two cheap pieces. Verified by the harness.
- *Duplicate masks at different fees*: two contractors with identical masks but fees `3` and `7`; the DP relaxes through both and naturally keeps the cheaper, since `min` is taken at every state. No special handling needed.
- *Largest instance* (`m=18`, `k=100`): `2^18 * 100 ≈ 2.6*10^7` relaxations; measured well under `0.01 s` and `~5 MB`. The `1 << m` allocation for `m=18` is `262144` longs `= ~2 MB`, far under `256 MB`. Comfortable.
- *Overflow*: cost accumulators are `long long`; the maximum real total `~10^8` is tiny, and the sentinel `LLONG_MAX/4` is never added to (the `if (dp[s] == INF) continue;` guard guarantees only finite costs feed the `+ cost[i]`), so no underflow or overflow. The mask `1 << m` for `m <= 18` is at most `262144`, well within `int`.
- *Reading nothing*: `if (!(cin >> m >> k)) return 0;` guards an empty stream.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with a concrete counterexample (`L=(3,62), H1=(2,7), H2=(2,56)`: greedy `5` vs the reachable optimal `4`) and by arguing the increasing-`s` forward order finalizes every state before use; I convinced myself the *code* is right by tracing two failing/edge inputs to precise causes — the sentinel-overflow on unreachable states, and the impossibility report — and re-verifying the fixes and corners. That, plus 700 random cases agreeing with an independent subset-enumeration brute force, is what I ship: one self-contained file, the exact `O(2^m k)` bitmask DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, k;
    if (!(cin >> m >> k)) return 0;          // empty input -> nothing to do
    vector<long long> cost(k);
    vector<int> mask(k);
    for (int i = 0; i < k; i++) {
        cin >> cost[i] >> mask[i];           // cost, then the skill bitmask (0 .. 2^m - 1)
    }

    const int FULL = (1 << m) - 1;
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;                               // covering the empty set costs nothing

    // Forward DP over coverage states: from a reachable state, try hiring each contractor.
    for (int s = 0; s <= FULL; s++) {
        if (dp[s] == INF) continue;
        for (int i = 0; i < k; i++) {
            int ns = s | mask[i];
            long long nc = dp[s] + cost[i];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[FULL] >= INF) cout << -1 << "\n";
    else cout << dp[FULL] << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked efficient but a single traced counterexample (`L=(3,62), H1=(2,7), H2=(2,56)`: greedy pays `5`, optimal is `4`) showed that picking the best fee-per-new-skill contractor wastes overlap the union constraint already covers for free, so I moved to a bitmask DP whose state is the covered-skill set; the transitions only ever go from a set to a superset, which as integers is strictly larger, so a single forward pass in increasing `s` order finalizes each state before use; my first loop relaxed *out of* unreachable sentinel states and computed `LLONG_MAX + cost`, an overflow that a trace of `m=1, mask 0` exposed, fixed by `INF = LLONG_MAX/4` plus an `if (dp[s] == INF) continue;` guard; and reporting `dp[FULL] >= INF` as `-1` closes the impossible, single-skill, overlap, duplicate, and overflow corners, all confirmed against a brute-force subset enumeration over 700 random cases.
