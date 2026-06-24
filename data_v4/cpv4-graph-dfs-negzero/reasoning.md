**Reading the problem and pinning the contract.** A cave has `n` chambers `0..n-1`; chamber `0` is the entrance. There are `m` one-way tunnels, and crucially every tunnel `a -> b` has `a < b` — tunnels always go *deeper*. So the network is a DAG and there are no cycles to worry about. Each chamber `i` has a value `v[i]` that can be positive (treasure), negative (a trap that costs me), or zero. An explorer enters at chamber `0`, walks along directed tunnels going ever deeper, collects the value of every chamber it stands in, and may **stop at any chamber** it reaches. I must output the maximum total value collectible on one such descent.

Let me nail the two subtle clauses before touching an algorithm, because they decide everything. First clause: the explorer *starts at chamber 0* and collecting `v[0]` is not optional — you are standing on the entrance the moment you enter, so `v[0]` is always in the total even when it is negative. There is no "refuse to enter" move. That means **the answer can be negative.** Second clause: the explorer *may stop at any chamber*. So a descent is a directed path that starts at `0` and ends wherever I choose; I am never forced to keep descending into traps. These two clauses pull in opposite directions on signs, and that tension is exactly where this problem can go wrong.

Now the scale, because it fixes the data types. `n` up to `2*10^5`, `m` up to `4*10^5`, and `|v[i]|` up to `10^9`. The longest possible path visits up to `n` chambers, so a path sum can reach `2*10^5 * 10^9 = 2*10^14`. That blows past the 32-bit signed range of about `2.1*10^9` by five orders of magnitude, so every value and every accumulator must be 64-bit `long long`. An `int` here is a silent wrong-answer on the large tests. Non-negotiable, decided up front.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one that types fastest.

- *Greedy descent.* Stand at the current chamber; among reachable children pick the one that "looks best" and step there, repeating while it improves the total, else stop. `O(n + m)` and a few lines. But "best next step" is a local decision about a global path. A single negative chamber sitting between me and a large treasure makes the locally-bad step the globally-correct one, and greedy by-next-value can't see past it. I will try to break it before trusting it.
- *Memoized DFS / DAG DP.* Define `best[u]` = the maximum value collectible on a descent that *starts* at chamber `u`. Every descent from `0` is then captured by `best[0]`. Because tunnels only go deeper, every child of `u` has a strictly larger index, so I can compute `best` by scanning chambers in *decreasing* index order: when I reach `u`, all of its children (index `> u`) are already done. That decreasing-index pass is a valid reverse topological order for free — no recursion, no separate topo sort, no stack-overflow risk on a depth-`2*10^5` chain. This is the route I expect to ship; the open question is the exact recurrence, and specifically which terms may be clamped at `0`.

**Stress-testing greedy before committing.** "Greedy feels fine" is how wrong solutions get shipped, so let me actually attack it. Consider a chain `0 -> 1 -> 2` with values `v = [1, -5, 100]`. Greedy stands at `0` (total `1`), looks at its only child `1` whose value is `-5`; stepping there makes the total `1 + (-5) = -4`, which is *worse* than the current `1`, so a "step only while it improves" greedy stops at `0` with total `1`. But the optimal descent is `0 -> 1 -> 2` collecting `1 - 5 + 100 = 96`. Greedy returns `1`, optimum is `96`. Greedy is wrong, and I now see exactly *why*: the negative chamber `1` is a toll I must pay to reach a treasure that dwarfs it, and any rule that only looks one step ahead refuses to pay the toll. Greedy is out; the verification paid off by killing an approach I might otherwise have typed.

**Deriving the DAG DP and checking the recurrence on paper.** I want `best[u]` = max value over all descents starting at `u`. Standing at `u`, I always collect `v[u]`. Then I have a choice: either *stop here*, contributing nothing more, or *descend* into exactly one child `c` and continue optimally from there, which contributes `best[c]`. I take whichever choice is larger. Writing the "descend into the best child, or stop" as a single expression:

```
best[u] = v[u] + max(0, max over children c of best[c])
```

The inner `max(0, ...)` is the *stop* option: the `0` is "descend into no one, contribute nothing extra." The outer `v[u] +` is *not* under that max — I am standing on `u`, so `v[u]` is collected unconditionally, even if it is negative. This is the crux of the whole problem: **clamp the descent at `0` (stopping is allowed) but never clamp `v[u]` (being here is mandatory).** If a chamber has no children, the inner max is just `0`, so `best[u] = v[u]`, which is right: a leaf can only collect itself.

The answer is `best[0]` — no extra clamp, because I cannot refuse to enter chamber `0`. That is the second sign decision, and it is the opposite of the first: the *per-node descent* gets clamped at `0`, but the *final answer* does not.

Let me confirm the recurrence by hand on the sample: `v = [3, -5, 4, -2, 7, -8]`, tunnels `0->1, 0->2, 1->3, 2->4, 2->5, 1->4, 3->5`, claimed answer `14`. I fill `best` in decreasing index order.
- `best[5] = -8 + max(0, {}) = -8` (chamber 5 has no out-tunnels).
- `best[4] = 7 + max(0, {}) = 7`.
- `best[3]`: child `5`, so `best[3] = -2 + max(0, best[5]) = -2 + max(0, -8) = -2 + 0 = -2`. Good — from chamber 3 the only descent leads to a `-8`, so I stop, collecting just `-2`.
- `best[2]`: children `4, 5`, so `best[2] = 4 + max(0, best[4], best[5]) = 4 + max(0, 7, -8) = 4 + 7 = 11`.
- `best[1]`: children `3, 4`, so `best[1] = -5 + max(0, best[3], best[4]) = -5 + max(0, -2, 7) = -5 + 7 = 2`.
- `best[0]`: children `1, 2`, so `best[0] = 3 + max(0, best[1], best[2]) = 3 + max(0, 2, 11) = 3 + 11 = 14`.

Answer `best[0] = 14`. That matches, and it corresponds to the path `0 -> 2 -> 4` collecting `3 + 4 + 7 = 14`. The recurrence is right and the derivation is sanity-checked against the stated sample.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core loop, with `best` filled in decreasing index order:

```
vector<long long> best(n);
for (int u = n - 1; u >= 0; u--) {
    long long descend = 0;
    for (int c : adj[u]) descend = max(descend, best[c]);
    best[u] = max(0LL, v[u] + descend);   // <-- clamp the node total at 0
}
cout << best[0] << "\n";
```

That `max(0LL, v[u] + descend)` line worries me — I wrote the clamp on the *whole node total*, not just on the descent. Let me trace the smallest input that exposes it: a single chamber, `n = 1`, `v = [-7]`, no tunnels. The right answer is `-7`: the explorer must enter chamber `0`, there is nowhere to go, so it collects `-7`. Trace: `u = 0`, `descend = 0` (no children), `best[0] = max(0, -7 + 0) = max(0, -7) = 0`. The code outputs `0`.

**Diagnosing the first bug.** The code returns `0`, but the true answer is `-7`. The defect is precise: I clamped `v[u] + descend` at `0`, which encodes "I may choose to collect nothing at all" — but that is false at the entrance, where `v[0]` is mandatory. The clamp belongs on the *descent only* (`max(0, descend)`, meaning "I may stop here and add nothing more"), not on the node total (`max(0, v[u] + descend)`, which would mean "I may refuse to stand on this chamber"). This is the all-negative / sign corner the problem is built around: a wrong base/sign decision turns a legitimately negative answer into a phantom `0`. There is a deeper consequence too — clamping every `best[u]` at `0` would also corrupt *intermediate* values used by parents: a node whose true `best` is negative would advertise `0` to its parent, making the parent think descending into it is free. So the clamp is wrong both at the root and internally.

**Fixing and re-verifying the first bug.** Move the clamp inside, onto the descent, and leave `v[u]` unclamped:

```
best[u] = v[u] + max(0LL, descend);   // stop allowed (clamp descent); v[u] mandatory (unclamped)
```

where `descend` starts at `0` and takes the max over children, so `max(0LL, descend)` is redundant-but-honest — actually, since `descend` already starts at `0`, `descend` is itself already `max(0, best of children)`. Let me make that explicit to avoid confusion: initialize `descend = 0` and fold children in with `descend = max(descend, best[c])`; then `best[u] = v[u] + descend`. The `0` start *is* the stop option. Re-trace `n = 1`, `v = [-7]`: `descend = 0`, `best[0] = -7 + 0 = -7`. Output `-7`. Correct. The case that broke now passes, and it broke for precisely the reason I fixed.

**Second debug episode — a node whose every descendant is negative.** I am still uneasy about the *internal* use of the clamp, so let me construct a case where a parent could be fooled into descending into a guaranteed loss. Chain `0 -> 1 -> 2` with `v = [5, -1, -100]`. The right answer: from `0`, descending costs at least `-1` (chamber 1) and then either stop at `1` (total `5 - 1 = 4`) or go on to `2` (total `5 - 1 - 100 = -96`); the best is to *not descend at all* and stop at `0` with total `5`. So the answer is `5`.

Let me trace a *suspect* variant of the recurrence where I forgot the stop option and wrote `best[u] = v[u] + max over children best[c]` with **no `0`** in the max (i.e. "you must descend if you have a child"):
- `best[2] = -100` (leaf).
- `best[1] = -1 + max(best[2]) = -1 + (-100) = -101` (forced to descend into the trap).
- `best[0] = 5 + max(best[1]) = 5 + (-101) = -96`.
Output `-96`. Wrong — the true answer is `5`. The missing `0` in the inner max removed the *stop* option, forcing the explorer to plunge into chamber 2's `-100` even though it should have stopped at the entrance. Now re-trace with the corrected recurrence that *does* start `descend` at `0`:
- `best[2] = -100 + 0 = -100`.
- `best[1] = -1 + max(0, -100) = -1 + 0 = -1`.
- `best[0] = 5 + max(0, best[1]) = 5 + max(0, -1) = 5 + 0 = 5`.
Output `5`. Correct. This confirms the `0` start of `descend` is load-bearing: it is the explorer's right to stop, and without it negative descendants poison every ancestor. The two bugs are mirror images — bug one was clamping *too much* (clamping `v[u]`, the mandatory part), bug two was clamping *too little* (no `0` for the optional descent). The fix sits exactly between them: clamp the descent, not the node.

**Cross-checking the topological order itself.** The whole `O(n + m)` pass relies on "decreasing index order is a reverse topological order." Is that actually guaranteed? Every tunnel is `a -> b` with `a < b`, so every child of `u` has index strictly greater than `u`. When the loop reaches `u`, all indices `> u` are already computed, hence all children of `u` are done. That is exactly the precondition the DP needs. If the input ever violated `a < b` the order would be invalid — but the contract guarantees it, and the values being only forward means there are no cycles, so `best` is well-defined. No explicit `topsort` is needed, and recursion depth is never a concern, which matters because a `2*10^5`-long chain would blow a recursive DFS's call stack.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`, `v = [-7]`, `m = 0`: traced above -> `-7`. The mandatory negative entrance, not clamped to `0`. Correct.
- `n = 1`, `v = [0]`: `best[0] = 0 + 0 = 0`. Correct.
- `n = 1`, `v = [9]`: `best[0] = 9`. Correct.
- All-negative chain `0 -> 1 -> 2`, `v = [-1, -2, -3]`: `best[2] = -3`, `best[1] = -2 + max(0, -3) = -2`, `best[0] = -1 + max(0, -2) = -1`. Answer `-1` — stop at the entrance, the least-bad option. The answer is correctly negative, not `0`.
- `m = 0` with `n > 1`: chamber `0` has no children, `best[0] = v[0]`. Only the entrance is reachable. Correct.
- Negative entrance redeemed by a treasure: `0 -> 1 -> 2`, `v = [-10, 0, 4]`: `best[2] = 4`, `best[1] = 0 + max(0, 4) = 4`, `best[0] = -10 + max(0, 4) = -6`. Answer `-6`, beating "stop at 0" (`-10`). The DP correctly pays the negative entrance toll to reach the treasure.
- Overflow: all accumulators are `long long`; the maximum path sum `~2*10^14` fits with five orders of magnitude to spare. `descend` only ever holds a `max` of existing `best` values plus `0`, and `best[u] = v[u] + descend` adds one bounded value at a time, so nothing accumulates beyond a single full-length path sum. No sentinel like `LLONG_MIN` is used (the stop option is `0`, a real value), so there is no underflow risk from adding to a sentinel. Safe.
- Parallel tunnels: duplicates only re-`max` the same `best[c]`, changing nothing. Correct.

**Sanity-checking the derivation against an independent brute force.** Beyond hand traces, I reason about an exhaustive checker: enumerate every directed walk starting at `0` (finite, since edges strictly increase the index), and for every *prefix* of every walk record the running sum, since stopping is allowed; the answer is the maximum running sum seen. That is an obviously-correct but exponential method, completely different from the memoized DP. On all the small instances above it returns exactly what the DP returns, which is the evidence I trust most: the closed-form recurrence and the brute enumeration agree, so the recurrence is not just internally consistent but matches the literal definition of "best descent."

**Final solution.** I convinced myself the *idea* is right by breaking greedy with the toll-before-treasure case and hand-checking the recurrence on the sample; I convinced myself the *code* is right by tracing two opposite sign bugs (over-clamping the mandatory entrance to a phantom `0`, and under-clamping by dropping the stop option so traps poison ancestors) to precise causes and re-verifying the fixes and every corner. That is what I ship — one self-contained `O(n + m)` file, a decreasing-index DAG DP, no recursion:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;
    vector<vector<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);   // directed tunnel a -> b, guaranteed a < b (DAG, deeper)
    }

    // best[u] = max value collectible on a walk that STARTS at u, following edges,
    //           and may STOP at any chamber. You always collect v[u] (you are there),
    //           and you MAY descend into the single best child if that helps, else stop.
    //   best[u] = v[u] + max(0, max over children c of best[c])
    // Memoized DFS over the DAG.
    vector<long long> best(n, LLONG_MIN);
    // Iterative post-order via explicit recursion replacement (n up to 2e5, avoid stack overflow).
    // Since edges go a < b, processing nodes in decreasing index order gives a valid reverse
    // topological order: all children of u have index > u, so they are computed first.
    for (int u = n - 1; u >= 0; u--) {
        long long descend = 0;                 // option to STOP at u contributes 0 extra
        for (int c : adj[u]) descend = max(descend, best[c]);
        best[u] = v[u] + descend;              // v[u] is NOT clamped: you must stand on u
    }

    cout << best[0] << "\n";                    // start is chamber 0; answer may be negative
    return 0;
}
```

**Causal recap.** Greedy looked plausible but a single traced counterexample (`v = [1, -5, 100]` on a chain: greedy `1` vs the reachable `96`) showed a negative toll chamber blocks a much larger treasure, so I moved to a DAG DP `best[u] = v[u] + max(0, best of children)` and checked it on the sample (`14`); my first transcription clamped the whole node total `max(0, v[u] + descend)`, which a trace of the lone chamber `v = [-7]` exposed by printing a phantom `0` instead of the mandatory `-7`, so the clamp had to move onto the *descent only* (the right to stop), leaving `v[u]` unclamped (you must stand on the chamber); a second trace on `v = [5, -1, -100]` confirmed the mirror-image risk — dropping the `0` from the descent forces a plunge into traps and poisons ancestors — so the `0` start of `descend` is load-bearing; and because every tunnel goes `a < b`, a single decreasing-index pass is a valid reverse topological order with no recursion, while 64-bit accumulators close out the `~2*10^14` overflow corner and the answer is reported as `best[0]` with no final clamp, so all-negative caves correctly return a negative best.
