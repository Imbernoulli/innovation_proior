The cave has `n` chambers `0..n-1` with the entrance at `0`, and `m` one-way tunnels, each `a -> b` with `a < b` — every tunnel goes strictly deeper. So the graph is a DAG with no cycles, and, usefully, the plain index order is already a topological order. Each chamber `i` carries a value `v[i]` (treasure, trap, or zero). An explorer enters at `0`, walks along tunnels collecting the value of every chamber it stands in, and may stop at any chamber. I want the maximum total collectible on one descent.

Two clauses decide the whole problem, and they pull on signs in opposite directions. The explorer *starts* at chamber `0`, so `v[0]` is collected unconditionally — there is no "refuse to enter" move, and therefore the answer can be negative. But the explorer *may stop anywhere*, so it is never forced to keep descending into traps. Mandatory at the entrance, optional everywhere below: that asymmetry is what this problem is built around, and confusing the two is exactly how a legitimately negative answer becomes a phantom `0`.

Scale fixes the datatypes before anything else. `n` up to `2*10^5`, `|v[i]|` up to `10^9`, and a path can visit up to `n` chambers, so a path sum can reach `2*10^14` — five orders of magnitude past the 32-bit signed range. Every value and accumulator is `long long`; an `int` here is a silent wrong-answer on the large tests.

Greedy first, because it is tempting and I want to know whether it survives. "Step into the best-looking child while the total improves, else stop" is `O(n+m)` and a few lines, but one step of lookahead is a local rule on a global path. On the chain `0 -> 1 -> 2` with `v = [1, -5, 100]`, greedy stands at `0` (total `1`), sees its only child `1` has value `-5`, and since stepping there would drop the total to `-4` it stops at `0` returning `1`. The optimum is `0 -> 1 -> 2` collecting `1 - 5 + 100 = 96`. The negative chamber is a toll guarding a treasure that dwarfs it, and one-step greedy refuses to pay it. Greedy is out.

So a DAG DP. Let `best[u]` be the most value collectible on a descent that *starts* at `u`. Standing at `u` always collects `v[u]`; then I either stop, adding nothing, or descend into exactly one child and continue optimally from there. Taking the better of the two:

```
best[u] = v[u] + max(0, max over children c of best[c])
```

The inner `max(0, …)` is the stop option — descend into no one, add nothing. The `v[u] +` sits *outside* it: being on `u` is mandatory, so `v[u]` is never clamped. A childless chamber gets `best[u] = v[u]`. The answer is `best[0]`, with no outer clamp, because the entrance cannot be refused — the per-node descent is clamped at `0`, the final answer is not.

Because every child of `u` has a strictly larger index, filling `best` in decreasing index order means all of a node's children are already computed when I reach it. That decreasing-index scan is a valid reverse topological order for free — no recursion (a depth-`2*10^5` chain would blow a recursive call stack), no separate topological sort.

On the sample `v = [3, -5, 4, -2, 7, -8]` with tunnels `0->1, 0->2, 1->3, 2->4, 2->5, 1->4, 3->5`, expected `14`, filling in decreasing index:
- `best[5] = -8` (leaf), `best[4] = 7`.
- `best[3] = -2 + max(0, best[5]=-8) = -2` — the only descent from `3` leads to a `-8`, so stop.
- `best[2] = 4 + max(0, 7, -8) = 11`.
- `best[1] = -5 + max(0, -2, 7) = 2`.
- `best[0] = 3 + max(0, 2, 11) = 14`.

Matches, via `0 -> 2 -> 4 = 3 + 4 + 7`.

Now the transcription, where the sign discipline is easy to get wrong. The reflex borrowed from max-subarray is to clamp the whole node total, `best[u] = max(0LL, v[u] + descend)`. The smallest input bites it: `n = 1`, `v = [-7]`, no tunnels, true answer `-7` (enter, nowhere to go, collect `-7`). That code computes `max(0, -7 + 0) = 0`. The clamp `max(0, v[u] + descend)` encodes "I may collect nothing at all," which is false at the entrance where `v[0]` is forced. And it is not only a root bug: a node whose true `best` is negative would advertise `0` upward, making its parent think descending into it is free. The clamp belongs on the descent alone: `best[u] = v[u] + max(0LL, descend)`, i.e. initialize `descend = 0`, fold children in with `descend = max(descend, best[c])`, and the `0` start *is* the stop option, so it reduces to `best[u] = v[u] + descend`. Re-trace `n=1, v=[-7]`: `descend = 0`, `best[0] = -7`. Correct.

The mirror mistake is dropping that `0` — writing `best[u] = v[u] + max over children` with no stop option, forcing a descent whenever a child exists. On `0 -> 1 -> 2`, `v = [5, -1, -100]`, the right answer is `5` (stop at the entrance; descending costs `-1` and then at best `-96`). Without the `0`: `best[2] = -100`, `best[1] = -1 + (-100) = -101`, `best[0] = 5 + (-101) = -96` — the explorer is dragged into the trap. With the `0`: `best[1] = -1 + max(0, -100) = -1`, `best[0] = 5 + max(0, -1) = 5`. Correct. The two bugs are opposites — over-clamping the mandatory entrance, under-clamping by dropping the optional stop — and the fix sits exactly between them: clamp the descent, never the node.

A couple of corners, since sign code dies here. All-negative chain `v = [-1, -2, -3]`: `best` fills `-3, -2, -1`, answer `-1` — stop at the entrance, correctly negative rather than `0`. Negative entrance redeemed by a deep treasure, `v = [-10, 0, 4]` on a chain: `4, 4, -6`, answer `-6`, beating "stop at `0`" (`-10`) — the DP pays the toll. `m = 0` with `n > 1`: chamber `0` has no children, `best[0] = v[0]`. Parallel tunnels only re-`max` the same child value and change nothing.

As an independent check, an exhaustive enumerator — walk every directed path from `0`, record the running sum at every prefix, take the maximum — is obviously correct and completely unlike the memoized DP; on all the small instances above the two agree, confirming the recurrence matches the literal "best descent."

The solution is one self-contained `O(n+m)` file: read the graph, fill `best` in decreasing index order with `best[u] = v[u] + max(0, best of children)`, and print `best[0]`. The full module is in the answer.
