I have `n` papers, each with an integer prestige `p[i]` that may be negative, and `m` directed citation links forming an acyclic graph. I choose a start paper, follow links one at a time, may stop whenever I like, and want to maximize the sum of prestige over the papers on the resulting directed path, each counted once, reading at least the start. Two constraints fix the shape of the code before I pick an algorithm. First, `n, m <= 2*10^5` and `|p[i]| <= 10^9`, and a path can hold up to `n` papers, so a score reaches `2*10^5 * 10^9 = 2*10^14` — two orders of magnitude past the ~`2.1*10^9` range of 32-bit `int`. Every prestige value, every accumulator, and the answer must be 64-bit `long long`; an `int` is a silent wrong-answer on the long-chain tests. Second, the phrase "must read at least one paper" bites: unlike a no-adjacent-subset problem where the empty set is legal and floors the answer at `0`, here I cannot read nothing, so an all-negative instance must return the largest (least-negative) single prestige, which is itself negative. That is exactly where a stray `max(..., 0)` would wrongly clamp to `0`, and I keep it in view.

Two routes are on the table. *Greedy by next-prestige*: from the current paper hop to the reachable out-neighbour with the largest `p[v]`, continue while the running score climbs, run from every start and keep the best walk. Near-linear, a dozen lines — but the path constraint is global: what matters about a neighbour is not its own prestige but the prestige of the best path it can lead into, and greedy compares only the local value. That mismatch is the classic place greedy breaks, so I try to break it before trusting it. *DFS-based DP on the DAG*: define `best[u]` as the maximum score of a path that *starts* at `u`. Because the graph is acyclic, `best[u]` depends only on the `best[v]` of its out-neighbours, which come "later" in any topological order, so a depth-first post-order traversal computes it in `O(n + m)`. The risk there is transcription — the exact recurrence and running the DFS on a chain of `2*10^5` nodes without blowing the call stack.

To break greedy I need a concrete instance. Nodes `0,1,2,3`; prestige `p = [1, 10, 1, 100]`; links `0->1`, `0->2`, `2->3`. From `0`, greedy sees neighbour `1` (prestige `10`) and neighbour `2` (prestige `1`), takes the richer-looking `1`, and dead-ends at `1 + 10 = 11` since `1` has no out-links. The optimal play is `0 -> 2 -> 3` = `1 + 1 + 100 = 102`, hiding behind a modest paper. Greedy's best over all starts is only `101` (start at `2`). The gap is precisely the structural flaw: a neighbour's own prestige is the wrong comparison; the right one is `best[neighbour]`, which greedy never computes. Greedy is out.

So `best[u]` is the max score over directed paths starting at `u`. I must read `u`, so `p[u]` is always counted; then I either stop, or move to exactly one out-neighbour `v` and continue optimally from there, contributing `best[v]` (which already includes `p[v]` and everything beyond). Taking the larger option:

```
best[u] = p[u] + max(0, max over edges u->v of best[v]).
```

The `max(0, ...)` is the option to stop at `u`; it declines any continuation whose `best[v]` is negative. The final answer is `max over all u of best[u]`, and I seed that outer maximum at `-infinity`, *not* `0`, for the must-read-one reason above — the best start might be a lone negative paper. In a DAG every out-neighbour `v` precedes `u` in reverse-topological order, so computing `best[u]` only after all its successors are done is well-founded, no circular dependency.

On the documented sample — `p = [3, 8, 2, 9, 1, 7]`, links `0->1, 0->2, 2->3, 3->4, 1->5, 4->5` — I fold from the sinks: `best[5]=7`, `best[4]=1+7=8`, `best[3]=9+8=17`, `best[2]=2+17=19`, `best[1]=8+7=15`, `best[0]=3+max(15,19)=22`. The outer maximum is `22`, matching. And `0` picks neighbour `2` because `best[2]=19 > best[1]=15` — exactly the `0->2->3->4->5` path greedy missed by comparing `p[1]=8` against `p[2]=2` instead of `best[1]` against `best[2]`.

The natural transcription is recursive — solve successors first:

```
long long dfs(int u) {
    if (done[u]) return best[u];
    done[u] = 1;
    long long ext = 0;
    for (int v : adj[u]) ext = max(ext, dfs(v));
    best[u] = p[u] + ext;
    return best[u];
}
...
long long ans = 0;
for (int u = 0; u < n; u++) ans = max(ans, dfs(u));
```

Two things here are wrong, and both are the corners the constraints invite. The `ans = 0` seed: on `n=1, m=0, p=[-5]`, `dfs(0)` returns `best[0] = -5`, then `ans = max(0, -5) = 0`. But I am forced to read the one paper, so the answer is `-5`. Seeding the outer answer at `0` smuggles in an empty path that is illegal here; the fix is `ans = LLONG_MIN`. The *inner* `max(0, ...)` stays legitimate — it only declines to *extend* after `p[u]` is already paid, it does not invent an empty path. Re-tracing `p=[-5]` with the fix gives `-5`, and the all-negative chain `p=[-1,-2,-3]` with `0->1->2` gives `best[0]=-1`, `ans=-1`, reading only the single best paper. The second problem is recursion depth: an induced chain `0->1->...->(n-1)` with `n = 2*10^5` recurses `2*10^5` frames deep before any return — on the order of 10 MB of call stack, past the usual few-MB limit, a deterministic segfault on exactly the long-chain tests. The logic is right, the recursive *form* is not, so I convert to an explicit stack.

The explicit-stack post-order DFS keeps, per node, an edge cursor `it[u]` so I can resume scanning its out-neighbours after diving into a child, and a `done[u]` flag so each memo is computed once. The post-order moment is when `it[u]` runs off the end of `adj[u]`: every reachable successor is then `done`, so I fold `best[u] = p[u] + max(0, max_v best[v])` and pop.

```
for (int s = 0; s < n; s++) {
    if (done[s]) continue;
    stk.push_back(s);
    while (!stk.empty()) {
        int u = stk.back();
        if (it[u] < adj[u].size()) {
            int v = adj[u][it[u]++];
            if (!done[v]) stk.push_back(v);   // descend into unfinished child
        } else {
            long long ext = 0;
            for (int v : adj[u]) ext = max(ext, best[v]);
            best[u] = p[u] + ext;             // post-order: all v are done
            done[u] = 1;
            stk.pop_back();
        }
    }
}
```

The `!done[v]` guard on the push is what makes this safe on a re-convergent DAG. If a child is reachable from two parents — a diamond `0->1, 0->2, 1->3, 2->3` — the first parent finalizes it (`done[3]=1`) before the second parent descends, and the second push is skipped. A node could only be pushed twice if it were still unfinished at the second push, but a child still on the stack cannot be an out-neighbour of a node deeper than it without forming a back-edge, i.e. a cycle, which the DAG forbids. So `best` is computed exactly once per node.

Two more corners exercise the recurrence rather than restate it. `m=0, p=[-7,4]`: every path is a single paper, so the answer is `4`, the largest lone prestige. A pass-through-vs-avoid chain `p=[5,-100,5]`, `0->1->2`: `best[2]=5`, `best[1]=-100+max(0,5)=-95`, `best[0]=5+max(0,-95)=5` — the recurrence declines to cross the `-100` because `5-100+5 < 5`, and the answer is `5`, not the full-chain `-90`. The explicit stack lives on the heap and grows to at most `n` entries, so the induced `2*10^5` chain that overflowed the recursion runs without trouble.

For confidence beyond hand-tracing I ran a small-case stress: random DAGs built by fixing a topological order and adding lower-to-higher edges, prestige in a small signed range, checked against an independent exhaustive path-enumerator; every case agreed. The `O(n + m)` iterative-DFS DP — `long long` throughout, outer answer seeded at `LLONG_MIN` — is the full program in the answer module.
