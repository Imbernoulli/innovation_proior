I have an undirected multigraph — `n <= 2*10^5` switches, `m <= 2*10^5` cables — and I must count the cables lying on at least one cycle, i.e. `m` minus the number of bridges. What lifts this above a textbook bridge count is written straight into the constraints: cables can be parallel (several on one pair), self-loops (`a == b`) are allowed, and each cable is a distinct object. Those two features are exactly where the ordinary bridge DFS quietly breaks. The output is a single integer bounded by `m`, so 32 bits hold it comfortably; I still form `m - #bridges` in `long long`, since a subtraction against an `int` count is a cheap place for a silent narrowing to sneak in. The real scale worry is not arithmetic but recursion depth: a legal input is a single path of `2*10^5` vertices, and a recursive DFS descends that `2*10^5` frames deep and overruns the 8 MB stack. I keep both traps — parent-edge handling and stack depth — in view from the start.

Two routes. The brute one — delete each cable, re-test whether its endpoints stay connected over the rest — is transparently correct (that *is* the definition of a non-bridge) but costs `O(m(n+m)) ≈ 8*10^10` on the largest input; it survives only as the oracle I check against. The submission is a single DFS computing `disc[u]` (discovery time) and `low[u]` (the smallest `disc` reachable from `u`'s subtree via at most one back edge), marking a tree edge `u -> v` as a bridge iff `low[v] > disc[u]`, then answering `m - #bridges`. That is `O(n+m)`; all the risk is in transcription, specifically the parent-edge and self-loop handling.

The recurrence is `low[u] = min(disc[u], min disc[w] over back edges u->w, min low[c] over tree children c)`. If `low[v] <= disc[u]`, some back edge from `v`'s subtree reaches `u` or an ancestor, giving a route around the tree edge, so it lies on a cycle; if `low[v] > disc[u]`, nothing in `v`'s subtree climbs past `v` except through that edge, so cutting it isolates the subtree — a bridge. The whole thing hinges on how "back edge to a non-parent vertex" is read, and in a multigraph that reading is the trap.

In a simple graph the shortcut is to ignore the neighbour equal to the DFS parent. But here two cables can join `u` and its parent `p`: the first is the tree edge I entered on, the second is a genuine back edge forming a 2-cycle, making *both* cables non-bridges. Skipping every edge to vertex `p` throws away that second back edge, `low[u]` never drops to `disc[p]`, and I wrongly flag the tree edge as a bridge. So the DFS must carry the specific parent *edge id* it entered on and skip only that one instance, not the parent vertex.

Self-loops settle cleanly: `(a, a)` can never be a bridge — removing it disconnects nothing. Stored once in `adj_[a]`, the DFS meets it as a back edge to the already-discovered `a`, it lowers `low[a]` to `disc[a]` (a no-op), and `isBridge` is never set for it, so it counts toward the answer automatically. Storing it once rather than twice also keeps the duplicate from ever being mistaken for some vertex's parent edge.

The version most people write skips by parent vertex; putting it on the smallest parallel case makes the cost concrete:

```
void dfs(int u, int pu) {                 // pu = parent VERTEX
    disc[u] = low_[u] = ++timer_;
    for (auto &e : adj_[u]) {
        int v = e[0], id = e[1];
        if (v == pu) continue;            // skip edges to the parent vertex
        if (!disc[v]) {
            dfs(v, u);
            low_[u] = min(low_[u], low_[v]);
            if (low_[v] > disc[u]) isBridge[id] = true;
        } else {
            low_[u] = min(low_[u], disc[v]);
        }
    }
}
```

Trace the smallest parallel case, `n = 2`, cables `(1,2)` id 0 and `(1,2)` id 1 — a 2-cycle, both non-bridges, answer must be `2`. `dfs(1,-1)`: `disc[1]=1`, take `{2,0}` as a tree edge → `dfs(2,1)`: `disc[2]=2`, and both `{1,0}` and `{1,1}` have `v == pu(1)`, so **both are skipped**; `low_[2]` stays `2`. Back at 1: `low_[2](2) > disc[1](1)` → id 0 marked bridge. Result: `#bridges = 1`, answer `2 - 1 = 1` — wrong, the truth is `2`. On the stated 7-cable sample this same code mismarks one of the parallel `4-5` cables and prints `4` instead of `5`. Skipping by vertex threw away the parallel cable's real back edge and manufactured a phantom bridge — off by one.

The fix is one line — carry the parent edge id `peId` and skip `if (id == peId) continue;` instead of `if (v == pu)`. Re-tracing the parallel case: inside `dfs(2, 0)` only `{1,0}` (the entry id) is skipped; `{1,1}` is now honoured as a back edge, `low_[2]` drops to `disc[1]=1`, so `low_[2](1) > disc[1](1)` is false, id 0 is not a bridge, `#bridges = 0`, answer `2`. On the 7-cable sample the corrected recurrence marks only `3-4` and `5-6`, giving `7 - 2 = 5`, matching the given answer — the triangle's three cables and the two parallel `4-5` cables are the five non-bridges.

That fix is logically correct but still recursive, and the stack trap is real. I run it on the promised `2*10^5`-vertex path (every edge a bridge, answer `0`): segmentation fault. The 8 MB stack holds far fewer than `2*10^5` frames of this size, so a perfectly legal input crashes with a runtime-error verdict. I rewrite the DFS iteratively with an explicit stack of frames `(u, peId, iterator-into-adj_[u])`.

The subtle part of an iterative bridge DFS is that the parent relaxation `low[p] = min(low[p], low[u])` and the bridge test happen when a child frame is *popped*, not when it is pushed. My first cut got this wrong: it relaxed using `low_[u]` after `--top` but read `peId` from the new top frame — the parent's own parent edge, not the edge from parent to `u`. Tracing the 3-cycle `n=3`, `(1,2)0 (2,3)1 (3,1)2` (answer `3`, no bridges) pinned the ordering: when the frame for `u=3` is exhausted I must read *its own* `peId` (= 1, the `2-3` edge) before decrementing, then relax the new top `p=2`, testing `low_[3] > disc[2]` and setting `isBridge[1]` only if it holds. So I read `u` and `peId` from the current top at the loop head and, on a pop, use those popped values while the parent is the post-decrement top. Retesting the `2*10^5`-vertex path returns `0` with no segfault — same `disc`/`low` math, depth problem gone.

The remaining corners fall straight out of the same recurrence. `m = 0` (and `n = 0`): the root loop and the bridge scan both run zero times, answer `0`. A single self-loop `(1,1)`: counted as a back edge to itself, never a bridge, answer `1`. A single cable `(1,2)`: its far endpoint has only the skipped parent edge, `low > disc`, a bridge, answer `0`. Two parallel cables: `2`, traced above. Disconnected graphs: the `for s = 1..n` loop launches a DFS from every undiscovered vertex, so bridges are counted per component and summed. Array sizes: `adj_/disc/low_` indexed by vertex fit `200005`; `isBridge` indexed by edge id fits `400005` (generous); the explicit-stack arrays need at most `n+1` live frames — all within 256 MB. `cin >> n >> m` then `m` pairs skips arbitrary whitespace and newlines, so token-vs-line layout does not matter, and `if (!(cin >> n >> m)) return 0;` guards truly empty input.

What I ship is one self-contained file: the `O(n+m)` iterative bridge DFS, skipping the parent edge by id (not by vertex), storing each self-loop once, and answering `m - #bridges` in `long long`. The full program is in the answer.
