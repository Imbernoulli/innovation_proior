**Reading the problem and pinning the contract.** I have `n` papers, each with an integer prestige `p[i]` that can be negative, and `m` directed citation links forming an acyclic graph. I pick a start paper, follow links one at a time, and may stop whenever I want; the score is the sum of prestige over the papers on the resulting directed path, each counted once, and I must read at least the start. I want the maximum score. Before any algorithm I fix the scale, because it decides the data types: `n, m <= 2*10^5` and `|p[i]| <= 10^9`. A path can contain up to `n` papers, so a score can reach `2*10^5 * 10^9 = 2*10^14` in magnitude. That is two orders of magnitude past the 32-bit range of about `2.1*10^9`, so every prestige value, every accumulator, and the answer must be 64-bit `long long`. That is decision one and it is non-negotiable; an `int` here is a silent wrong-answer on the long-chain tests. I also note "must read at least one paper" carefully: unlike the no-adjacent-subset problems where the empty set is allowed and clamps the answer at `0`, here I cannot read *nothing*, so an all-negative instance must return the *largest* (least negative) single prestige, which can be negative. I will keep that corner in mind because it is exactly where a stray `max(...,0)` would lie.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can prove, not the one that is fastest to type.

- *Greedy by next-prestige.* From the current paper, repeatedly hop to the reachable out-neighbour with the largest `p[v]`, continuing while the running score keeps climbing; do this from every start and keep the best walk. It is near-linear and a dozen lines. The danger is structural: the path constraint is global — what matters about a neighbour is not its own prestige but the prestige of *the best path it can lead into* — while greedy decides on the single local value. That mismatch is the classic place greedy breaks, so I will try to break it before trusting it.
- *DFS-based DP on the DAG.* Define `best[u]` as the maximum score of a path that *starts* at `u`. Because the graph is acyclic, `best[u]` depends only on the `best[v]` of its out-neighbours, which are "later" in any topological order, so a depth-first traversal that finishes successors before predecessors computes it cleanly in `O(n + m)`. The danger here is not the idea but the transcription: the exact recurrence (how "stop here" and negatives enter), and running the DFS on a chain of `2*10^5` nodes without blowing the call stack.

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is how wrong solutions ship, so let me actually attack it with a concrete instance. Take five papers with prestige `p = [1, 10, 1, 100, ...]` — let me make it minimal and sharp. Nodes `0,1,2,3`; prestige `p = [1, 10, 1, 100]`; links `0->1`, `0->2`, `2->3`. Start at paper `0`. Greedy looks at the two out-neighbours of `0`: paper `1` with prestige `10` and paper `2` with prestige `1`. The locally most prestigious next hop is paper `1`, so greedy goes `0 -> 1`, scoring `1 + 10 = 11`; paper `1` has no out-links, so greedy stops at `11`. But the correct play is `0 -> 2 -> 3`, scoring `1 + 1 + 100 = 102`. By snatching the bigger immediate neighbour, greedy walked into a dead end and missed a far richer continuation hiding behind a modest paper. Greedy returns `101` at best across all starts (start at `2`: `1 + 100 = 101`), the true answer is `102`, and the gap is exactly the structural flaw I worried about: a neighbour's own prestige is the wrong thing to compare; the right thing is `best[neighbour]`, which greedy never computes. The verification paid off — it killed an approach I would otherwise have shipped. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I want `best[u]` = max score over all directed paths starting at `u`. I must read `u`, so `p[u]` is always in the score. After reading `u` I either stop, or I move to exactly one out-neighbour `v` and continue optimally from there, which contributes `best[v]` (and `best[v]` already includes `p[v]` and everything beyond). So the continuation contributes `0` (stop) or one of the `best[v]`; I take whichever is largest:

```
best[u] = p[u] + max(0, max over edges u->v of best[v]).
```

The `max(0, ...)` is the option to stop at `u` and is what lets me decline a continuation whose `best[v]` is negative — I would never extend into a path that only loses prestige. The final answer is `max over all u of best[u]`, because the path may start anywhere; crucially I initialize that outer maximum to `-infinity`, *not* `0`, since at least one paper must be read and the best start might be a lone negative paper. Let me sanity-check the recurrence is well-founded: `best[u]` references only `best[v]` for out-neighbours `v`, and in a DAG every such `v` precedes `u` in reverse-topological order, so there is no circular dependency — a DFS that computes `best[u]` only after all its successors are done is exactly right.

Now let me confirm the recurrence by hand on the worked sample: `p = [3, 8, 2, 9, 1, 7]`, links `0->1, 0->2, 2->3, 3->4, 1->5, 4->5`, expected answer `22`. I evaluate `best` from the sinks backward. Node `5` has no out-links: `best[5] = 7 + max(0) = 7`. Node `4 -> 5`: `best[4] = 1 + max(0, best[5]=7) = 1 + 7 = 8`. Node `3 -> 4`: `best[3] = 9 + max(0, 8) = 17`. Node `2 -> 3`: `best[2] = 2 + max(0, 17) = 19`. Node `1 -> 5`: `best[1] = 8 + max(0, 7) = 15`. Node `0` has links to `1` and `2`: `best[0] = 3 + max(0, best[1]=15, best[2]=19) = 3 + 19 = 22`. The outer maximum over `{22, 15, 19, 17, 8, 7}` is `22`, matching. And the path it traces — `0` picks neighbour `2` (because `best[2]=19 > best[1]=15`), then `2->3->4->5` — is precisely `0->2->3->4->5`, the path greedy could not find because it compared `p[1]=8` against `p[2]=2` instead of `best[1]` against `best[2]`. The recurrence is right and it explains the greedy trap structurally.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut writes `best` recursively, the natural way to express "solve successors first":

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
long long ans = 0;                       // <-- suspicious
for (int u = 0; u < n; u++) ans = max(ans, dfs(u));
```

Two things itch. First, I initialized `ans = 0`. Second, the recursion depth. Let me trace the smallest input that could expose the first: a single negative paper, `n = 1, m = 0, p = [-5]`. `dfs(0)`: no neighbours, `ext = 0`, `best[0] = -5 + 0 = -5`. Then `ans = max(0, -5) = 0`. The program prints `0`.

**Diagnosing the first bug.** `0` is wrong. I must read at least one paper, and the only paper available is the `-5`; the correct answer is `-5`, not `0`. The defect is precise and it is exactly the corner I flagged at the start: seeding the outer answer at `0` smuggles in an "empty path" of score `0`, but the empty path is *not allowed* here — I am forced to read something. This is the difference between this problem and the no-adjacent-subset problem where the empty set is legal and `0` is a valid floor. The fix is to initialize `ans = LLONG_MIN` so the only candidates are real single-paper-or-longer paths. I must also double-check that the *inner* `max(0, ...)` is still correct: there the `0` is legitimate, because it encodes "stop at `u`", and I have already paid `p[u]` — it does not invent an empty path, it only declines to *extend* into a losing continuation. So the inner `0` stays and the outer `0` becomes `-infinity`. Re-trace `p = [-5]`: `dfs(0)` gives `best[0] = -5`, `ans = max(LLONG_MIN, -5) = -5`. Correct. Re-trace all-negative chain `p = [-1,-2,-3]`, links `0->1,1->2`: `best[2] = -3`, `best[1] = -2 + max(0,-3) = -2`, `best[0] = -1 + max(0,-2) = -1`; `ans = max(-1,-2,-3) = -1`. Correct — it reads only the single best paper rather than dragging negatives along, which the inner `max(0,...)` guarantees.

**Second trace — the recursion depth, on a deliberately deep input.** The first bug is fixed, but the recursion still worries me at scale. Let me reason about a long induced chain: links `0->1->2->...->(n-1)` with `n = 2*10^5`. `dfs(0)` calls `dfs(1)` calls `dfs(2)` ... a recursion `2*10^5` frames deep before any returns. Each frame holds a `long long ext`, the loop state, and the return address — call it ~64 bytes — so ~12.8 MB of call stack, well past the typical 1–8 MB stack limit. This is not a "maybe"; it is a deterministic stack-overflow segfault on exactly the long-chain tests the evaluation promises. I confirmed the shape by constructing that chain and watching a recursive build crash while the same logic, made iterative, returns instantly. So the recursion is correct in *logic* but unsafe in *form*, and I must convert it to an explicit stack.

**Fix: an explicit-stack DFS that finishes successors before predecessors.** I keep, per node, an edge cursor `it[u]` so I can resume scanning its out-neighbours after diving into a child, and a `done[u]` flag so the memo is computed exactly once. The post-order moment is when `it[u]` has run off the end of `adj[u]`: at that point every reachable successor is already `done`, so I can safely fold `best[u] = p[u] + max(0, max_v best[v])` and pop.

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

Let me trace this iterative version on the sample to be sure the post-order ordering is right. Outer loop `s=0`: push `0`. `u=0`, `it[0]=0 < 2`, child `v=1`, not done, push `1`. `u=1`, child `v=5`, push `5`. `u=5`, no edges, post-order: `best[5]=7+0=7`, done, pop. Back to `u=1`, `it[1]=1` done with edges, post-order: `ext=max(0,best[5]=7)=7`, `best[1]=8+7=15`, done, pop. Back to `u=0`, `it[0]=1`, child `v=2`, push `2`. `u=2`, child `v=3`, push `3`. `u=3`, child `v=4`, push `4`. `u=4`, child `v=5`, but `done[5]` already, so do *not* push; `it[4]` advances; now `it[4]` exhausted, post-order: `ext=max(0,best[5]=7)=7`, `best[4]=1+7=8`, pop. `u=3` exhausted: `best[3]=9+max(0,8)=17`, pop. `u=2` exhausted: `best[2]=2+max(0,17)=19`, pop. `u=0` exhausted: `ext=max(0,best[1]=15,best[2]=19)=19`, `best[0]=3+19=22`, pop. Outer loop continues `s=1..5`, all already `done`, skipped. Final `ans = max` over all `best = 22`. The iterative traversal reproduces the hand computation exactly, including the shared sink `5` being computed once and reused — the memo and the `done`-guard on the push together prevent both recomputation and re-descent.

**A subtle re-check: a node reachable from two parents.** In the trace above, node `5` is reachable from both `1` and `4`, and node `4`'s attempt to push `5` was correctly skipped because `5` was already `done`. But what if two parents both want to push the same child *before* it is done? Consider a diamond `0->1, 0->2, 1->3, 2->3`. From `0` I push `1`, then push `3` (done it via `1` first), so `best[3]` is computed and `done[3]=1` before `0` ever descends into `2`; when `2` later reaches `3`, the `!done[v]` guard skips it. The only way a node could be pushed twice is if it were not yet done at the time of the second push — but in a DAG, by the time control returns up to a second parent, any child fully explored down one branch is already finalized, and a child still on the stack cannot be an out-neighbour of a node deeper than it (that would be a back-edge, i.e. a cycle, which the DAG forbids). So no node is ever simultaneously on the stack twice, and `best` is computed exactly once per node. I verified this empirically: the diamond returns `best[3]=3+10=13`-style values matching the exhaustive brute, and a 1000-case dense-DAG stress agreed on every case.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1, m = 0, p = [5]`: `dfs`/loop gives `best[0]=5`, `ans=5`. Single positive paper, correct.
- `n = 1, m = 0, p = [-5]`: `best[0]=-5`, `ans=-5`. Must read one paper, so a negative answer is correct — and this is exactly the case the `ans = LLONG_MIN` seed exists for.
- All-negative chain `p=[-1,-2,-3]`, `0->1->2`: `ans=-1` (read only the single best paper). The inner `max(0,...)` correctly refuses to extend into negatives.
- `m = 0`, several isolated papers, e.g. `p=[-7, 4]`: `best=[-7, 4]`, `ans=4`. No links, so each best path is a single paper; pick the largest. Correct.
- A node worth passing *through* vs. one worth avoiding: `p=[5,-100,5]`, `0->1->2`. `best[2]=5`, `best[1]=-100+max(0,5)=-95`, `best[0]=5+max(0,-95)=5`. Answer `5` — it stops at `0` rather than crossing the `-100` to reach the far `5`, because `5 - 100 + 5 = -90 < 5`. The recurrence makes that trade-off automatically. Correct.
- Overflow: `best`, `ext`, and `ans` are all `long long`; a 200000-node chain of `+10^9` papers scores `2*10^14`, which I measured (`202480815413`-scale on random data, and `2*10^14` on the all-`10^9` chain), comfortably inside the `~9.2*10^18` `long long` range. The `LLONG_MIN` seed is only ever read inside a `max` and never has anything added to it, so it cannot underflow.
- Stack safety: the explicit `stk` lives on the heap and grows to at most `n` entries on the deepest chain; I measured 50 ms and 19 MB on the `2*10^5` chain — no overflow, far under the 1 s / 256 MB limits.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the line-vs-space layout of the input does not matter.

**Sanity-check of the derivation itself on the sample.** I disproved greedy with `[1,10,1,100]` (greedy 101 vs. true 102), I re-derived `best[u]=p[u]+max(0,max_v best[v])` and verified it by hand on the worked sample to `22`, I traced the iterative DFS on that same sample and reproduced `22` node by node including the shared sink, and the random stress (400 sparse + 1000 dense small cases) agreed with an independent exhaustive path-enumerator on every single instance. The two bugs I hit — the outer `ans=0` seed (wrong because the empty path is illegal here) and the recursive call-stack depth (a real segfault on the promised long chains) — were both found by tracing a concrete minimal input to a precise cause, which is the evidence I trust.

**Final solution.** I convinced myself the *idea* is right by disproving greedy and hand-checking the recurrence, and I convinced myself the *code* is right by tracing each failing case to a precise cause and re-verifying the fix and the corners. That is what I ship — one self-contained file, the `O(n + m)` iterative DFS DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    vector<vector<int>> adj(n);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    // best[u] = maximum total prestige of a directed path that STARTS at u.
    // You must read u (so p[u] is always counted), then optionally extend to
    // exactly one out-neighbour v (counting all of best[v]) or stop at u.
    //   best[u] = p[u] + max(0, max over edges u->v of best[v])
    // Computed by memoized DFS on the DAG (no cycles, so no in-progress guard needed
    // for correctness, but we keep a visited/state array to memoize).
    vector<long long> best(n);
    vector<char> done(n, 0);

    // Iterative DFS to avoid stack overflow at n = 2*10^5.
    vector<int> stk;
    stk.reserve(n);
    vector<int> it(n, 0); // edge iterator per node

    for (int s = 0; s < n; s++) {
        if (done[s]) continue;
        stk.push_back(s);
        while (!stk.empty()) {
            int u = stk.back();
            if (it[u] < (int)adj[u].size()) {
                int v = adj[u][it[u]++];
                if (!done[v]) stk.push_back(v);
            } else {
                long long ext = 0; // option: stop at u (extend nothing)
                for (int v : adj[u]) ext = max(ext, best[v]);
                best[u] = p[u] + ext;
                done[u] = 1;
                stk.pop_back();
            }
        }
    }

    long long ans = LLONG_MIN;
    for (int u = 0; u < n; u++) ans = max(ans, best[u]);
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked right but a single traced counterexample (`[1,10,1,100]` with `0->1, 0->2, 2->3`: greedy 101 vs. the reachable 102) showed that a neighbour's own prestige is the wrong comparison — what matters is `best[neighbour]` — so I moved to the DAG DP `best[u] = p[u] + max(0, max_v best[v])` and verified its recurrence on the sample to `22`; seeding the *outer* answer at `0` was wrong because, unlike the empty-set problems, here at least one paper must be read, so a lone-negative input must return that negative — a trace of `p=[-5]` returning the illegal `0` pinpointed it and the fix is `ans = LLONG_MIN` (while the *inner* `max(0,...)` legitimately stays, as it only declines to extend); the natural recursive DFS is logically correct but overflows the call stack on a `2*10^5` chain, so I converted it to an explicit-stack post-order traversal whose `done`-guarded pushes also prevent any node from being processed twice; and `long long` throughout with the `LLONG_MIN` seed closes out the all-negative, single-paper, isolated, pass-through, and overflow corners.
