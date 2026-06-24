**Reading the problem and pinning the contract.** I have a rooted tree of `n` junctions, numbered `1..n`. Each non-root junction `i` has a pipe up to its parent `par[i]` with per-unit cost `cost[i]`. Every junction with no children is an *outlet*, and each outlet must receive exactly one unit of water that travels from the root all the way down to it, paying each pipe's per-unit cost once on the way. I must output the *total pumping work* = sum over all outlets of the sum of pipe costs along that outlet's root-to-outlet path. Input is `n`, then for `i = 1..n` a line `p_i c_i`, with the root flagged by `p_i = -1`; the root id is not necessarily `1`, and a parent may be listed after its child. I print a single integer.

Before any algorithm I fix the scale, because that is what decides the data types and there is a deliberate trap here. `n <= 2*10^5`, costs `c_i <= 10^6`. Consider the worst shape: a root, one child junction `c` directly under it, and roughly `2*10^5` outlets all hanging directly off `c`, every pipe at cost `10^6`. The pipe from `c` to the root carries one unit per outlet, so it alone contributes `cost = 10^6` times `leaves ≈ 2*10^5` = `2*10^11`. Each outlet's own pipe contributes `10^6 * 1`, and there are `~2*10^5` of those, another `~2*10^11`. So the total is on the order of `4*10^11`. That is roughly 186 times larger than the 32-bit signed maximum `2^31 - 1 ≈ 2.147*10^9`. So **the answer overflows a 32-bit int by two orders of magnitude**, and worse, a *single product* `cost[v] * leaves[v]` can be `~2*10^11`, which overflows `int` before I even add it into a total. Both the products and the accumulator must be 64-bit. `long long` maxes out near `9.2*10^18`, and `4*10^11` sits far below that with enormous headroom, so `long long` is safe and `int` is a silent wrong answer. I write this down now so I do not forget it under the pressure of getting the traversal right; it is the first non-negotiable decision.

**Laying out the candidate approaches.** Once the tree is parsed, every method is some depth-first aggregation; the question is what I aggregate and whether it is fast enough.

- *Per-outlet path summation.* For each outlet, walk up parent pointers to the root, summing pipe costs; add that into the total. This is the literal definition of the problem, so it is *obviously correct* and makes a perfect independent reference. But its cost is (number of outlets) times (path length). On a "broom" — a long stem of length `L` ending in a bush of `M` leaves — every leaf walks the full stem, giving `O(L*M)`, which is `Θ(n^2)` when `L, M ≈ n/2`. At `n = 2*10^5` that is `10^{10}` operations: far over a 1-second limit. So this is my brute, not my submission.
- *Subtree-outlet-count DFS.* Here is the structural insight. A pipe `(v -> par[v])` is traversed once for *each* outlet in `v`'s subtree, because every such outlet's water descends through that pipe. So if `leaves[v]` is the count of outlets in the subtree rooted at `v`, then the pipe above `v` contributes exactly `cost[v] * leaves[v]` to the total, and the answer is the sum of that over all non-root `v`. Computing every `leaves[v]` is one DFS: a leaf (no children) has `leaves = 1`; an internal node has `leaves = sum of children's leaves`. That is `O(n)` time and `O(n)` memory. This is the one I can defend, provided I get the aggregation order and the integer width right.

I want to be sure the equivalence "sum of root-path costs over outlets" = "sum of `cost[v]*leaves[v]` over edges" is actually a theorem and not wishful thinking, because the whole efficient method rests on it.

**Deriving and checking the edge/path equivalence.** Fix any outlet `L`. Its contribution to the per-outlet definition is `sum of cost[v]` over every `v` on the path from `L` up to (but not including) the root — i.e. over every pipe on that path. Now sum over all outlets `L`. Swap the order of summation: instead of "for each outlet, for each pipe on its path", do "for each pipe `(v -> par[v])`, for each outlet whose path uses that pipe". An outlet `L` uses pipe `(v -> par[v])` exactly when `L` lies in the subtree of `v` (because the path from `L` to the root passes through `v`'s pipe iff `L` is at or below `v`). The number of such outlets is, by definition, `leaves[v]`. So the double sum equals `sum over non-root v of cost[v] * leaves[v]`. The swap is just reindexing a finite double sum, so it is exact. Good — the equivalence is a genuine identity, and it is precisely why my brute (per-outlet) and my submission (per-edge) can be expected to agree to the last digit.

Let me confirm both views on the sample. Tree: root `1`; `2` under `1` (cost `5`), `3` under `1` (cost `3`); `4` (cost `7`) and `5` (cost `2`) under `2`; `6` (cost `4`) under `3`. Outlets are `4, 5, 6`. Per-outlet: `4` -> `5+7 = 12`, `5` -> `5+2 = 7`, `6` -> `3+4 = 7`, total `26`. Per-edge: `leaves[4]=leaves[5]=leaves[6]=1`, `leaves[2]=2`, `leaves[3]=1`, `leaves[1]=3`. Non-root edge terms: `cost[2]*leaves[2]=5*2=10`, `cost[3]*leaves[3]=3*1=3`, `cost[4]*1=7`, `cost[5]*1=2`, `cost[6]*1=4`; sum `10+3+7+2+4 = 26`. Both give `26`. The derivation checks out on the sample.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My instinct for the DFS is recursion: read the tree, find the root, then `dfs(root)` returning the outlet count of a subtree while accumulating `cost[v]*leaves[v]` into a global. Here is the core I first wrote, deliberately with `int` where I will trip:

```
int total = 0;                      // <-- will revisit this type
int dfs(int u) {                    // returns number of outlets under u
    if (children[u].empty()) return 1;
    int s = 0;
    for (int w : children[u]) {
        int lw = dfs(w);
        total += cost[w] * lw;      // pipe (w -> u) carries lw outlets
        s += lw;
    }
    return s;
}
```

Two things nag me, so I trace. First the small correctness trace on the sample, ignoring overflow for the moment by pretending the numbers are tiny. `dfs(1)`: children `2,3`. `dfs(2)`: children `4,5`. `dfs(4)` -> `1`; `total += cost[4]*1 = 7` (total 7); `dfs(5)` -> `1`; `total += cost[5]*1 = 2` (total 9); `dfs(2)` returns `2`; back in `dfs(1)`, `total += cost[2]*2 = 10` (total 19); then `dfs(3)`: child `6`; `dfs(6)` -> `1`; `total += cost[6]*1 = 4` (total 23); `dfs(3)` returns `1`; back in `dfs(1)`, `total += cost[3]*1 = 3` (total 26). Returns `3`. Total `26`. The *logic* is right and the accumulation order is a clean post-order: I add a pipe's term only after its child's subtree count is known.

**The first bug — overflow, exactly the trap I flagged.** Now I stop pretending the numbers are tiny and look at the types. `total` is `int`, and `cost[w]` and `lw` would be `int` too. On the broom worst case (`leaves ≈ 2*10^5`, `cost = 10^6`), the line `total += cost[w] * lw` first computes `cost[w] * lw` as an `int * int` -> `int`, which is `10^6 * 2*10^5 = 2*10^11`, wrapping modulo `2^32` *before* the addition even happens. And `total` itself, climbing toward `4*10^11`, has long since wrapped. I do not want to merely reason about this — I built the broom input (`n = 200000`, stem cost `10^6`, `199998` leaves at cost `10^6`) and ran both an all-`int` version and a `long long` version. The `int` version printed `564041472`; the correct value is `399996000000`. That is not "a little off", it is garbage from two independent wraparounds: the product overflows `int`, and the accumulator overflows `int`. The fix is to make `total` a `long long`, make `cost[]` a `long long` (or at least cast in the multiply), and let the subtree count be `long long` as well so the product is evaluated in 64-bit. With `long long` the product `2*10^11` and the total `4*10^11` both fit with a factor of more than `10^7` to spare. This is the headline pitfall and the trace nails its precise mechanism: the overflow is in the multiply *and* in the sum, so widening only the accumulator would not be enough — the operands of `*` must be 64-bit too.

**The second bug — recursion depth on a deep chain.** The other thing that nagged me: recursive `dfs` on a path of `2*10^5` nodes recurses `2*10^5` frames deep. I built a chain of `200000` nodes (each cost `1`, so the answer is the single outlet's path cost `199999`) and ran the recursive version: it segfaulted from stack overflow — a typical default stack is ~`8 MB`, and `2*10^5` frames each holding the loop state and the call overhead can blow past that, and on judges with smaller stacks it is even more certain. So even with the integer types fixed, recursion is fragile here. I rewrite the DFS *iteratively*. The clean trick for post-order accumulation is: do an iterative pre-order with an explicit stack to produce a visiting `order` (parent before children), then process that `order` in reverse — reverse pre-order is a valid post-order for the purpose of "every node is finalized after all its descendants", because a child always appears later than its parent in pre-order, hence earlier in the reversed pass. So I compute `leaves[u]` in the reverse pass: if `u` has no children it is `1`, else it is the sum of its children's already-computed `leaves`. Then a final flat loop adds `cost[v]*leaves[v]` for every non-root `v`. No recursion, depth-proof.

**Fixing and re-verifying.** Here is the corrected core, with `long long` where it matters and an iterative DFS:

```
vector<long long> leaves(n + 1, 0);
vector<int> order; order.reserve(n);
vector<int> st; st.push_back(root);
while (!st.empty()) { int u = st.back(); st.pop_back();
    order.push_back(u);
    for (int w : children[u]) st.push_back(w); }
for (int i = (int)order.size()-1; i >= 0; i--) {
    int u = order[i];
    if (children[u].empty()) leaves[u] = 1;
    else { long long s = 0; for (int w : children[u]) s += leaves[w]; leaves[u] = s; } }
long long total = 0;
for (int v = 1; v <= n; v++) if (v != root) total += cost[v] * leaves[v];
```

Re-trace the broom mentally: `leaves[leaf] = 1` for each of `199998` leaves, `leaves[stem] = 199998`, `leaves[root] = 199998`. Total `= sum over leaves cost*1 + cost[stem]*199998 = 199998*10^6 + 10^6*199998 = 399996000000`. I ran it: `399996000000` — matches the hand computation, and matches what the `long long` reference gives. Re-trace the chain: the iterative version completes (no recursion to overflow) and prints `199999`, the single outlet's path cost. Re-run the sample tree: prints `26`. The two cases that broke before — broom (overflow) and chain (depth) — now pass, and they pass for exactly the reasons I changed: 64-bit arithmetic and an explicit stack.

**Sanity check on the post-order claim itself.** I should not trust "reverse pre-order is post-order" on faith; let me check the dependency it must satisfy: when I compute `leaves[u]` I read `leaves[w]` for every child `w`, so every child must be finalized *before* its parent in the reverse pass, i.e. every child must come *after* its parent in `order`. In my pre-order build I push `u` into `order` and then push its children onto the stack, so children are popped and appended to `order` strictly later than `u`. Hence in `order`, parent precedes all its descendants; in the reverse pass, descendants precede ancestors; so when I reach `u`, all its children are done. The invariant holds for every node, so the aggregation is correct regardless of tree shape. (It does not depend on the stack visiting children in any particular sibling order — only on parent-before-child, which the pre-order guarantees.)

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`: the single junction is the root, has no children, so it is itself an outlet but it has *no pipe above it*. The `order` is just `[root]`; the reverse pass sets `leaves[root] = 1`; the final loop skips `v == root`, so `total = 0`. There are no pipes, so zero work — correct. I ran it: `0`.
- Root not at id `1`: I locate the root by `p_i == -1` (I also accept `0` as a root marker defensively), not by assuming id `1`, so a root placed at, say, junction `2` or `3` is handled. I tested a 3-node tree rooted at junction `3`; sol and the brute both gave `9`.
- Parent listed after child: I never assume input order matches tree order. I first read *all* lines into `par/cost/children`, *then* traverse from the located root. So a child line appearing before its parent's line is fine — the adjacency is fully built before any DFS.
- Zero-cost pipes (`c_i = 0`): they contribute `0 * leaves = 0`, harmless; the generator includes zeros and the stress tests pass.
- Overflow corner (the headline): products and total are `long long`; worst total `~4*10^11` versus `LLONG_MAX ≈ 9.2*10^18` — more than seven orders of magnitude of headroom. `cost` is stored as `long long` so the multiply `cost[v] * leaves[v]` is evaluated in 64-bit, not `int`. No widening gap remains.
- Depth corner: iterative stack, so a `2*10^5`-deep chain does not touch the call stack at all.

**Verification at scale.** Beyond the hand traces I ran the independent per-outlet brute against this solution on `400` random small trees (rooted at junction `1`, `n` up to `8`, costs `0..6`) and a further `350` random trees with the *root placed at an arbitrary junction id* and `n` up to `12` — `750` cases total, **zero mismatches**. The brute is genuinely independent: it sums root-paths per outlet (the definition), while the solution sums `cost*leaves` per edge (the derived identity), so agreement is real evidence the identity and its transcription are both correct, not a tautology. I also confirmed the documented sample prints `26`, the broom prints `399996000000` (with the `int` variant printing the wrong `564041472`, proving the overflow is real), and the deep chain prints `199999` without crashing.

**Final solution.** I convinced myself the *idea* is right by proving the edge/path summation identity and checking it on the sample, and I convinced myself the *code* is right by tracing the two failure modes (int overflow on the broom, stack overflow on the chain) to precise causes, fixing each, and re-verifying against an independent brute at scale. That is what I ship — one self-contained file, iterative DFS, 64-bit throughout:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    // parent[i] and cost[i]: edge from i up to parent[i] with given cost.
    // Root is the unique node with parent -1 (0). Nodes are 1..n.
    vector<int> par(n + 1, 0);
    vector<long long> cost(n + 1, 0);
    int root = -1;
    vector<vector<int>> children(n + 1);
    for (int i = 1; i <= n; i++) {
        int p; long long c;
        cin >> p >> c;
        par[i] = p;
        cost[i] = c;
        if (p == -1 || p == 0) {
            root = i;
        } else {
            children[p].push_back(i);
        }
    }

    // leaves[v] = number of leaves in subtree of v.
    // A node with no children is a leaf and contributes 1.
    // total work = sum over non-root nodes v of cost[v] * leaves[v].
    // Iterative DFS (post-order) to avoid stack overflow on deep chains.
    vector<long long> leaves(n + 1, 0);
    vector<int> order;
    order.reserve(n);
    {
        vector<int> st;
        st.push_back(root);
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            order.push_back(u);
            for (int w : children[u]) st.push_back(w);
        }
    }
    // process in reverse pre-order = valid post-order accumulation
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int u = order[i];
        if (children[u].empty()) {
            leaves[u] = 1;
        } else {
            long long s = 0;
            for (int w : children[u]) s += leaves[w];
            leaves[u] = s;
        }
    }

    long long total = 0;
    for (int v = 1; v <= n; v++) {
        if (v == root) continue;
        total += cost[v] * leaves[v];
    }

    cout << total << "\n";
    return 0;
}
```

**Causal recap.** The problem is a DFS subtree aggregation, and the key identity — total work = sum over edges of `cost[v]` times the outlet count below `v` — I proved by swapping the order of a finite double sum and checked on the sample (`26`). My first recursive `int` version was logically correct on tiny numbers but failed twice at scale: on the broom worst case the `int` product `cost*leaves ≈ 2*10^11` and the `int` total `≈ 4*10^11` both wrap (the `int` build printed `564041472` for a true `399996000000`), so I widened the product operands *and* the accumulator to `long long`; and on a `2*10^5`-deep chain the recursion blew the call stack, so I replaced it with an iterative pre-order whose reverse pass is a valid post-order (children always finalized before parents). With both fixes the solution matches an independent per-outlet brute on `750` random cases and handles the `n = 1`, non-root-id, out-of-order-parent, zero-cost, deep-chain, and overflow corners.
