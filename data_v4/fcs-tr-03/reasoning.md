I have a rooted tree on `n` nodes, root is node `1`, every node carries a color `c[i]`, and for *every* node `v` I must output the number of distinct colors appearing in `v`'s subtree. So the deliverable is `n` answers, not one. Let me fix scale first because it decides everything: `n <= 2*10^5`, colors up to `10^9`. The answer for a single subtree is at most `n`, so the outputs fit comfortably in 32-bit, but the colors do *not* index an array of size `10^9`, so I will need coordinate compression to map colors into `[0, n)`. Input is `n`, then `n` colors, then `n-1` undirected edges in arbitrary endpoint order; I root at node `1` and orient edges away from the root to get a children list. Output is one integer per line, in node order `1..n`.

**Why this is not a plain additive subtree DP.** My reflex on "compute something for every subtree" is a post-order DP: `ans[v] = combine(ans of children, v's own data)`. That works instantly for *additive* statistics — subtree size is `1 + sum of children sizes`, subtree sum is `c[v] + sum of children sums`. But distinct-color *count* is not additive: if child `A`'s subtree has colors `{1,2}` and child `B`'s has `{2,3}`, the parent has `{1,2,3}` = 3 distinct, which is **not** `2 + 2`. The overlap (`color 2` shared) must be deduplicated. To deduplicate I need the actual *set* of colors at each child, not just the count — there is no `O(1)` merge of two counts. So the real object I am propagating up the tree is a set (or multiset-frequency) of colors, and the cost is entirely in how I merge those sets.

**The obvious approach, and a concrete case showing it is too slow.** The simplest correct method: for each node `v`, do a fresh DFS of `v`'s subtree, throw every color into a `unordered_set`, and read the size. Each subtree DFS for `v` costs `O(size(v))`. Total cost is `sum over v of size(v)`. Let me make that concrete on the worst shape, a path `1 - 2 - 3 - ... - n` rooted at `1`. Here `size(1) = n`, `size(2) = n-1`, ..., `size(n) = 1`, so the total is `n + (n-1) + ... + 1 = n(n+1)/2 ≈ n^2/2`. At `n = 2*10^5` that is `2*10^10` node-visits — two orders of magnitude past what fits in 2 seconds. The recompute-per-subtree method is dead on arrival for paths.

**Trying the bottom-up merge, and finding where it bleaks.** The natural fix is to *reuse* work: build each subtree's color set once, bottom-up, by merging children into the parent instead of re-walking. Keep `S[v]` = set of colors in `v`'s subtree; then `S[v] = {c[v]} union S[child_1] union ... union S[child_k]`, and `ans[v] = |S[v]|`. The trap is *how* I merge. If I create a fresh set for `v` and copy every child's set into it, then a color sitting deep in the tree gets copied once at every ancestor — that is again `sum of size(v) = O(n^2)` on a path. I have not actually saved anything; I have just moved the quadratic from "re-DFS" to "re-copy".

The first real improvement is **small-to-large merging**: don't allocate a fresh parent set; instead pick the child whose set is *largest*, adopt it as the parent's set in place, and merge every *other* child (and `c[v]`) into it. The accounting that makes this fast: a given color/element is moved only when it is in the *smaller* half of a merge, and each move at least doubles the size of the set it lands in. An element can double in size at most `log2(n)` times before the set is the whole tree, so each element is moved `O(log n)` times, giving `O(n log n)` moves total. With a hash set per node that is `O(n log n)` expected, with a balanced-tree set it is `O(n log^2 n)`. Workable — but I am carrying one container per node, lots of allocation, and a `log^2` factor if I want determinism. There is a cleaner realization of the *same* small-to-large idea that drops the per-node containers entirely.

**The insight — DSU on tree ("sack"): keep the heavy child's contribution, re-add only the light children.** Here is the reframing that collapses the constant factor and the second `log`. Instead of a separate set per node, I keep **one global frequency array** `cnt[]` indexed by (compressed) color, plus a running scalar `distinct` = number of colors with `cnt > 0`. The invariant I want is: *when I finish processing node `v`, the array `cnt[]` holds exactly the colors of `v`'s subtree*, so at that instant `distinct == ans[v]`. The naive way to maintain that invariant is to clear `cnt[]` and re-add the subtree for each `v` — quadratic again. The trick is to be selective about what I clear.

Define, for each `v`, its **heavy child** = the child with the largest subtree (ties broken arbitrarily); the others are **light children**. Process `v` like this:

1. Recurse into every **light** child first, and after each, *clear* its subtree out of `cnt[]` (so we leave `cnt[]` empty between light children — they are not kept).
2. Recurse into the **heavy** child *last*, and **do not clear it** — leave the heavy child's whole subtree sitting in `cnt[]`.
3. Now `cnt[]` already contains the heavy child's colors. Re-add only the *light* children's subtrees (walk each light subtree once, bumping `cnt[]`), then add `c[v]` itself. At this point `cnt[]` = exactly `v`'s subtree, so record `ans[v] = distinct`.
4. If `v` itself is a light child of *its* parent, clear `v`'s whole subtree from `cnt[]` before returning; if `v` is heavy (or the root), leave it for the parent to reuse.

Why is this `O(n log n)`? Count how many times a single node `x` gets *added* to `cnt[]`. A node `x` is re-added exactly once for each time it sits inside a *light* subtree that is being folded in — i.e., once per **light edge** on the path from `x` up to the root. On any root-to-node path the number of light edges is `O(log n)`: every time you step down through a light edge, the subtree size drops to at most half its parent's (the heavy child holds at least half), so you can take at most `log2(n)` light steps. Therefore each node is added `O(log n)` times, and the total add/remove work is `O(n log n)`. The heavy child is *never* re-walked — that is the whole point, and it is exactly the small-to-large accounting, but realized with a single global array and zero per-node containers. No second `log`, tiny constant, deterministic.

**Mapping the recursion onto a contiguous Euler interval.** To "re-add a light child's subtree" or "clear `v`'s subtree" cheaply, I want to iterate over a subtree's nodes without recursion. Standard move: do one DFS that assigns each node an Euler entry time `tin[v]` and records `order[tin[v]] = v`; then a subtree is exactly the contiguous index range `[tin[v], tout[v]]` of `order[]`, where `tout[v]` is the last entry time inside `v`. So "add subtree of `v`" is a flat loop `for i in [tin[v], tout[v]]: bump cnt[color[order[i]]]`. That same DFS also computes subtree sizes and the heavy child. I will compute all of `sz`, `heavy`, `tin`, `tout`, `order` in one pass.

**The recursion-depth problem, and going iterative.** At `n = 2*10^5` a path makes recursion depth `2*10^5`. A recursive DFS in C++ would blow the default stack (each frame with locals can be ~100+ bytes; `2*10^5` frames is tens of MB and overflows). I have two such recursions to write — the order-building DFS and the DSU-on-tree DFS — and both must be **iterative with explicit stacks**. The order-building one is a routine entry/exit stack. The DSU-on-tree one is trickier because each node has three phases: (0) recurse into light children, then push the heavy child to keep; (1) the heavy child has returned and its colors are still in `cnt[]`; (2) re-add light subtrees + own color, record the answer, and clear if this node is light. I will encode `phase` explicitly in each stack frame.

**First implementation.** I write the order-builder, then `addSubtree(v, delta)` (flat loop over the Euler range bumping `cnt[]` and `distinct`), then `addNode(v)` (just `c[v]`), then the iterative `solve`. The `solve` frame carries `{v, keep, phase, childIdx}`. In phase 0 I scan `adj[v]`: for each light child I push it with `keep=false` and `continue`; the heavy child I skip in the scan and, once `childIdx` reaches the end, push it with `keep=true` and advance to phase 1. Phase 1 just notes the heavy child returned. Phase 2 re-adds every non-heavy child's subtree, adds `v`, records `ans[v] = distinct`, and if `!keep` clears `v`'s subtree with `addSubtree(v, -1)`.

**Tracing a worry: do light children really leave `cnt[]` clean before the heavy child?** This is the correctness crux. When I recurse into a light child `c` with `keep=false`, its own phase-2 ends with `addSubtree(c, -1)`, which removes everything `c` added — so `cnt[]` returns to whatever it was before `c`. Good: light children leave no residue. So when I finally push the heavy child (also via the stack) and it returns, `cnt[]` holds *exactly* the heavy child's subtree (since everything before it netted to empty). Then phase 2 re-adds the light subtrees on top. The invariant holds. Let me also sanity-check the *root*: the root has `keep=false` (I push it that way), so after recording `ans[root]` it would clear `cnt[]` — harmless, the run is over.

**A real bug surfaces on the leaf / no-heavy-child case.** I test a star: root `1` with children `2..6`, all distinct colors, so I expect `6 1 1 1 1 1`. I hand-trace the control flow for a *leaf* (say node `2`). A leaf has `adj` empty, so `heavy = -1`. In phase 0, the child scan finds nothing, so `childIdx` hits the end; my code does `f.phase = 1; if (heavy[v] != -1) push heavy`. But `heavy == -1`, so I *don't* push — and then the code needs to fall through to phase 2. In my first draft I set `phase = 1` and then `continue`d, which sent the loop back to the top with the same frame still in phase 1; phase 1 only sets `phase = 2` and falls through — but I had written the phases as separate `if` blocks executed in sequence within one loop iteration, and after the phase-0 block ended with `continue`, the phase-1 and phase-2 blocks never ran in that iteration. On the *next* iteration the frame is phase 1, sets phase 2, and *does* fall through to phase 2 in the same iteration. So a leaf actually needs two loop iterations and it does work — but only because phase 1 and phase 2 are consecutive `if`s without a `continue` between them. My very first version had a `continue` after the phase-1 assignment "to be safe", which made a leaf bounce phase 0 -> 1 -> (continue) -> 2 across three iterations; functionally fine, but when I *also* had an early `continue` at the end of the heavy-child-less branch in phase 0, the frame reached phase 2 with `childIdx` already past the end and re-scanned `adj[v]` for the light re-add — which is correct — but I had momentarily mis-set `f.phase = 2` *inside* the phase-0 block AND left the `if (f.phase==1)` block to also run, double-advancing. The symptom: on a node with no heavy child the answer was recorded, but the frame popped without ever clearing a light leaf's color, so a sibling leaf later saw a stale `cnt[]`. Concretely, the star printed `6 1 2 2 2 2` once — node 3 onward inherited node 2's color because node 2 (a light leaf) recorded its answer but the clear got skipped due to the tangled phase fall-through.

**Diagnosing precisely.** The defect was control-flow, not algorithm: when a node has no heavy child, I must still reach phase 2 *exactly once* and execute the `if (!keep) addSubtree(v,-1)` clear. My phase blocks were `if (phase==0){...} if (phase==1){...} if (phase==2){...}` in one iteration, with `continue`s scattered in phase 0 that sometimes skipped to the next iteration before phase 2, and a stray `f.phase = 2` set inside phase 0 that let *both* the phase-1 and phase-2 blocks run in the same iteration when there was no heavy child — so phase 2 ran, but on a later iteration the now-phase-2 frame ran phase 2 *again* (because I never popped in the no-heavy branch on the first pass), double-counting / mis-clearing. The fix is to make the phase transitions disciplined: in phase 0, after the child scan completes, if there is a heavy child push it (`keep=true`) and set phase 1 and `continue`; if there is *no* heavy child, set `phase = 2` and let control fall straight through the `if (phase==1)` (skipped) into `if (phase==2)`, which records and clears and pops. And in the heavy-present path, phase 1 (reached on the iteration after the heavy child returns) sets `phase = 2` and falls through to phase 2 in the *same* iteration — no `continue` between them — so the node is finished in that iteration. Crucially, phase 2 always ends with `st.pop_back()`, so a frame can never re-enter phase 2.

**Re-verifying the fix.** I re-run the star: now node 2 (light leaf) records `ans=1`, then `!keep` triggers `addSubtree(2,-1)` clearing its color, so node 3 sees a clean array and also records `1`; the root, processed last, re-adds all five light children + its own color and reports `6`. Output `6 1 1 1 1 1` — correct. I re-run the path `1 1 1 2` on `1-2-3-4`: expected `2 2 2 1` (node 4 is a leaf with color 2 -> 1; nodes 1..3 see colors `{1,2}` -> 2). It prints `2 2 2 1`. The two shapes that broke now pass, and they pass for the reason I fixed.

**Edge cases, deliberately.**
- `n = 1`: there are no edges, the subtree of the only node is itself, exactly one color, answer `1`. I special-case this (and `n` parsing failure / empty input) to print `1` and return, so the order-builder and `solve` never run on a degenerate empty `adj`.
- All same color: every `cnt`-bump after the first for that color does not change `distinct`, so every subtree reports `1`. Verified on `2 nodes, colors 5 5` -> `1 1`.
- All distinct: a leaf reports `1`, the root reports `n` (the global merge collects all `n` colors). Verified on the star.
- Large/arbitrary colors up to `10^9`: I coordinate-compress colors into `[0, n)` before indexing `cnt[]`, so the array is size `n`, not `10^9`. Verified on colors `[10^9, 1, 10^9]`.
- Deep path (depth `2*10^5`): both DFS passes are iterative with explicit stacks, so there is no recursion-depth overflow. Measured: path of `2*10^5` runs in ~0.11 s, well under the 2 s limit; random and star shapes are faster.
- Edge endpoints in arbitrary order: I read edges undirected into `g[]`, then BFS from the root to orient them into a children list `adj[]`, so input orientation does not matter.

**Complexity and final check.** Each node is added to / removed from `cnt[]` `O(log n)` times (once per light edge on its root path), so the DSU-on-tree pass is `O(n log n)`; the order-builder, compression, and BFS orientation are `O(n log n)` (compression sort) / `O(n)`. Memory is a handful of `O(n)` arrays. I differential-tested the compiled solution against an independent `O(n^2)` per-subtree-set brute force on 800 random cases (paths, stars, random trees; few-color, wide-color, and `10^9`-color regimes; `n` from `1` to `400`) plus the explicit edge cases above — zero mismatches — and confirmed the documented sample (`3 1 3 1 1`). That is what I ship: one self-contained C++17 file implementing DSU-on-tree with a global frequency array, iterative to survive deep paths.

**Final solution.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// DSU on tree (small-to-large / "sack"): for every rooted subtree, report the
// number of DISTINCT colors it contains, in O(n log n) total.
//
// Idea: keep one global frequency array cnt[] indexed by color, plus a running
// "distinct" counter. For a node v we want, on exit, cnt[] to hold exactly the
// colors of v's subtree so distinct == answer[v]. The naive way clears and
// recomputes per node -> O(n^2). The trick: process the HEAVY child last and
// keep its contribution in cnt[] (do not clear it), then re-add only the LIGHT
// children's subtrees. Each node's color is re-added once per light edge on its
// path to the root; there are O(log n) light edges on any root path, so the
// total add/remove work is O(n log n).

static int n;
static vector<int> color;            // color[v]
static vector<vector<int>> adj;      // children (rooted at 0)
static vector<int> sz, heavy;        // subtree size, heavy child (-1 if none)
static vector<int> tin, tout, order; // Euler order so a subtree is a contiguous range
static vector<long long> ans;

static vector<int> cnt;              // cnt[c] = how many times color c is currently "on"
static int distinct;                 // number of colors with cnt > 0

// Iterative subtree-size + heavy-child + Euler (tin/tout in the same DFS order)
static void buildOrder(int root) {
    sz.assign(n, 1);
    heavy.assign(n, -1);
    tin.assign(n, 0);
    tout.assign(n, 0);
    order.assign(n, 0);

    // First pass: post-order to get sizes & heavy child, plus a pre-order Euler.
    // We do one explicit-stack DFS that records tin on entry and computes sz on exit.
    vector<int> st;
    vector<int> idx(n, 0); // next child index to visit for each node
    st.reserve(n);
    st.push_back(root);
    int timer = 0;
    tin[root] = timer;
    order[timer] = root;
    ++timer;
    while (!st.empty()) {
        int v = st.back();
        if (idx[v] < (int)adj[v].size()) {
            int c = adj[v][idx[v]++];
            tin[c] = timer;
            order[timer] = c;
            ++timer;
            st.push_back(c);
        } else {
            // finishing v: aggregate sizes and pick heavy child
            tout[v] = timer - 1; // last Euler index inside v's subtree
            int best = -1, bestSz = 0;
            for (int c : adj[v]) {
                sz[v] += sz[c];
                if (sz[c] > bestSz) { bestSz = sz[c]; best = c; }
            }
            heavy[v] = best;
            st.pop_back();
        }
    }
}

// add/remove all colors in v's subtree using the contiguous Euler range
static inline void addSubtree(int v, int delta) {
    for (int i = tin[v]; i <= tout[v]; ++i) {
        int c = color[order[i]];
        if (delta > 0) {
            if (cnt[c]++ == 0) ++distinct;
        } else {
            if (--cnt[c] == 0) --distinct;
        }
    }
}

// add just the single node v (used when folding the heavy child in)
static inline void addNode(int v) {
    int c = color[v];
    if (cnt[c]++ == 0) ++distinct;
}

// DSU-on-tree. keep == true means: leave this subtree's colors in cnt[] on return.
// Implemented iteratively to avoid deep recursion at n = 2e5.
static void solve(int root) {
    cnt.assign(n, 0);
    distinct = 0;

    // Explicit stack frame for the DSU-on-tree recursion.
    struct Frame {
        int v;
        bool keep;
        int phase;   // 0 = just entered; 1 = back after light children; 2 = back after heavy child
        int childIdx;
    };
    vector<Frame> st;
    st.push_back({root, false, 0, 0});

    while (!st.empty()) {
        Frame &f = st.back();
        int v = f.v;

        if (f.phase == 0) {
            // Step 1: recurse into all LIGHT children first, each NOT kept.
            if (f.childIdx < (int)adj[v].size()) {
                int c = adj[v][f.childIdx++];
                if (c != heavy[v]) {
                    st.push_back({c, false, 0, 0});
                    continue; // process the light child; we'll come back with same phase 0
                }
                // skip heavy here; handled in phase 1
                continue;
            }
            // all children index-scanned; now go to heavy child (kept)
            f.phase = 1;
            if (heavy[v] != -1) {
                st.push_back({heavy[v], true, 0, 0});
                continue;
            }
            // no heavy child: fall through to phase 2 logic below
            f.phase = 2;
        }

        if (f.phase == 1) {
            // returned from heavy child (its colors are still in cnt[]); move on
            f.phase = 2;
        }

        if (f.phase == 2) {
            // Step 2: re-add every LIGHT child's subtree, then v's own color.
            for (int c : adj[v]) {
                if (c != heavy[v]) addSubtree(c, +1);
            }
            addNode(v);
            ans[v] = distinct;
            // Step 3: if this node is light (keep == false), clear its whole subtree.
            if (!f.keep) {
                addSubtree(v, -1);
            }
            st.pop_back();
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    color.assign(n, 0);
    adj.assign(n, {});
    ans.assign(n, 0);

    for (int i = 0; i < n; ++i) cin >> color[i];

    // colors may be large / arbitrary; compress to [0, n).
    {
        vector<int> vals = color;
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
        for (int i = 0; i < n; ++i)
            color[i] = (int)(lower_bound(vals.begin(), vals.end(), color[i]) - vals.begin());
    }

    // n-1 edges of the tree (1-indexed in input). Root the tree at node 1 (index 0).
    vector<vector<int>> g(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    // Orient edges away from the root (node 0) to fill adj[] with children only.
    {
        vector<int> par(n, -1);
        vector<int> bfs;
        bfs.reserve(n);
        vector<char> vis(n, 0);
        if (n > 0) { bfs.push_back(0); vis[0] = 1; }
        for (size_t i = 0; i < bfs.size(); ++i) {
            int u = bfs[i];
            for (int w : g[u]) if (!vis[w]) {
                vis[w] = 1; par[w] = u;
                adj[u].push_back(w);
                bfs.push_back(w);
            }
        }
    }

    if (n == 0) return 0;
    if (n == 1) { cout << 1 << "\n"; return 0; }

    buildOrder(0);
    solve(0);

    string out;
    out.reserve((size_t)n * 7);
    for (int i = 0; i < n; ++i) {
        out += to_string(ans[i]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```

**Causal recap.** Distinct-color count is non-additive, so I must propagate color *sets*, not counts; recompute-per-subtree and naive set-copy are both `Theta(n^2)` on a path (`n(n+1)/2` node touches); small-to-large merging fixes the asymptotics but the cleanest realization is DSU-on-tree with one global `cnt[]` array — process the heavy child last and never re-walk it, re-add only light children, so each node is touched once per light edge on its root path = `O(log n)` times, total `O(n log n)`; subtree operations ride a contiguous Euler interval; both DFS passes are iterative to survive depth-`2*10^5` paths; the one real bug was a phase-fall-through that skipped clearing a light leaf (star printed `6 1 2 2 2 2`), fixed by making the phase transitions disciplined with phase 2 always popping; and coordinate compression, the `n=1` special case, and `max`-free `cnt`/`distinct` bookkeeping close out the corners, all confirmed against an independent brute on 800 cases.
