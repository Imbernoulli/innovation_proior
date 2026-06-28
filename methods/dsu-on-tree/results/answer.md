# DSU on tree for dominating colors

## Method

For each node `u`, keep the color counts for one child subtree alive and rebuild
only the other child subtrees around it. The child whose subtree is largest is
stored as `pivot_child[u]`; all other children are processed, answered, and
cleared before the preserved child is processed. Then `u` and the cleared child
subtrees are added back, so the active count table becomes exactly the subtree
of `u`.

The query state is maintained during additions:

- `cnt[c]`: current active frequency of color `c`
- `max_freq`: maximum active frequency
- `sum_dom`: sum of colors whose active frequency equals `max_freq`

When a color `c` is added and reaches frequency `f`, reset the answer state if
`f > max_freq`, append `c` if `f == max_freq`, and otherwise leave it unchanged.
Removal is only used while clearing a whole active subtree, and no answer is read
mid-clear, so removal only decrements `cnt`; after the clear finishes,
`max_freq` and `sum_dom` are reset to zero.

## Complexity

The amortized bound counts cleanup as well as re-addition. For a node `x`, every
light edge above it can cause a constant number of extra touches: a cleanup
removal from the light subtree and a later re-add when the ancestor assembles its
own active table. A light edge at least halves subtree size, so there are at most
`log_2 n` light edges on any root-to-node path. Including the local add and the
final root cleanup, every node is touched `O(log n)` times. The total running
time is `O(n log n)`, and the memory use is `O(n)`.

## Code

This is CF600E. The program reads from stdin the integer `n`, then `n` colors, then
`n - 1` edges (1-based vertices, tree rooted at vertex 1), and prints `n`
space-separated sums of dominating colors, one per vertex, to stdout. Sums use
`long long` because a subtree of all-distinct colors can make the answer exceed the
32-bit range (e.g. `sum(1..10^5)` is about `5 * 10^9`). The heavy-edge recursion is
unrolled with an explicit frame stack so a path of `n = 10^5` nodes cannot overflow
the call stack.

```cpp
// CF600E: reads n, then n colors, then n-1 edges (1-based, tree rooted at vertex 1)
// from stdin; prints n space-separated sums of dominating colors per vertex to stdout.
#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> color;
vector<vector<int>> g;
vector<int> sz, pivot_child, parent_;
vector<long long> cnt;        // cnt[c] = active occurrences of color c
long long max_freq = 0;       // current maximum active frequency
long long sum_dom = 0;        // sum of colors achieving max_freq
vector<long long> ans;

static inline void add_node(int u) {
    int c = color[u];
    cnt[c] += 1;
    long long f = cnt[c];
    if (f > max_freq) { max_freq = f; sum_dom = c; }
    else if (f == max_freq) { sum_dom += c; }
}

static inline void remove_node(int u) {
    cnt[color[u]] -= 1;
}

// iterative subtree add/clear over an explicit child stack (never crosses parent edge)
static void add_subtree(int root) {
    vector<int> st{root};
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        add_node(u);
        for (int w : g[u]) if (w != parent_[u]) st.push_back(w);
    }
}
static void clear_subtree(int root) {
    vector<int> st{root};
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        remove_node(u);
        for (int w : g[u]) if (w != parent_[u]) st.push_back(w);
    }
}

// main dsu-on-tree pass; recursion descends only the heavy edge so its depth is the
// tree height. We unroll it with an explicit frame stack to avoid stack overflow on
// a path of n = 1e5 nodes (where the heavy-edge depth is the full n).
static void run() {
    // mode==false: clean own subtree before returning; mode==true: leave it for parent
    struct Frame { int u; bool mode; int phase; };
    vector<Frame> fs;
    fs.push_back({0, false, 0});
    while (!fs.empty()) {
        Frame &fr = fs.back();
        int u = fr.u;
        if (fr.phase == 0) {
            // light children first (cleared afterwards); push them to be processed
            fr.phase = 1;
            for (int w : g[u]) if (w != parent_[u] && w != pivot_child[u])
                fs.push_back({w, false, 0});
        } else if (fr.phase == 1) {
            // heavy child last; keep its accumulated counts in place
            fr.phase = 2;
            if (pivot_child[u] != -1)
                fs.push_back({pivot_child[u], true, 0});
        } else {
            // fold in u itself and the light subtrees (heavy counts already present)
            add_node(u);
            for (int w : g[u]) if (w != parent_[u] && w != pivot_child[u])
                add_subtree(w);
            ans[u] = sum_dom;
            if (!fr.mode) {
                clear_subtree(u);
                max_freq = 0;
                sum_dom = 0;
            }
            fs.pop_back();
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n)) return 0;
    color.assign(n, 0);
    g.assign(n, {});
    for (int i = 0; i < n; ++i) cin >> color[i];
    for (int i = 0; i < n - 1; ++i) {
        int x, y; cin >> x >> y;
        --x; --y;
        g[x].push_back(y);
        g[y].push_back(x);
    }

    sz.assign(n, 1);
    pivot_child.assign(n, -1);
    parent_.assign(n, -1);
    cnt.assign(n + 1, 0);
    ans.assign(n, 0);

    // one iterative post-order pass: roots tree at 0, fills parent_, size,
    // and records the maximum-size child in pivot_child[u]
    vector<array<int,3>> stack0;     // {u, p, processed}
    stack0.push_back({0, -1, 0});
    while (!stack0.empty()) {
        auto [u, p, processed] = stack0.back(); stack0.pop_back();
        if (processed) {
            int best = 0;
            for (int w : g[u]) if (w != p) {
                sz[u] += sz[w];
                if (sz[w] > best) { best = sz[w]; pivot_child[u] = w; }
            }
            continue;
        }
        parent_[u] = p;
        stack0.push_back({u, p, 1});
        for (int w : g[u]) if (w != p) stack0.push_back({w, u, 0});
    }

    run();

    // print n space-separated answers on one line
    string out;
    out.reserve(n * 7);
    for (int i = 0; i < n; ++i) {
        if (i) out.push_back(' ');
        out += to_string(ans[i]);
    }
    out.push_back('\n');
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
```
