We are given a tree of $n$ nodes rooted at node $0$, each carrying a color in $[1, n]$, and for every node $v$ we must report the sum of the dominating colors of its subtree, where a color is dominating if no other color appears more often in that subtree (so all colors tied for the maximum frequency count). The task is trivial to state and expensive to do literally. For a fixed $v$ I can walk the whole subtree, tally the colors, find the largest frequency, and sum every color that reaches it; that is plainly correct for one node. But repeating it independently for every node costs $\sum_v \text{size}(v)$, and on a path that sum is $1 + 2 + \dots + n$, which is $\Theta(n^2)$ — far too slow at $n = 10^5$. The waste is visible in the state I keep discarding: after I tally one child's subtree, the parent needs that very same count table plus the parent itself and the other children, yet the literal approach rebuilds the table from zero at every ancestor. That rebuild from scratch is the part that has to go. I cannot simply keep every child's table separately either, since the count array is shared — if two children's tallies sit in it at once they mix and I can no longer tell whose frequency is whose — so naively preserving all children would force me to duplicate large tables or merge many maps.

The method I propose is DSU on tree (small-to-large on the static count table). The single idea that makes it work is to arrange the traversal so that exactly one child's count table survives into its parent, and to choose that surviving child to be the one whose re-walk I most want to avoid: the child with the largest subtree. I call this the heavy child and store it as `pivot_child[u]`; every other child is light, and I pay to re-walk each light subtree by hand. The order at a node $u$ is then forced. I first recurse into every light child in a mode that cleans its contribution back out before returning, so no light sibling can contaminate the next. Then I recurse into the heavy child in a mode that leaves its count table in place. When that returns, the active table holds exactly the heavy child's subtree; I add $u$ itself, then re-walk each light subtree once and add those nodes, and at that instant the active table is precisely the subtree of $u$, so I read $\text{ans}[u]$. Finally, if my caller is not meant to inherit $u$'s table, I walk $u$'s whole subtree and remove it back to empty; otherwise I leave it standing for the parent to inherit. The cleanup is not an afterthought — it is part of the accounting. Fix a node $x$. Each time an ancestor reaches $x$ across a light edge there are at most two extra touches charged to that edge: the light subtree's cleanup may remove $x$, and the ancestor may re-add $x$ when assembling its own table; both are constant work attributable to that single light edge. A larger later clear that contains $x$ is charged to the light edge above it or to the final root cleanup. The decisive geometric fact is that a light edge at least halves the subtree size: if $x$ sits in a light child of a node whose subtree has size $s$, and that light child's subtree has size $s'$, then the preserved heavy child has size at least $s'$, and the two together fit inside the parent, so $2s' \le s$. Hence at most $\log_2 n$ light edges lie on any root-to-node path, every node is touched $O(\log n)$ times, and the whole traversal runs in $O(n \log n)$ time with $O(n)$ memory.

The color query needs incremental state so that finishing a subtree is constant time rather than a full color scan. While adding nodes I maintain $\text{cnt}[c]$, the active frequency of color $c$, together with $\text{max\_freq}$ and $\text{sum\_dom}$, the sum of the colors currently at that maximum. When I add color $c$ and it reaches frequency $f = \text{cnt}[c]$, the update is

$$
(\text{max\_freq}, \text{sum\_dom}) \leftarrow
\begin{cases}
(f,\; c) & f > \text{max\_freq}\quad(\text{$c$ alone at the new max})\\[2pt]
(\text{max\_freq},\; \text{sum\_dom} + c) & f = \text{max\_freq}\quad(\text{$c$ joins the leaders})\\[2pt]
(\text{max\_freq},\; \text{sum\_dom}) & f < \text{max\_freq}\quad(\text{no change}),
\end{cases}
$$

which is $O(1)$ per add. Removal looks harder, because decrementing a color that was tied for the maximum could in general require searching for the next maximum — but I never query during a removal phase. Removals happen only when I clear an entire active subtree back to empty, so `remove(u)` need only decrement $\text{cnt}[\text{color}[u]]$; once the walk finishes I know the summary without any search, namely $\text{max\_freq} = 0$ and $\text{sum\_dom} = 0$. Thus the summary is maintained exactly during add phases, inherited intact when a preserved subtree is inherited, and reset only after the table has been fully cleared. A small sanity check fixes the bookkeeping: a root $r$ with two single-node children $a, b$ and three distinct colors, $\text{pivot\_child}[r] = a$. Solving light child $b$ first with $\text{mode}=\text{False}$ gives $\text{ans}[b] = \text{color}[b]$ then clears it; the heavy child $a$ with $\text{mode}=\text{True}$ gives $\text{ans}[a] = \text{color}[a]$ and stays; back at $r$ I add $r$ and re-add $b$, all three colors sit at frequency $1$ and tie, so $\text{sum\_dom} = \text{color}[a] + \text{color}[r] + \text{color}[b]$ — exactly the right whole-tree answer. The implementation pins this down with one iterative postorder pass that roots the tree at $0$, fills `parent_` and `sz`, and records the maximum-size child in `pivot_child[u]`; `add_subtree` and `clear_subtree` are explicit stack walks under an already-rooted child so they never cross a parent edge; and the dsu-on-tree pass runs light children with $\text{mode}=\text{False}$, the pivot with $\text{mode}=\text{True}$, then adds $u$ and the light subtrees, records the answer, and clears when $\text{mode}$ is false. Because that pass descends the heavy edge to a depth equal to the tree height — which on a path is the full $n$ — I unroll the recursion into an explicit frame stack rather than relying on the call stack, and the root frame is entered with $\text{mode}=\text{False}$ so its final linear cleanup runs only after every answer is already written.

This is CF600E, so the deliverable is one self-contained C++17 program with `int main()`. It reads from stdin the integer $n$, then $n$ colors, then $n-1$ edges (1-based vertices, tree rooted at vertex 1), and prints $n$ space-separated sums of dominating colors, one per vertex, to stdout. The running totals are `long long`, because a subtree of all-distinct colors makes the answer the sum of those colors — for $n = 10^5$ that is about $5 \times 10^9$, well past the 32-bit range.

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
