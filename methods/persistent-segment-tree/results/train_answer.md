We are given a fixed array of $n$ integers and a stream of $q$ queries $(l, r, k)$, each asking for the $k$-th smallest value in the subarray $a[l..r]$ counted with multiplicity. The array never changes after input, so the only honest difficulty is sharing work across queries. Copying out the slice $a[l..r]$, sorting it, and indexing the $k$-th position is correct and even handles duplicates cleanly, but a single wide query costs $O(n \log n)$, and across many overlapping wide queries we re-sort the same elements over and over. Nothing about that approach reuses the fact that the array is fixed. We need to preprocess once and then answer each query by descending through a value-indexed counting structure rather than touching the requested slice at all, all while keeping memory near linear up to logarithmic factors.

The first move is to stop thinking about positions and think about values. Compress the distinct array values into a dense domain $1..\text{len}$ and build a segment tree over that value domain, where each node stores how many inserted elements fall into its value interval. If such a tree already held exactly the multiset $a[l..r]$, the $k$-th smallest is a simple order-statistic descent: at an internal node, let $c$ be the count in the left (lower-value) half; if $k \le c$ the answer lives in the lower half, otherwise it lives in the upper half with rank reduced to $k - c$. The leaf reached is a compressed value index, and $\text{vals}[\text{index}-1]$ recovers the real value. That descent is exactly the inner operation we want — but it only works once the tree contains precisely the elements of the query window, and building such a tree per query would just reintroduce the per-query cost we are trying to avoid.

What rescues it is that a static range is the difference of two prefixes. I propose the persistent segment tree: keep one value-count tree $T_i$ for each prefix $a[1..i]$, so that the count of window elements inside any value interval is the count in $T_r$ minus the count in $T_{l-1}$. Because every $T_i$ shares the same compressed value domain, a node covering a given value interval means the same thing in both trees, so subtracting node counts is meaningful. Crucially we never materialize the difference tree. We run the order-statistic descent on the two roots in lockstep: at each internal node the number of window elements in the lower half is

$$x = \text{sum}[\text{ls}[v]] - \text{sum}[\text{ls}[u]], \qquad u = \text{root}[l-1],\; v = \text{root}[r],$$

and if $k \le x$ we recurse into both left children, otherwise into both right children with $k$ replaced by $k - x$. The orientation matters: the sign must be right-prefix minus left-before-range-prefix, which is why $u$ is the root at index $l-1$ (everything strictly before the window) and $v$ is the root at index $r$ (everything up to and including the window's end); their difference is exactly the window's multiset.

The naive way to realize $n+1$ prefix trees is to store them independently, but a full value tree has $O(\text{len})$ nodes and $\text{len}$ can be as large as $n$, so independent copies cost $O(n \cdot \text{len})$, which is quadratic. The escape is that consecutive prefixes differ by a single insertion: moving from $T_{i-1}$ to $T_i$ inserts only $a[i]$, which bumps exactly one leaf and the $O(\log \text{len})$ ancestors on its root-to-leaf path; every subtree hanging off that path is bit-for-bit identical before and after. So the new tree should own only the changed path and share everything else with the old one. This is incompatible with the usual implicit heap layout where children are at $2\cdot\text{node}$ and $2\cdot\text{node}+1$, because there node identity is welded to array position and you cannot let two trees share a subtree. We therefore give nodes explicit ids and explicit child links: three parallel arrays $\text{sum}$, $\text{ls}$, $\text{rs}$ form a node pool where a node id points to its stored count and its two child ids. To insert into an old node $\text{prev}$, we allocate a fresh node $\text{cur}$ with count $\text{sum}[\text{prev}] + 1$, copy both child links from $\text{prev}$ by default, and then recurse only into the half containing the inserted value, overwriting that one child link with the freshly returned node while the untouched child still points into the old subtree. Because every write lands in a newly allocated node, all older roots stay valid — that is what "persistent" buys us.

This yields one root per prefix: $\text{root}[0]$ is the empty tree and $\text{root}[i]$ is the node returned after inserting $a[i-1]$ into $\text{root}[i-1]$. We do not even build a complete zero tree; node $0$ stands in for every absent zero-count subtree, since $\text{sum}[0]$, $\text{ls}[0]$, $\text{rs}[0]$ are all zero and following missing links consistently reads zeros. Each inserted value creates one node per level plus the leaf, so building is $O(n \log n)$ nodes and time, and each query is a single paired descent of $O(\log n)$.

A small concrete check confirms the orientation. With $a = [5,1,3]$ the compressed values are $[1,3,5]$, so $5\!\to\!3$, $1\!\to\!1$, $3\!\to\!2$. For $k=2$ over $a[1..3]$ we compare $\text{root}[0]$ and $\text{root}[3]$; the window is all three indices and the descent reaches compressed index $2$, value $3$. For $k=1$ over $a[2..3]$ we compare $\text{root}[1]$ and $\text{root}[3]$: in the lower half $[1,2]$ the difference is two elements versus zero, so we go left, and inside $[1,2]$ the left leaf has difference one, returning compressed index $1$, value $1$ — exactly the smallest of $[1,3]$.

```cpp
// Reads: n q, then n array values, then q queries (l, r, k);
// prints, one per line, the k-th smallest value in a[l..r] (1-based l, r, k).
#include <bits/stdc++.h>
using namespace std;

// Persistent (value-indexed count) segment tree stored in parallel arrays.
// One persisted version per array prefix: root[i] holds the multiset a[1..i].
// A query window a[l..r] is the node-wise difference of root[r] and root[l-1].
static vector<long long> tsum;   // count stored at a node
static vector<int> ls, rs;       // child node ids (node 0 is the shared empty node)

static int newNode(long long s, int l, int r) {
    tsum.push_back(s);
    ls.push_back(l);
    rs.push_back(r);
    return (int)tsum.size() - 1;
}

// Insert value-index k into the version rooted at prev; return the new root.
// Copies only the root-to-leaf path; every untouched child stays shared.
static int insertNode(int k, int l, int r, int prev) {
    int cur = newNode(tsum[prev] + 1, ls[prev], rs[prev]);
    if (l == r) return cur;
    int mid = (l + r) >> 1;
    if (k <= mid) ls[cur] = insertNode(k, l, mid, ls[cur]);
    else          rs[cur] = insertNode(k, mid + 1, r, rs[cur]);
    return cur;
}

// u = root[l-1], v = root[r]; window counts are v minus u, node by node.
static int queryNode(int u, int v, int l, int r, long long k) {
    if (l == r) return l;
    int mid = (l + r) >> 1;
    long long x = tsum[ls[v]] - tsum[ls[u]];   // window items in the lower value half
    if (k <= x) return queryNode(ls[u], ls[v], l, mid, k);
    return queryNode(rs[u], rs[v], mid + 1, r, k - x);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];

    // Coordinate compression: dense value domain [1, len].
    vector<long long> vals(a.begin(), a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int len = (int)vals.size();
    auto getid = [&](long long x) {
        return (int)(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
    };

    // Node 0 is the shared empty node; root[0] = empty version.
    tsum.assign(1, 0);
    ls.assign(1, 0);
    rs.assign(1, 0);
    tsum.reserve((size_t)1 + (size_t)n * (33 - __builtin_clz(len > 1 ? len : 1)));
    ls.reserve(tsum.capacity());
    rs.reserve(tsum.capacity());

    vector<int> root(n + 1, 0);
    if (len > 0) {
        for (int i = 1; i <= n; ++i)
            root[i] = insertNode(getid(a[i - 1]), 1, len, root[i - 1]);
    }

    string out;
    for (int i = 0; i < q; ++i) {
        int l, r;
        long long k;
        cin >> l >> r >> k;
        int idx = queryNode(root[l - 1], root[r], 1, len, k);
        out += to_string(vals[idx - 1]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
