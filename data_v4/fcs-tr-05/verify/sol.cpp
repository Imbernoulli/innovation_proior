#include <bits/stdc++.h>
using namespace std;

// ---------- Heavy-Light Decomposition + segment tree with lazy XOR ----------
// Each node has a value < 2^BITS. We support:
//   path XOR-assign: v[w] ^= x for every node w on path(u,v)
//   path SUM query : sum of v[w] for every node w on path(u,v)
// XOR is bit-independent, so the segment tree stores, per node-range, the count
// of set bits in each bit-plane. XOR by mask x flips bit b on the whole range:
//   if bit b of x is set, count_b -> (range_length - count_b).
// The SUM contributed by bit b is count_b * 2^b. Lazy = accumulated XOR mask.

static const int BITS = 30;

int N;                       // number of tree nodes
vector<int> adj[200005];
int valNode[200005];         // initial value of each node

int parent_[200005], depth_[200005], heavy_[200005], sizeSub[200005];
int chainHead[200005], posIn[200005];   // HLD: head of chain, position in base array
int basePos = 0;
int baseVal[200005];         // value indexed by position (posIn order)

// ---- iterative subtree sizes + heavy child (avoid recursion stack overflow) ----
void computeSizes(int root) {
    // order: get a DFS order, then process in reverse to accumulate sizes
    vector<int> order;
    order.reserve(N);
    vector<int> st;
    st.reserve(N);
    parent_[root] = 0;       // 0 used as "no parent" sentinel (nodes are 1..N)
    depth_[root] = 0;
    st.push_back(root);
    vector<char> visited(N + 1, 0);
    // produce a parent/order via explicit stack
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        order.push_back(u);
        for (int w : adj[u]) {
            if (w == parent_[u]) continue;
            parent_[w] = u;
            depth_[w] = depth_[u] + 1;
            st.push_back(w);
        }
    }
    for (int i = 1; i <= N; i++) { sizeSub[i] = 1; heavy_[i] = 0; }
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int u = order[i];
        int best = -1, bestChild = 0;
        for (int w : adj[u]) {
            if (w == parent_[u]) continue;
            sizeSub[u] += sizeSub[w];
            if (sizeSub[w] > best) { best = sizeSub[w]; bestChild = w; }
        }
        heavy_[u] = bestChild;
    }
}

// ---- assign chain heads and positions (iterative HLD decomposition) ----
void decompose(int root) {
    // We must process heavy chains top-down. Use an explicit stack of (node, head).
    basePos = 0;
    vector<pair<int,int>> st; // (node, head)
    st.push_back({root, root});
    // To get correct ordering along a heavy chain (contiguous positions), we
    // walk each chain fully before branching off light children.
    while (!st.empty()) {
        auto [u, head] = st.back(); st.pop_back();
        // walk the heavy chain starting at u
        int cur = u;
        int curHead = head;
        while (cur != 0) {
            chainHead[cur] = curHead;
            posIn[cur] = basePos;
            baseVal[basePos] = valNode[cur];
            basePos++;
            // push light children as new chains (each starts its own head)
            for (int w : adj[cur]) {
                if (w == parent_[cur] || w == heavy_[cur]) continue;
                st.push_back({w, w});
            }
            cur = heavy_[cur];
        }
    }
}

// ---------- segment tree ----------
struct SegTree {
    int n;
    // cnt[b][node] = number of set bits at bit-plane b within the node's range
    // stored as cnt[node*BITS + b] for cache locality
    vector<int> cnt;     // size (2n) * BITS? we use array of long long sums instead
    vector<long long> sumv;   // sum over the range
    vector<int> lazy;    // pending XOR mask
    vector<int> cntBit;  // cntBit[node*BITS + b]
    vector<int> segLen;  // length of each node's range

    void init(int n_, int *vals) {
        n = n_;
        sumv.assign(2 * n, 0);
        lazy.assign(2 * n, 0);
        cntBit.assign(2 * n * BITS, 0);
        segLen.assign(2 * n, 0);
        // leaves
        for (int i = 0; i < n; i++) {
            int node = n + i;
            segLen[node] = 1;
            long long v = vals[i];
            sumv[node] = v;
            for (int b = 0; b < BITS; b++)
                if ((v >> b) & 1) cntBit[node * BITS + b] = 1;
        }
        for (int i = n - 1; i >= 1; i--) {
            segLen[i] = segLen[2*i] + segLen[2*i+1];
            sumv[i] = sumv[2*i] + sumv[2*i+1];
            for (int b = 0; b < BITS; b++)
                cntBit[i * BITS + b] = cntBit[2*i*BITS + b] + cntBit[(2*i+1)*BITS + b];
        }
    }

    inline void applyXor(int node, int x) {
        if (x == 0) return;
        long long delta = 0;
        int base = node * BITS;
        for (int b = 0; b < BITS; b++) {
            if ((x >> b) & 1) {
                int newc = segLen[node] - cntBit[base + b];
                delta += (long long)(newc - cntBit[base + b]) << b;
                cntBit[base + b] = newc;
            }
        }
        sumv[node] += delta;
        lazy[node] ^= x;
    }

    inline void pushDown(int node) {
        if (lazy[node]) {
            applyXor(2*node, lazy[node]);
            applyXor(2*node+1, lazy[node]);
            lazy[node] = 0;
        }
    }

    // recursive update/query over [l,r] using a 1-based node layout with explicit ranges.
    // We use a recursive helper over the implicit segment tree on positions [0, n).
    void updateRange(int node, int nodeL, int nodeR, int l, int r, int x) {
        if (r < nodeL || nodeR < l) return;
        if (l <= nodeL && nodeR <= r) { applyXor(node, x); return; }
        pushDown(node);
        int mid = (nodeL + nodeR) >> 1;
        updateRange(2*node, nodeL, mid, l, r, x);
        updateRange(2*node+1, mid+1, nodeR, l, r, x);
        sumv[node] = sumv[2*node] + sumv[2*node+1];
        int base = node * BITS;
        for (int b = 0; b < BITS; b++)
            cntBit[base + b] = cntBit[2*node*BITS + b] + cntBit[(2*node+1)*BITS + b];
    }

    long long queryRange(int node, int nodeL, int nodeR, int l, int r) {
        if (r < nodeL || nodeR < l) return 0;
        if (l <= nodeL && nodeR <= r) return sumv[node];
        pushDown(node);
        int mid = (nodeL + nodeR) >> 1;
        return queryRange(2*node, nodeL, mid, l, r)
             + queryRange(2*node+1, mid+1, nodeR, l, r);
    }
} seg;

int segN;   // padded size = smallest power of two >= N (for clean recursion)

void segUpdate(int l, int r, int x) { seg.updateRange(1, 0, segN - 1, l, r, x); }
long long segQuery(int l, int r)    { return seg.queryRange(1, 0, segN - 1, l, r); }

// ---------- path operations via HLD ----------
void pathXor(int u, int v, int x) {
    while (chainHead[u] != chainHead[v]) {
        if (depth_[chainHead[u]] < depth_[chainHead[v]]) swap(u, v);
        int head = chainHead[u];
        segUpdate(posIn[head], posIn[u], x);
        u = parent_[head];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    segUpdate(posIn[u], posIn[v], x);
}

long long pathSum(int u, int v) {
    long long res = 0;
    while (chainHead[u] != chainHead[v]) {
        if (depth_[chainHead[u]] < depth_[chainHead[v]]) swap(u, v);
        int head = chainHead[u];
        res += segQuery(posIn[head], posIn[u]);
        u = parent_[head];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    res += segQuery(posIn[u], posIn[v]);
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> N >> q)) return 0;
    for (int i = 1; i <= N; i++) cin >> valNode[i];
    for (int i = 0; i < N - 1; i++) {
        int a, b; cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    computeSizes(1);
    decompose(1);

    // pad base array to power of two for the iterative-leaf segment tree
    segN = 1;
    while (segN < N) segN <<= 1;
    static int padded[1 << 19];
    for (int i = 0; i < segN; i++) padded[i] = (i < N) ? baseVal[i] : 0;
    seg.init(segN, padded);

    string out;
    out.reserve(1 << 20);
    for (int i = 0; i < q; i++) {
        int type; cin >> type;
        if (type == 1) {
            int u, v, x; cin >> u >> v >> x;
            pathXor(u, v, x);
        } else {
            int u, v; cin >> u >> v;
            out += to_string(pathSum(u, v));
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
