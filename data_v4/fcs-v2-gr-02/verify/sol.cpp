#include <bits/stdc++.h>
using namespace std;

// Lengauer-Tarjan dominator tree.
// Nodes are 1..n. Source is s. idom[v] = immediate dominator of v (0 if v is the
// source or v is unreachable from s). Output idom[1..n].

static const int MAXN = 200005;

int n, m, s;
vector<int> g[MAXN];   // forward edges
vector<int> rg[MAXN];  // reverse edges (used for the semidominator scan)
vector<int> bucket[MAXN]; // bucket[w] = vertices whose semidominator is w

int dfn[MAXN];     // DFS preorder number of a vertex (0 = unvisited)
int order[MAXN];   // order[i] = vertex with DFS number i
int par[MAXN];     // par[v] = DFS-tree parent of v (by vertex id)
int semi[MAXN];    // semi[v] = DFS number of the semidominator of v
int idom[MAXN];    // immediate dominator (vertex id), filled in two phases
int cnt;           // DFS counter

// Link-eval forest with path compression that tracks the vertex of minimum
// semidominator along the compressed path.
int anc[MAXN];     // ancestor (forest parent) in the link-eval structure
int label[MAXN];   // label[v] = vertex on the path to anc with min semi[]

// Iterative DFS to assign preorder numbers (recursion would overflow the stack
// at n = 2e5).
void dfs() {
    // explicit stack of (vertex, index-into-adjacency)
    static int stk[MAXN];
    static size_t it[MAXN];
    int top = 0;
    stk[top] = s;
    it[s] = 0;
    cnt = 0;
    cnt++;
    dfn[s] = cnt;
    order[cnt] = s;
    semi[s] = cnt;
    label[s] = s;
    while (top >= 0) {
        int u = stk[top];
        if (it[u] < g[u].size()) {
            int v = g[u][it[u]++];
            if (dfn[v] == 0) {
                cnt++;
                dfn[v] = cnt;
                order[cnt] = v;
                semi[v] = cnt;
                label[v] = v;
                par[v] = u;
                ++top;
                stk[top] = v;
                it[v] = 0;
            }
        } else {
            --top;
        }
    }
}

// Compress the path from v to the root of its link-eval tree, keeping label[]
// pointing to the vertex of minimum semi[] encountered. Iterative to avoid
// stack overflow.
void compress(int v) {
    static int path[MAXN];
    int len = 0;
    while (anc[anc[v]] != 0) {
        path[len++] = v;
        v = anc[v];
    }
    // now anc[v] is a root (anc[anc[v]] == 0); v's label is already correct
    for (int i = len - 1; i >= 0; --i) {
        int x = path[i];
        if (semi[label[anc[x]]] < semi[label[x]])
            label[x] = label[anc[x]];
        anc[x] = anc[v];
    }
}

// eval(v): minimum-semi label among the link-eval ancestors of v (v inclusive).
int eval(int v) {
    if (anc[v] == 0) return label[v]; // v is a forest root
    compress(v);
    return label[v];
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> m >> s)) return 0;
    for (int i = 0; i < m; ++i) {
        int a, b;
        cin >> a >> b;
        g[a].push_back(b);
        rg[b].push_back(a);
    }

    for (int v = 1; v <= n; ++v) { dfn[v] = 0; idom[v] = 0; anc[v] = 0; }

    dfs();

    // Process vertices in decreasing DFS order (skip the root order[1] = s).
    for (int i = cnt; i >= 2; --i) {
        int w = order[i];
        // Step 2: compute semidominator of w.
        for (int u : rg[w]) {
            if (dfn[u] == 0) continue;     // u unreachable from s -> ignore
            int t = eval(u);
            if (semi[t] < semi[w]) semi[w] = semi[t];
        }
        bucket[order[semi[w]]].push_back(w);
        // Link w into the forest under its DFS parent.
        anc[w] = par[w];
        // Step 3: process the bucket of w's parent.
        int p = par[w];
        for (int v : bucket[p]) {
            int u = eval(v);
            idom[v] = (semi[u] < semi[v]) ? u : p; // tentative
        }
        bucket[p].clear();
    }

    // Step 4: fill in deferred immediate dominators in DFS order.
    for (int i = 2; i <= cnt; ++i) {
        int w = order[i];
        if (idom[w] != order[semi[w]]) idom[w] = idom[idom[w]];
    }
    idom[s] = 0; // root has no immediate dominator

    // Output idom[1..n]; 0 means "source" or "unreachable".
    for (int v = 1; v <= n; ++v) {
        cout << idom[v];
        cout << (v == n ? '\n' : ' ');
    }
    return 0;
}
