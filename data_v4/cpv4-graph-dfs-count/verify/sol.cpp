#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<array<int,2>> adj_[200005]; // {neighbor, edgeId}
int disc[200005], low_[200005], timer_;
bool isBridge[400005];

// Iterative DFS for bridges: explicit stack of frames so a long path cannot
// overflow the call stack at n = 2*10^5. peId = id of the edge used to enter u.
void dfs(int root) {
    // frame: vertex u, parent-edge id, and an index into adj_[u]
    static int stU[200005], stPE[200005], stIt[200005];
    int top = 0;
    stU[top] = root; stPE[top] = -1; stIt[top] = 0;
    disc[root] = low_[root] = ++timer_;
    while (top >= 0) {
        int u = stU[top], peId = stPE[top];
        if (stIt[top] < (int)adj_[u].size()) {
            auto &e = adj_[u][stIt[top]++];
            int v = e[0], id = e[1];
            if (id == peId) continue;          // skip exactly the one parent-edge instance
            if (!disc[v]) {                    // tree edge: descend
                disc[v] = low_[v] = ++timer_;
                ++top;
                stU[top] = v; stPE[top] = id; stIt[top] = 0;
            } else {                           // back edge
                low_[u] = min(low_[u], disc[v]);
            }
        } else {                               // done with u: pop, relax parent
            --top;
            if (top >= 0) {
                int p = stU[top];
                low_[p] = min(low_[p], low_[u]);
                if (low_[u] > disc[p]) isBridge[peId] = true;
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m)) return 0;
    for (int i = 0; i < m; i++) {
        int a, b; cin >> a >> b;
        adj_[a].push_back({b, i});
        if (a != b) adj_[b].push_back({a, i}); // self-loop stored once (never a bridge anyway)
    }
    for (int s = 1; s <= n; s++)
        if (!disc[s]) dfs(s);

    long long bridges = 0;
    for (int i = 0; i < m; i++) if (isBridge[i]) bridges++;
    cout << (long long)m - bridges << "\n";
    return 0;
}
