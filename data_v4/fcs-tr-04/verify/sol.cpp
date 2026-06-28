#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v;
        cin >> u >> v;            // 1-indexed endpoints of an undirected edge
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    if (n == 1) {                 // single node: distance to everyone (itself) is 0
        cout << 0 << "\n";
        return 0;
    }

    // ---- Pass 1: iterative DFS from root 1 to get parent order, subtree sizes,
    //              and S(root) = sum of depths over all nodes. ----
    vector<int> parent(n + 1, 0);
    vector<int> order;            // nodes in DFS discovery order (root first)
    order.reserve(n);
    vector<long long> size(n + 1, 1);
    vector<long long> ans(n + 1, 0);

    {
        vector<int> st;
        st.reserve(n);
        st.push_back(1);
        parent[1] = 0;
        vector<char> visited(n + 1, 0);
        visited[1] = 1;
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            order.push_back(u);
            for (int w : adj[u]) {
                if (!visited[w]) {
                    visited[w] = 1;
                    parent[w] = u;
                    st.push_back(w);
                }
            }
        }
    }

    // depth of root is 0; depth[child] = depth[parent] + 1, processed in discovery order.
    vector<long long> depth(n + 1, 0);
    long long rootSum = 0;
    for (int idx = 0; idx < (int)order.size(); idx++) {
        int u = order[idx];
        if (u != 1) depth[u] = depth[parent[u]] + 1;
        rootSum += depth[u];
    }

    // subtree sizes: process discovery order in reverse so children precede parents.
    for (int idx = (int)order.size() - 1; idx >= 1; idx--) {
        int u = order[idx];
        size[parent[u]] += size[u];
    }

    ans[1] = rootSum;

    // ---- Pass 2: reroot in discovery order (parent computed before child). ----
    // Moving the root from par to child: the size[child] nodes in child's subtree
    // get one step closer, the other (n - size[child]) get one step farther.
    for (int idx = 1; idx < (int)order.size(); idx++) {
        int u = order[idx];
        int p = parent[u];
        ans[u] = ans[p] + (long long)n - 2LL * size[u];
    }

    string out;
    out.reserve((size_t)n * 12);
    for (int v = 1; v <= n; v++) {
        out += to_string(ans[v]);
        out += '\n';
    }
    cout << out;
    return 0;
}
