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
