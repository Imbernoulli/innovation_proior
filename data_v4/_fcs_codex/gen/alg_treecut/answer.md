```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<vector<int>> g(n + 1);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> parent(n + 1, 0), order;
    order.reserve(n);

    stack<int> st;
    st.push(1);
    parent[1] = -1;

    while (!st.empty()) {
        int v = st.top();
        st.pop();
        order.push_back(v);

        for (int to : g[v]) {
            if (to == parent[v]) continue;
            parent[to] = v;
            st.push(to);
        }
    }

    vector<int> subtree(n + 1, 1), leafSubtree(n + 1, 0);
    int totalLeaves = 0;

    for (int v = 1; v <= n; ++v) {
        if ((int)g[v].size() == 1) {
            leafSubtree[v] = 1;
            ++totalLeaves;
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        int v = order[i];
        for (int to : g[v]) {
            if (parent[to] == v) {
                subtree[v] += subtree[to];
                leafSubtree[v] += leafSubtree[to];
            }
        }
    }

    int answer = n;

    auto consider = [&](int k) {
        if (1 <= k && k <= n - 1) {
            answer = min(answer, abs(n - 2 * k));
        }
    };

    for (int v = 2; v <= n; ++v) {
        int s = subtree[v];

        consider(s);
        consider(n - s);

        bool hasMovableLeafInside = leafSubtree[v] - (((int)g[v].size() == 1) ? 1 : 0) > 0;
        bool hasMovableLeafOutside = totalLeaves - leafSubtree[v] > 0;

        if (hasMovableLeafInside) {
            consider(s - 1);
            consider(n - s + 1);
        }

        if (hasMovableLeafOutside) {
            consider(s + 1);
            consider(n - s - 1);
        }
    }

    cout << answer << '\n';
    return 0;
}
```