#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // color[i] in {0,1}; -1 = unassigned. BFS each component, 2-color by layer parity.
    vector<int> color(n + 1, -1);
    bool ok = true;
    vector<int> q;
    q.reserve(n);
    for (int s = 1; s <= n && ok; s++) {
        if (color[s] != -1) continue;
        color[s] = 0;
        q.clear();
        q.push_back(s);
        size_t head = 0;
        while (head < q.size()) {
            int u = q[head++];
            int cu = color[u];
            for (int w : adj[u]) {
                if (color[w] == -1) {
                    color[w] = cu ^ 1;
                    q.push_back(w);
                } else if (color[w] == cu) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }

    if (!ok) {
        cout << -1 << "\n";
        return 0;
    }

    string out;
    out.reserve(2 * n);
    for (int i = 1; i <= n; i++) {
        out += char('1' + color[i]); // emit '1' or '2'
        out += (i == n ? '\n' : ' ');
    }
    if (n == 0) out = "\n";
    cout << out;
    return 0;
}
