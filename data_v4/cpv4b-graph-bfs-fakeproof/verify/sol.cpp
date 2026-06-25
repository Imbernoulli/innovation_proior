#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // BFS from node 1 over the unweighted graph.
    const int INF = -1;
    vector<int> d(n + 1, INF);
    queue<int> q;
    d[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int x = q.front();
        q.pop();
        for (int y : adj[x]) {
            if (d[y] == INF) {
                d[y] = d[x] + 1;
                q.push(y);
            }
        }
    }

    // For each bit position b, cnt1[b] = number of nodes whose distance has bit b set.
    // The number of UNORDERED pairs {u,v} (u != v) whose distances differ at bit b is
    // cnt0[b] * cnt1[b], with cnt0[b] = n - cnt1[b]. Summing popcount(d[u] XOR d[v])
    // over all pairs equals sum over bits b of cnt0[b] * cnt1[b].
    // Distances are < n <= 2*10^5 < 2^18, so 18 bits suffice, but we use 20 to be safe.
    const int BITS = 20;
    vector<long long> cnt1(BITS, 0);
    for (int v = 1; v <= n; v++) {
        int dv = d[v]; // every node is reachable (graph is connected), dv >= 0
        for (int b = 0; b < BITS; b++) {
            if ((dv >> b) & 1) cnt1[b]++;
        }
    }

    long long answer = 0;
    for (int b = 0; b < BITS; b++) {
        long long c1 = cnt1[b];
        long long c0 = (long long)n - c1;
        answer += c0 * c1;
    }

    cout << answer << "\n";
    return 0;
}
