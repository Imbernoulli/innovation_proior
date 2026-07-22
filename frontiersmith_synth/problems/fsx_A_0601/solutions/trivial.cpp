// TIER: trivial
// Reproduces the checker's baseline reference cut: delete the k least-central
// (smallest endpoint degree-product) edges lying outside a BFS spanning tree.
// This is the "cut the least-important links" recipe -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m; long long k;
    scanf("%d %d %lld", &n, &m, &k);
    vector<int> eu(m), ev(m), deg(n, 0);
    vector<vector<pair<int,int>>> tadj(n);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v); u--; v--;
        if (u > v) swap(u, v);
        eu[i] = u; ev[i] = v; deg[u]++; deg[v]++;
        tadj[u].push_back({v, i});
        tadj[v].push_back({u, i});
    }
    // BFS spanning tree in edge-index order (identical to the checker).
    vector<char> intree(m, 0), seen(n, 0);
    for (int s = 0; s < n; s++) {
        if (seen[s]) continue;
        seen[s] = 1; queue<int> q; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &pr : tadj[u])
                if (!seen[pr.first]) { seen[pr.first] = 1; intree[pr.second] = 1; q.push(pr.first); }
        }
    }
    vector<int> cand;
    for (int i = 0; i < m; i++) if (!intree[i]) cand.push_back(i);
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        long long pa = (long long)deg[eu[a]] * deg[ev[a]];
        long long pb = (long long)deg[eu[b]] * deg[ev[b]];
        if (pa != pb) return pa < pb;
        return a < b;
    });
    long long take = min<long long>(k, (long long)cand.size());
    for (long long j = 0; j < take; j++)
        printf("%d %d\n", eu[cand[j]] + 1, ev[cand[j]] + 1);
    return 0;
}
