// TIER: greedy
// The obvious first pass: guess a "reasonable" class count from graph size alone
// (k = round(sqrt(N))), cluster vertices by BFS visiting order into equal-size chunks
// (structure-blind, ignores any voltage/group pattern), then read off the most frequent
// (class,class,offset) triples as base edges. No search over k, no attempt to align
// sheets with any hidden periodicity.
#include <bits/stdc++.h>
using namespace std;

long long canonKey(int ca, int cb, int delta, int n) {
    if (ca > cb) { swap(ca, cb); delta = (n - delta) % n; }
    else if (ca == cb) { int alt = (n - delta) % n; delta = min(delta, alt); }
    return ((long long)ca * 1001 + cb) * 1001 + delta;
}
void decodeKey(long long key, int& ca, int& cb, int& g) {
    g = (int)(key % 1001); key /= 1001;
    cb = (int)(key % 1001); key /= 1001;
    ca = (int)key;
}

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    vector<int> EA(M), EB(M);
    vector<vector<int>> adj(N);
    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        u--; v--;
        EA[i] = u; EB[i] = v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int k = (int)llround(sqrt((double)N));
    k = max(2, min(20, k));
    int n = (N + k - 1) / k;

    // BFS order from vertex 0, continuing into any unvisited component.
    vector<int> order;
    order.reserve(N);
    vector<char> vis(N, 0);
    for (int s = 0; s < N; s++) {
        if (vis[s]) continue;
        queue<int> q;
        q.push(s); vis[s] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            for (int v : adj[u]) if (!vis[v]) { vis[v] = 1; q.push(v); }
        }
    }

    vector<int> cls(N), sheet(N);
    for (int t = 0; t < N; t++) {
        int v = order[t];
        cls[v] = t / n;
        sheet[v] = t % n;
    }

    unordered_map<long long, int> freq;
    for (int i = 0; i < M; i++) {
        int a = EA[i], b = EB[i];
        int delta = ((sheet[b] - sheet[a]) % n + n) % n;
        freq[canonKey(cls[a], cls[b], delta, n)]++;
    }
    vector<pair<int,long long>> byCount;
    byCount.reserve(freq.size());
    for (auto& kv : freq) byCount.push_back({kv.second, kv.first});
    sort(byCount.begin(), byCount.end(), greater<pair<int,long long>>());
    int take = min((int)byCount.size(), 64);

    printf("%d %d\n", k, n);
    for (int i = 0; i < N; i++) printf("%d %d\n", cls[i], sheet[i]);
    printf("%d\n", take);
    for (int i = 0; i < take; i++) {
        int ca, cb, g;
        decodeKey(byCount[i].second, ca, cb, g);
        printf("%d %d %d\n", ca, cb, g);
    }
    return 0;
}
