// TIER: trivial
// Standard-stock-only baseline: build the maximum-weight spanning forest using ONLY
// batch-1 ("standard stock", unlimited) pipes. Never touches a premium/capped batch.
// This reproduces the checker's own reference construction B exactly (ratio ~0.1).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int par[100005], rnk_[100005];
int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
bool unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return false;
    if (rnk_[a] < rnk_[b]) swap(a, b);
    par[b] = a;
    if (rnk_[a] == rnk_[b]) rnk_[a]++;
    return true;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int V, E, K;
    cin >> V >> E >> K;
    vector<ll> cap_(K + 1);
    for (int b = 1; b <= K; b++) cin >> cap_[b];
    vector<int> eu(E), ev(E), eb(E);
    vector<ll> ew(E);
    for (int i = 0; i < E; i++) cin >> eu[i] >> ev[i] >> ew[i] >> eb[i];

    for (int i = 1; i <= V; i++) { par[i] = i; rnk_[i] = 0; }
    vector<int> ids;
    for (int i = 0; i < E; i++) if (eb[i] == 1) ids.push_back(i);
    sort(ids.begin(), ids.end(), [&](int a, int c) { return ew[a] > ew[c]; });
    vector<int> chosen;
    for (int id : ids) if (unite(eu[id], ev[id])) chosen.push_back(id + 1);

    cout << chosen.size() << "\n";
    for (int x : chosen) cout << x << " ";
    cout << "\n";
    return 0;
}
