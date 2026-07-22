// TIER: greedy
// The obvious single-pass approach: sort ALL candidate pipes (every batch) by flow value
// descending, and greedily accept a pipe iff it (a) does not close a cycle among pipes
// already accepted, and (b) does not exceed its batch's remaining cap. This is exactly
// what a strong coder writes first -- and it is well known that this "respects both
// constraints locally" heuristic is NOT optimal for a genuine two-matroid intersection
// (unlike a single matroid, where weight-sorted greedy alone would be exact).
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
    vector<int> ids(E);
    iota(ids.begin(), ids.end(), 0);
    // sort by weight desc; ties broken by input index asc for determinism
    sort(ids.begin(), ids.end(), [&](int a, int c) {
        if (ew[a] != ew[c]) return ew[a] > ew[c];
        return a < c;
    });
    vector<ll> cnt(K + 1, 0);
    vector<int> chosen;
    for (int id : ids) {
        if (cnt[eb[id]] >= cap_[eb[id]]) continue;
        if (unite(eu[id], ev[id])) {
            cnt[eb[id]]++;
            chosen.push_back(id + 1);
        }
    }

    cout << chosen.size() << "\n";
    for (int x : chosen) cout << x << " ";
    cout << "\n";
    return 0;
}
