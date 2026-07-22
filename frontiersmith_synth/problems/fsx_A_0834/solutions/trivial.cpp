// TIER: trivial
// Naive baseline: "patrol the harbor mouth". Only ever considers edges
// leaving node 0 (the canal's single entrance) -- ignores everything else in
// the network, even if the budget allows more. This deliberately never looks
// at bundle structure or downstream value at all.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, K; ll B;
    cin >> n >> m >> K >> B;
    vector<int> u(m), v(m);
    vector<ll> cost(m), cap(m);
    for (int i = 0; i < m; i++) cin >> u[i] >> v[i] >> cost[i] >> cap[i];
    for (int i = 0; i < K; i++) { ll a,b,c,d; cin >> a >> b >> c >> d; }

    vector<int> pick;
    for (int i = 0; i < m && (int)pick.size() < B; i++)
        if (u[i] == 0) pick.push_back(i);

    cout << pick.size() << "\n";
    for (size_t i = 0; i < pick.size(); i++) cout << pick[i] << " \n"[i+1==pick.size()];
    if (pick.empty()) cout << "\n";
    return 0;
}
