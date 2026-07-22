// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: report one more remaining ligament than the strip
// actually has (r_1 = k+1), violating the upper bound of every column's
// feasible range [S,k]. The checker's bounded readInt(S,k,...) must reject it.
int main() {
    int m, k, S;
    ll M, C;
    cin >> m >> k >> S >> M >> C;
    vector<ll> t(m);
    for (auto &x : t) cin >> x;
    cout << (k + 1);
    for (int i = 1; i < m; i++) cout << ' ' << k;
    cout << '\n';
    return 0;
}
