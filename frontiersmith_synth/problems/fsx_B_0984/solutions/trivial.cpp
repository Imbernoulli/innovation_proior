// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Do nothing: no ligaments slit anywhere. Always feasible (cost 0 <= C, and
// r=k is within [S,k]). This is exactly the checker's internal baseline B.
int main() {
    int m, k, S;
    ll M, C;
    cin >> m >> k >> S >> M >> C;
    vector<ll> t(m);
    for (auto &x : t) cin >> x;
    for (int i = 0; i < m; i++) cout << k << (i + 1 == m ? '\n' : ' ');
    return 0;
}
