// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll L, K, W;
    int Q;
    if (!(cin >> L >> K >> Q >> W)) return 0;
    for (int i = 0; i < Q; i++) {
        ll p, c;
        cin >> p >> c;
    }
    // Deliberately infeasible: the very first stake is out of range [0, L].
    cout << (L + 1000000) << "\n";
    for (ll i = 1; i < K; i++) cout << 0 << (i + 1 < K ? ' ' : '\n');
    return 0;
}
