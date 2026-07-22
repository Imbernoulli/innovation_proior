// TIER: trivial
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
    // Ignore the trace entirely: the textbook "one stake every L/(K+1)
    // meters" uniform layout (this is exactly the checker's baseline B).
    for (ll i = 1; i <= K; i++)
        cout << (L * i / (K + 1)) << (i < K ? ' ' : '\n');
    return 0;
}
