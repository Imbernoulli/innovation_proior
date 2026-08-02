// TIER: trivial
// Do nothing: spend zero firings. This is exactly the checker's internal
// baseline B, so it must score ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll n, m, K;
    cin >> n >> m >> K;
    for (ll i = 0; i < n; i++) cout << 0 << " \n"[i + 1 == n];
    return 0;
}
