// TIER: trivial
// Do-nothing baseline: depots evenly spaced by RAW vertex id number
// (idx = 1 + floor(i*n/k)), completely oblivious to population or road
// structure. This exactly reproduces one of the checker's own two reference
// constructions, so it scores at or just under the checker's baseline ratio
// (~0.1) by construction.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long n, m, k;
    cin >> n >> m >> k;
    for (long long i = 0; i < n; i++){ long long p; cin >> p; }
    for (long long i = 0; i < m; i++){ long long u, v; cin >> u >> v; }
    vector<char> used(n + 1, 0);
    for (long long i = 0; i < k; i++){
        long long idx = 1 + (i * n) / max(1LL, k);
        if (idx < 1) idx = 1; if (idx > n) idx = n;
        while (used[idx]) idx = (idx % n) + 1;
        used[idx] = 1;
        cout << idx << (i + 1 < k ? ' ' : '\n');
    }
    return 0;
}
