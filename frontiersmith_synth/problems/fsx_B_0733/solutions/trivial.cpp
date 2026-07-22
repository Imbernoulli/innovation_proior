// TIER: trivial
// Do nothing: add zero shortcuts. This is exactly the checker's internal
// baseline B, so this solution reproduces ratio == 0.1 on every test.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long n, m0, t, k, M;
    if (!(cin >> n >> m0 >> t >> k >> M)) return 0;
    for (long long i = 0; i < m0; i++){ long long u, v; cin >> u >> v; }
    for (long long i = 0; i < M; i++){ long long a, b; cin >> a >> b; }
    cout << 0 << "\n";
    return 0;
}
