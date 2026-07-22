// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Do-nothing seating: metronome k stays at seat k. This is exactly the
// construction the checker uses for its internal baseline B.
int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    for (int i = 0; i < m; i++) { int u, v; cin >> u >> v; }
    for (int i = 0; i < n; i++) { long long f; cin >> f; }
    for (int i = 1; i <= n; i++) cout << i << (i < n ? ' ' : '\n');
    return 0;
}
