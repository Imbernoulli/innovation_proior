// TIER: trivial
// Cut nothing. Feasible (cost 0, s-t untouched) -> effective resistance == baseline -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long n, m, s, t, B;
    if (!(cin >> n >> m >> s >> t >> B)) return 0;
    for (int i = 0; i < m; i++) { long long u, v, r, c; cin >> u >> v >> r >> c; }
    printf("0\n\n");
    return 0;
}
