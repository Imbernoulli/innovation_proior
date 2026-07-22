// TIER: invalid
// Deliberately infeasible: the FIRST printed "depot" is n+1, one past the
// valid vertex range [1,n], regardless of k (works even when k==1, where a
// merely-duplicated single vertex would still be technically feasible). The
// checker's bounded read ouf.readInt(1,n,"depot") must reject this -> 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long n, m, k;
    if (!(cin >> n >> m >> k)) return 0;
    for (long long i = 0; i < n; i++){ long long p; cin >> p; }
    for (long long i = 0; i < m; i++){ long long u, v; cin >> u >> v; }
    for (long long i = 0; i < k; i++) cout << (n + 1) << (i + 1 < k ? ' ' : '\n');
    return 0;
}
