// TIER: invalid
// Deliberately infeasible: claims c = k+1 shortcuts, one more than the budget
// allows. The checker's bounded read of c in [0,k] must reject this -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long n, m0, t, k, M;
    if (!(cin >> n >> m0 >> t >> k >> M)) return 0;
    for (long long i = 0; i < m0; i++){ long long u, v; cin >> u >> v; }
    for (long long i = 0; i < M; i++){ long long a, b; cin >> a >> b; }
    cout << (k + 1) << "\n";
    for (long long i = 0; i < k + 1; i++) cout << 1 << " ";
    cout << "\n";
    return 0;
}
