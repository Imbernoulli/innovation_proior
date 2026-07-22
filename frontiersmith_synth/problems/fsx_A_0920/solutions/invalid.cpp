// TIER: invalid
// Deliberately infeasible: claims to route every packet over edge id M,
// which is one past the last valid edge index -- must be rejected and
// score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, K; ll T;
    cin >> N >> M >> K >> T;
    for (int i = 0; i < M; i++){ ll u, v, d; cin >> u >> v >> d; }
    for (int k = 0; k < K; k++){ ll s, t, r, d, v; cin >> s >> t >> r >> d >> v; }
    for (int k = 0; k < K; k++) cout << 1 << ' ' << 0 << ' ' << M << '\n';
    return 0;
}
