// TIER: invalid
// Deliberately infeasible: the very first setup places a knife outside [1, W-1]
// (and, independently, its motion from the declared start position S would also
// exceed the bound m for any reasonable instance). Must score Ratio: 0.0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll W, m; int K;
    cin >> W >> K >> m;
    vector<ll> S(K);
    for (int j = 0; j < K; j++) cin >> S[j];
    int n; cin >> n;
    for (int i = 0; i < n; i++) { ll a, b; cin >> a >> b; }
    ll maxSetups, setupCost, penalty;
    cin >> maxSetups >> setupCost >> penalty;

    cout << 1 << "\n";
    for (int j = 0; j < K - 1; j++) cout << (j + 1) << " ";
    cout << (W + 1000000) << " " << 1 << "\n";   // out of [1, W-1] range -> guaranteed WA
    return 0;
}
