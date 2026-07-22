// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Do-nothing baseline: every ratio at its window minimum, every index 0 (silent
// modulation -- only the bare carrier sounds). Exactly the checker's baseline B.
int main() {
    int K, H, rc;
    double lambda;
    cin >> K >> H >> rc >> lambda;
    vector<int> Rlo(K), Rhi(K);
    vector<double> Cmax(K);
    for (int i = 0; i < K; i++) cin >> Rlo[i] >> Rhi[i] >> Cmax[i];
    vector<double> T(H + 1);
    for (int h = 1; h <= H; h++) cin >> T[h];

    for (int i = 0; i < K; i++) printf("%d %.6f\n", Rlo[i], 0.0);
    return 0;
}
