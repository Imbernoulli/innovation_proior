// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: ratio far outside its window and a negative index
// (violates both the window bound and the [0,Cmax] index bound). Must score 0.
int main() {
    int K, H, rc;
    double lambda;
    cin >> K >> H >> rc >> lambda;
    vector<int> Rlo(K), Rhi(K);
    vector<double> Cmax(K);
    for (int i = 0; i < K; i++) cin >> Rlo[i] >> Rhi[i] >> Cmax[i];
    vector<double> T(H + 1);
    for (int h = 1; h <= H; h++) cin >> T[h];

    for (int i = 0; i < K; i++)
        printf("%d %.6f\n", Rhi[i] + 1000000, -5.0);
    return 0;
}
