// TIER: trivial
// Fully-serial timetable: every green window back-to-back, never overlapping.
// Makespan = sum of all green windows = the checker's baseline B  =>  ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int M, J;
    if (scanf("%d %d", &M, &J) != 2) return 0;
    vector<int> L(J);
    vector<vector<int>> mach(J), p(J);
    for (int j = 0; j < J; j++) {
        scanf("%d", &L[j]);
        mach[j].resize(L[j]);
        p[j].resize(L[j]);
        for (int i = 0; i < L[j]; i++) scanf("%d %d", &mach[j][i], &p[j][i]);
    }
    ll clk = 0;
    for (int j = 0; j < J; j++) {
        for (int i = 0; i < L[j]; i++) {
            printf("%lld ", clk);
            clk += p[j][i];
        }
        printf("\n");
    }
    return 0;
}
