// TIER: trivial
// Fully-serial schedule: run every leg one after another on a single global
// timeline. Feasible (no two legs ever overlap; precedence trivially holds).
// Makespan = total work = the checker's baseline B  ->  ratio = 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> o(n);
    vector<vector<int>> M(n), D(n);
    for (int j = 0; j < n; j++) {
        scanf("%d", &o[j]);
        M[j].resize(o[j]); D[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) scanf("%d %d", &M[j][k], &D[j][k]);
    }
    long long t = 0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < o[j]; k++) {
            printf("%lld%c", t, k + 1 < o[j] ? ' ' : '\n');
            t += D[j][k];
        }
    }
    return 0;
}
