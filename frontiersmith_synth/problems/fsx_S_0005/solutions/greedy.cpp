// TIER: greedy
// One-pass SPT (shortest-processing-time) list scheduling: repeatedly dispatch
// the ready leg with the smallest duration into its earliest feasible slot.
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
    vector<vector<long long>> st(n);
    for (int j = 0; j < n; j++) st[j].assign(o[j], 0);
    vector<int> nx(n, 0);
    vector<long long> jobReady(n, 0), machFree(m, 0);
    int total = 0;
    for (int j = 0; j < n; j++) total += o[j];

    for (int c = 0; c < total; c++) {
        int bj = -1; long long bd = LLONG_MAX;
        for (int j = 0; j < n; j++) {
            if (nx[j] < o[j]) {
                long long d = D[j][nx[j]];
                if (d < bd || (d == bd && (bj < 0 || j < bj))) { bd = d; bj = j; }
            }
        }
        int j = bj, k = nx[j], mch = M[j][k];
        long long s = max(jobReady[j], machFree[mch]);
        st[j][k] = s;
        jobReady[j] = s + D[j][k];
        machFree[mch] = s + D[j][k];
        nx[j]++;
    }
    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", st[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
