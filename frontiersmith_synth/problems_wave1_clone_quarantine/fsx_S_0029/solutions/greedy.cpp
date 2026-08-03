// TIER: greedy
// Non-delay list scheduling with a shortest-processing-time (SPT) priority: repeatedly
// take the ready segment with the smallest earliest feasible start (ties -> shorter
// duration, then corridor index) and place it there. One pass, no restarts.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<int> o(n);
    vector<long long> w(n);
    vector<vector<int>> mach(n);
    vector<vector<long long>> dur(n);
    for (int j = 0; j < n; j++) {
        scanf("%d %lld", &o[j], &w[j]);
        mach[j].resize(o[j]);
        dur[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++)
            scanf("%d %lld", &mach[j][k], &dur[j][k]);
    }

    vector<long long> machFree(m, 0), jobAvail(n, 0);
    vector<int> next(n, 0);                 // next unscheduled segment of each corridor
    vector<vector<long long>> start(n);
    for (int j = 0; j < n; j++) start[j].assign(o[j], 0);

    int total = 0;
    for (int j = 0; j < n; j++) total += o[j];

    for (int done = 0; done < total; done++) {
        long long bestStart = LLONG_MAX, bestDur = LLONG_MAX;
        int bj = -1;
        for (int j = 0; j < n; j++) {
            if (next[j] >= o[j]) continue;   // corridor finished
            int k = next[j];
            int c = mach[j][k];
            long long st = max(jobAvail[j], machFree[c]);
            if (st < bestStart || (st == bestStart && dur[j][k] < bestDur) ||
                (st == bestStart && dur[j][k] == bestDur && (bj < 0 || j < bj))) {
                bestStart = st; bestDur = dur[j][k]; bj = j;
            }
        }
        int k = next[bj];
        int c = mach[bj][k];
        start[bj][k] = bestStart;
        long long e = bestStart + dur[bj][k];
        machFree[c] = e;
        jobAvail[bj] = e;
        next[bj]++;
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", start[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
