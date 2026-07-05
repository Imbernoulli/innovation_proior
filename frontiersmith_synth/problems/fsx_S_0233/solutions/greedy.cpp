// TIER: greedy
// Earliest-completion-time (ECT) list scheduling, APPEND-ONLY: each dispatched step is
// placed at max(job_ready, resource_last_end) -- no idle-gap filling. Repeatedly pick the
// ready step whose completion time is smallest. Beats serial via cross-telescope parallelism.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> r(n);
    vector<int> o(n);
    vector<vector<int>> mac(n);
    vector<vector<long long>> dur(n);
    for (int j = 0; j < n; j++) {
        scanf("%lld %d", &r[j], &o[j]);
        mac[j].resize(o[j]);
        dur[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            scanf("%d %lld", &mac[j][k], &dur[j][k]);
        }
    }

    vector<long long> resEnd(m, 0);       // last busy end per resource
    vector<long long> jobReady(n);        // earliest time next step may start
    vector<int> nxt(n, 0);                // next step index per telescope
    for (int j = 0; j < n; j++) jobReady[j] = r[j];

    vector<vector<long long>> st(n);
    for (int j = 0; j < n; j++) st[j].resize(o[j]);

    int total = 0;
    for (int j = 0; j < n; j++) total += o[j];

    for (int done = 0; done < total; done++) {
        int bj = -1;
        long long bestComp = LLONG_MAX, bestStart = 0;
        for (int j = 0; j < n; j++) {
            if (nxt[j] >= o[j]) continue;
            int k = nxt[j];
            long long start = max(jobReady[j], resEnd[mac[j][k]]);
            long long comp = start + dur[j][k];
            if (comp < bestComp) {
                bestComp = comp;
                bestStart = start;
                bj = j;
            }
        }
        int k = nxt[bj];
        st[bj][k] = bestStart;
        resEnd[mac[bj][k]] = bestStart + dur[bj][k];
        jobReady[bj] = bestStart + dur[bj][k];
        nxt[bj]++;
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", st[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
