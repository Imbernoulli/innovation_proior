// TIER: greedy
// One-pass, job-by-job list scheduling.  Process jobs in input order; for each
// task pick the eligible asset giving the earliest completion (append at end of
// that asset's timeline, respecting the job's ready time).  No gap insertion.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;

    vector<vector<vector<pair<int,int>>>> jobs(J);
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        jobs[j].resize(k);
        for (int t = 0; t < k; t++) {
            int e; scanf("%d", &e);
            for (int r = 0; r < e; r++) {
                int a, d; scanf("%d %d", &a, &d);
                jobs[j][t].push_back({a, d});
            }
        }
    }

    vector<long long> machineFree(M + 1, 0);
    // Output must be in input order; we go job by job, task by task.
    for (int j = 0; j < J; j++) {
        long long jobReady = 0;
        for (auto& op : jobs[j]) {
            long long bestFinish = LLONG_MAX, bestStart = 0;
            int bestA = op[0].first;
            for (auto& pr : op) {
                int a = pr.first; long long d = pr.second;
                long long s = max(machineFree[a], jobReady);
                long long f = s + d;
                if (f < bestFinish) { bestFinish = f; bestStart = s; bestA = a; }
            }
            // apply
            long long d = 0;
            for (auto& pr : op) if (pr.first == bestA) d = pr.second;
            machineFree[bestA] = bestStart + d;
            jobReady = bestStart + d;
            printf("%d %lld\n", bestA, bestStart);
        }
    }
    return 0;
}
