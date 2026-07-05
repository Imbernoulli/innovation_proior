// TIER: greedy
// Earliest-completion-time list scheduling, append-only per dam. Process orders in
// input order; for each pass choose the eligible dam giving the earliest completion,
// placing it at the end of that dam's current timeline.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> os(n);
    vector<vector<vector<pair<int,int>>>> opt(n);
    for (int j = 0; j < n; j++) {
        int o; scanf("%d", &o); os[j] = o; opt[j].resize(o);
        for (int k = 0; k < o; k++) {
            int c; scanf("%d", &c);
            for (int e = 0; e < c; e++) {
                int dam, dur; scanf("%d %d", &dam, &dur);
                opt[j][k].push_back({dam, dur});
            }
        }
    }

    vector<long long> machFree(m, 0);
    for (int j = 0; j < n; j++) {
        long long jobReady = 0;
        string line;
        for (int k = 0; k < os[j]; k++) {
            int bestDam = -1; long long bestStart = 0, bestComp = LLONG_MAX;
            for (auto& pr : opt[j][k]) {
                int dam = pr.first, dur = pr.second;
                long long st = max(jobReady, machFree[dam]);
                long long comp = st + dur;
                if (comp < bestComp) { bestComp = comp; bestStart = st; bestDam = dam; }
            }
            machFree[bestDam] = bestComp;
            jobReady = bestComp;
            line += to_string(bestDam) + " " + to_string(bestStart);
            if (k + 1 < os[j]) line += " ";
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
