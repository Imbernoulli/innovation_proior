// TIER: greedy
// One-pass ECT dispatch: repeatedly take the frontier task (next unscheduled task of
// some ride) that can *finish* earliest and place it at its earliest feasible slot;
// break ties toward heavier (more popular) rides so they open sooner.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> w(n);
    vector<vector<int>> mach(n), dur(n);
    int totalOps = 0;
    for (int j = 0; j < n; j++) {
        int o;
        scanf("%d %d", &w[j], &o);
        mach[j].resize(o); dur[j].resize(o);
        for (int k = 0; k < o; k++) { scanf("%d %d", &mach[j][k], &dur[j][k]); }
        totalOps += o;
    }

    vector<int> nextOp(n, 0);
    vector<long long> jobReady(n, 0), machFree(m, 0);
    vector<vector<long long>> start(n);
    for (int j = 0; j < n; j++) start[j].assign(mach[j].size(), 0);

    for (int done = 0; done < totalOps; done++) {
        int bj = -1;
        long long bestFinish = LLONG_MAX;
        int bestW = -1;
        for (int j = 0; j < n; j++) {
            int k = nextOp[j];
            if (k >= (int)mach[j].size()) continue;
            long long st = max(jobReady[j], machFree[mach[j][k]]);
            long long fin = st + dur[j][k];
            if (fin < bestFinish || (fin == bestFinish && w[j] > bestW)) {
                bestFinish = fin; bestW = w[j]; bj = j;
            }
        }
        int k = nextOp[bj];
        long long st = max(jobReady[bj], machFree[mach[bj][k]]);
        start[bj][k] = st;
        jobReady[bj] = st + dur[bj][k];
        machFree[mach[bj][k]] = st + dur[bj][k];
        nextOp[bj]++;
    }

    for (int j = 0; j < n; j++) {
        string line;
        for (size_t k = 0; k < start[j].size(); k++) {
            if (k) line += ' ';
            line += to_string(start[j][k]);
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
