#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int J = inf.readInt();
    int M = inf.readInt();

    // Read jobs.  For each global task record: eligible (asset->dur), job id.
    vector<vector<pair<int,int>>> elig;   // elig[i] = list of (asset,dur)
    vector<int> jobOfTask;
    vector<vector<int>> jobTasks(J);      // ordered global task indices per job

    long long B = 0;                      // baseline = sum of first-listed durations
    int total = 0;
    for (int j = 0; j < J; j++) {
        int k = inf.readInt();
        for (int t = 0; t < k; t++) {
            int e = inf.readInt();
            vector<pair<int,int>> opts;
            for (int r = 0; r < e; r++) {
                int a = inf.readInt();
                int d = inf.readInt();
                opts.push_back({a, d});
            }
            B += opts[0].second;          // first-listed duration
            int idx = (int)elig.size();
            elig.push_back(opts);
            jobOfTask.push_back(j);
            jobTasks[j].push_back(idx);
            total++;
        }
    }

    // Read participant output: for each task in input order, (asset, start).
    vector<long long> start(total), dur(total);
    vector<int> asset(total);
    for (int i = 0; i < total; i++) {
        int a = ouf.readInt(1, M, "asset");
        long long s = ouf.readLong(0LL, 2000000000LL, "start");
        // asset must be eligible for this task; find its duration.
        long long d = -1;
        for (auto& pr : elig[i]) {
            if (pr.first == a) { d = pr.second; break; }
        }
        if (d < 0)
            quitf(_wa, "task %d assigned to asset %d which is not eligible", i + 1, a);
        asset[i] = a;
        start[i] = s;
        dur[i] = d;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d tasks", total);

    // Feasibility: precedence within each job.
    for (int j = 0; j < J; j++) {
        for (int p = 1; p < (int)jobTasks[j].size(); p++) {
            int prev = jobTasks[j][p - 1];
            int cur  = jobTasks[j][p];
            if (start[cur] < start[prev] + dur[prev])
                quitf(_wa, "precedence violated in job %d: task starts at %lld before predecessor finishes at %lld",
                      j + 1, start[cur], start[prev] + dur[prev]);
        }
    }

    // Feasibility: no overlap per asset.
    vector<vector<pair<long long,long long>>> byAsset(M + 1); // (start,end)
    for (int i = 0; i < total; i++)
        byAsset[asset[i]].push_back({start[i], start[i] + dur[i]});
    for (int a = 1; a <= M; a++) {
        auto& v = byAsset[a];
        sort(v.begin(), v.end());
        for (int q = 1; q < (int)v.size(); q++) {
            if (v[q].first < v[q - 1].second)
                quitf(_wa, "asset %d has overlapping tasks: [%lld,%lld) and [%lld,%lld)",
                      a, v[q - 1].first, v[q - 1].second, v[q].first, v[q].second);
        }
    }

    // Objective: makespan.
    long long F = 0;
    for (int i = 0; i < total; i++) F = max(F, start[i] + dur[i]);

    if (B <= 0) B = 1; // safety; B is always positive for valid inputs
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
