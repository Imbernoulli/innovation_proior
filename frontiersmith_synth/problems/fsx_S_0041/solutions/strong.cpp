// TIER: strong
// Active-schedule dispatch with GAP INSERTION plus machine selection, run under
// several priority rules and many seeded random restarts; keep the best makespan.
#include <bits/stdc++.h>
using namespace std;

int J, M, total;
vector<vector<pair<int,int>>> elig;   // elig[gid] = (asset,dur)
vector<vector<int>> jobTasks;         // ordered global ids per job
vector<int> minDur;                   // cheapest eligible duration per task
vector<long long> workRemainFromMin;  // sum of minDur for this and later tasks in job

// earliest feasible start >= ready on machine a given occupied intervals (sorted).
long long earliestGap(const vector<pair<long long,long long>>& iv, long long ready, long long d) {
    long long t = ready;
    for (auto& e : iv) {
        if (e.second <= t) continue;          // interval already before t
        if (t + d <= e.first) return t;       // fits before this interval
        t = max(t, e.second);
    }
    return t;
}

int main() {
    if (scanf("%d %d", &J, &M) != 2) return 0;
    jobTasks.assign(J, {});
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int t = 0; t < k; t++) {
            int e; scanf("%d", &e);
            vector<pair<int,int>> opts;
            for (int r = 0; r < e; r++) { int a, d; scanf("%d %d", &a, &d); opts.push_back({a,d}); }
            int gid = (int)elig.size();
            elig.push_back(opts);
            jobTasks[j].push_back(gid);
        }
    }
    total = (int)elig.size();
    minDur.assign(total, 0);
    for (int i = 0; i < total; i++) {
        int md = INT_MAX;
        for (auto& pr : elig[i]) md = min(md, pr.second);
        minDur[i] = md;
    }
    workRemainFromMin.assign(total, 0);
    for (int j = 0; j < J; j++) {
        long long acc = 0;
        for (int p = (int)jobTasks[j].size() - 1; p >= 0; p--) {
            int gid = jobTasks[j][p];
            acc += minDur[gid];
            workRemainFromMin[gid] = acc;
        }
    }

    vector<int> bestAsset(total, -1);
    vector<long long> bestStart(total, 0);
    long long bestMakespan = LLONG_MAX;

    auto buildSchedule = [&](int rule, unsigned seed) {
        mt19937 rng(seed);
        vector<vector<pair<long long,long long>>> occ(M + 1);
        vector<int> nextIdx(J, 0);
        vector<long long> jobReady(J, 0);
        vector<int> outA(total, -1);
        vector<long long> outS(total, 0);
        int remaining = total;
        long long makespan = 0;

        while (remaining > 0) {
            // candidate jobs with a pending task
            int chosenJob = -1;
            double bestKey = 0; bool has = false;
            for (int j = 0; j < J; j++) {
                if (nextIdx[j] >= (int)jobTasks[j].size()) continue;
                int gid = jobTasks[j][nextIdx[j]];
                double key;
                switch (rule) {
                    case 0: key = -minDur[gid]; break;                 // SPT
                    case 1: key = minDur[gid]; break;                  // LPT
                    case 2: key = workRemainFromMin[gid]; break;       // MWKR
                    default: key = (double)(rng() % 100000); break;    // random
                }
                // random jitter to break ties / diversify restarts
                key += (double)(rng() % 1000) / 1e6;
                if (!has || key > bestKey) { bestKey = key; chosenJob = j; has = true; }
            }
            int j = chosenJob;
            int gid = jobTasks[j][nextIdx[j]];
            long long ready = jobReady[j];

            // choose machine minimizing completion (random tie-break)
            long long bestF = LLONG_MAX, bestS = 0; int selA = elig[gid][0].first;
            // random offset so ties resolve differently across restarts
            int base = rng() % (int)elig[gid].size();
            for (int r = 0; r < (int)elig[gid].size(); r++) {
                auto& pr = elig[gid][(base + r) % elig[gid].size()];
                int a = pr.first; long long d = pr.second;
                long long s = earliestGap(occ[a], ready, d);
                long long f = s + d;
                if (f < bestF) { bestF = f; bestS = s; selA = a; }
            }
            long long selD = 0;
            for (auto& pr : elig[gid]) if (pr.first == selA) selD = pr.second;

            occ[selA].push_back({bestS, bestS + selD});
            sort(occ[selA].begin(), occ[selA].end());
            outA[gid] = selA; outS[gid] = bestS;
            jobReady[j] = bestS + selD;
            makespan = max(makespan, bestS + selD);
            nextIdx[j]++;
            remaining--;
        }

        if (makespan < bestMakespan) {
            bestMakespan = makespan;
            bestAsset = outA;
            bestStart = outS;
        }
    };

    int restarts = 40;
    for (int rule = 0; rule < 4; rule++)
        for (int r = 0; r < restarts; r++)
            buildSchedule(rule, (unsigned)(rule * 100003 + r * 7919 + 1));

    for (int i = 0; i < total; i++)
        printf("%d %lld\n", bestAsset[i], bestStart[i]);
    return 0;
}
