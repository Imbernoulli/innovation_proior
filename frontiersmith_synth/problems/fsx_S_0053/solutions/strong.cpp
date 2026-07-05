// TIER: strong
// Multi-restart randomized flexible-job-shop scheduler.
// Each restart chooses a crew assignment (mix of fastest-crew and load-balanced /
// random eligible) and a dispatch policy (earliest-release+SPT, or randomized), then
// list-schedules with gap insertion. We keep the shortest makespan found. Restart 0
// reproduces the pure greedy config, so strong is never worse than greedy.
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<int> o;
vector<vector<vector<pair<int, int>>>> elig;

// Given an assignment (crew,dur per op) and a dispatch mode, build an active schedule.
long long buildSchedule(const vector<vector<int>>& asC,
                        const vector<vector<int>>& asD,
                        int mode, mt19937& rng,
                        vector<vector<long long>>& start) {
    vector<vector<pair<long long, long long>>> busy(m);
    vector<int> ptr(n, 0);
    vector<long long> lastEnd(n, 0);
    for (int j = 0; j < n; j++) start[j].assign(o[j], 0);
    int remaining = 0;
    for (int j = 0; j < n; j++) remaining += o[j];

    auto place = [&](int c, long long release, long long d) -> long long {
        auto& v = busy[c];
        sort(v.begin(), v.end());
        long long tpos = release;
        for (auto& iv : v) {
            if (tpos + d <= iv.first) break;
            if (iv.second > tpos) tpos = iv.second;
        }
        v.push_back({tpos, tpos + d});
        return tpos;
    };

    long long makespan = 0;
    while (remaining > 0) {
        // collect candidates (jobs with a ready next op)
        int bj = -1;
        long long bestKey1 = LLONG_MAX, bestKey2 = LLONG_MAX;
        vector<int> cands;
        for (int j = 0; j < n; j++) if (ptr[j] < o[j]) cands.push_back(j);
        if (mode == 2) {
            // random pick among candidates
            bj = cands[rng() % cands.size()];
        } else {
            for (int j : cands) {
                long long rel = lastEnd[j];
                long long dur = asD[j][ptr[j]];
                long long k1, k2;
                if (mode == 0) { k1 = rel; k2 = dur; }        // earliest release, SPT
                else           { k1 = dur; k2 = rel; }        // SPT, earliest release
                if (k1 < bestKey1 || (k1 == bestKey1 && k2 < bestKey2)) {
                    bestKey1 = k1; bestKey2 = k2; bj = j;
                }
            }
        }
        int k = ptr[bj];
        int c = asC[bj][k];
        long long d = asD[bj][k];
        long long s = place(c, lastEnd[bj], d);
        start[bj][k] = s;
        lastEnd[bj] = s + d;
        makespan = max(makespan, s + d);
        ptr[bj]++;
        remaining--;
    }
    return makespan;
}

int main() {
    scanf("%d %d", &n, &m);
    o.resize(n);
    elig.resize(n);
    for (int j = 0; j < n; j++) {
        scanf("%d", &o[j]);
        elig[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            int e;
            scanf("%d", &e);
            for (int idx = 0; idx < e; idx++) {
                int c, d;
                scanf("%d %d", &c, &d);
                elig[j][k].push_back({c, d});
            }
        }
    }

    mt19937 rng(987654321u);

    // fastest-crew assignment (greedy base)
    vector<vector<int>> fastC(n), fastD(n);
    for (int j = 0; j < n; j++) {
        fastC[j].resize(o[j]);
        fastD[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            int bc = elig[j][k][0].first, bd = elig[j][k][0].second;
            for (auto& pr : elig[j][k]) if (pr.second < bd) { bd = pr.second; bc = pr.first; }
            fastC[j][k] = bc; fastD[j][k] = bd;
        }
    }

    vector<vector<long long>> bestStart(n), bestC(n), tmp(n);
    for (int j = 0; j < n; j++) { bestC[j].assign(o[j], 0); bestStart[j].assign(o[j], 0); }
    long long best = LLONG_MAX;

    // scale restarts with instance size (kept well under the time limit)
    int totalOps = 0;
    for (int j = 0; j < n; j++) totalOps += o[j];
    int RESTARTS = max(60, 4000 / max(1, totalOps));

    for (int r = 0; r < RESTARTS; r++) {
        vector<vector<int>> asC(n), asD(n);
        int mode;
        if (r == 0) { mode = 0; asC = fastC; asD = fastD; }  // pure greedy
        else {
            mode = (int)(rng() % 3);
            // load-aware / randomized assignment
            vector<long long> load(m, 0);
            for (int j = 0; j < n; j++) {
                asC[j].resize(o[j]); asD[j].resize(o[j]);
                for (int k = 0; k < o[j]; k++) {
                    auto& cand = elig[j][k];
                    int pick;
                    int coin = (int)(rng() % 3);
                    if (coin == 0) {
                        // fastest
                        pick = 0;
                        for (int i = 1; i < (int)cand.size(); i++)
                            if (cand[i].second < cand[pick].second) pick = i;
                    } else if (coin == 1) {
                        // least loaded (break ties by duration)
                        pick = 0;
                        long long bestScore = load[cand[0].first] + cand[0].second;
                        for (int i = 1; i < (int)cand.size(); i++) {
                            long long sc = load[cand[i].first] + cand[i].second;
                            if (sc < bestScore) { bestScore = sc; pick = i; }
                        }
                    } else {
                        // random eligible
                        pick = (int)(rng() % cand.size());
                    }
                    asC[j][k] = cand[pick].first;
                    asD[j][k] = cand[pick].second;
                    load[cand[pick].first] += cand[pick].second;
                }
            }
        }
        long long mk = buildSchedule(asC, asD, mode, rng, tmp);
        if (mk < best) {
            best = mk;
            for (int j = 0; j < n; j++) {
                bestStart[j] = tmp[j];
                bestC[j] = vector<long long>(asC[j].begin(), asC[j].end());
            }
        }
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld %lld%c", bestC[j][k], bestStart[j][k], k + 1 == o[j] ? '\n' : ' ');
    return 0;
}
