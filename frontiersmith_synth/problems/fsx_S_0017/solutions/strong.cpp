// TIER: strong
// Active-schedule generation with gap insertion. At each step the next ready
// operation of some job is chosen by a priority rule, and placed into the
// EARLIEST fitting gap of its workstation timeline (respecting job precedence).
// Several dispatch rules (ECT / SPT / LPT / MWKR / LWKR) plus seeded random
// restarts are tried; the schedule with the smallest makespan is emitted.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int J, M;
vector<vector<pair<int,int>>> jobs;   // (machine, dur)
int total;

// earliest feasible start >= ready for a duration `dur` on a machine whose busy
// intervals `iv` are sorted by start and mutually disjoint.
ll earliest(const vector<pair<ll,ll>>& iv, ll ready, int dur) {
    ll cur = ready;
    for (const auto& p : iv) {
        if (p.second <= cur) continue;          // interval entirely before cur
        if (p.first >= cur + dur) break;        // gap [cur, cur+dur) fits
        cur = max(cur, p.second);               // overlap -> jump past it
    }
    return cur;
}

void insertIv(vector<pair<ll,ll>>& iv, ll s, ll e) {
    auto it = lower_bound(iv.begin(), iv.end(), make_pair(s, e));
    iv.insert(it, {s, e});
}

// Run one schedule with a given rule; fill `outStart` (global op order) and
// return the makespan. rule: 0=ECT,1=SPT,2=LPT,3=MWKR,4=LWKR,5=random.
ll runRule(int rule, uint64_t seed, vector<ll>& outStart) {
    mt19937_64 rng(seed);
    vector<vector<pair<ll,ll>>> byM(M + 1);
    vector<int> ptr(J, 0);
    vector<ll> jobReady(J, 0);
    // remaining work per job (sum of remaining durations)
    vector<ll> remWork(J, 0);
    for (int j = 0; j < J; j++)
        for (auto& op : jobs[j]) remWork[j] += op.second;

    // base index of each job's first op in global order
    vector<int> baseIdx(J, 0);
    { int c = 0; for (int j = 0; j < J; j++) { baseIdx[j] = c; c += (int)jobs[j].size(); } }

    outStart.assign(total, 0);
    ll makespan = 0;
    int done = 0;
    while (done < total) {
        int best = -1;
        ll bestKey = 0, bestTie = 0;
        ll bestEst = 0;
        for (int j = 0; j < J; j++) {
            if (ptr[j] >= (int)jobs[j].size()) continue;
            int m = jobs[j][ptr[j]].first;
            int d = jobs[j][ptr[j]].second;
            ll est = earliest(byM[m], jobReady[j], d);
            ll key;
            switch (rule) {
                case 0: key = est + d;                 break; // earliest completion
                case 1: key = d;                       break; // shortest processing
                case 2: key = -(ll)d;                  break; // longest processing
                case 3: key = -remWork[j];             break; // most work remaining
                case 4: key = remWork[j];              break; // least work remaining
                default: key = (ll)(rng() & 0xffffff); break; // random
            }
            ll tie = (ll)(rng() & 0xffff);
            if (best == -1 || key < bestKey || (key == bestKey && tie < bestTie)) {
                best = j; bestKey = key; bestTie = tie; bestEst = est;
            }
        }
        int j = best;
        int m = jobs[j][ptr[j]].first;
        int d = jobs[j][ptr[j]].second;
        ll s = bestEst;
        ll e = s + d;
        insertIv(byM[m], s, e);
        outStart[baseIdx[j] + ptr[j]] = s;
        makespan = max(makespan, e);
        jobReady[j] = e;
        remWork[j] -= d;
        ptr[j]++;
        done++;
    }
    return makespan;
}

int main() {
    if (scanf("%d %d", &J, &M) != 2) return 0;
    jobs.assign(J, {});
    total = 0;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int o = 0; o < k; o++) {
            int m, d; scanf("%d %d", &m, &d);
            jobs[j].push_back({m, d});
            total++;
        }
    }

    vector<ll> bestSched;
    ll bestMk = LLONG_MAX;
    // deterministic set of runs: the 5 fixed rules + several seeded random runs
    for (int rule = 0; rule <= 4; rule++) {
        vector<ll> sched;
        ll mk = runRule(rule, 1234567u + rule, sched);
        if (mk < bestMk) { bestMk = mk; bestSched = sched; }
    }
    for (int r = 0; r < 24; r++) {
        vector<ll> sched;
        ll mk = runRule(5, 0x9e3779b97f4a7c15ULL * (r + 1), sched);
        if (mk < bestMk) { bestMk = mk; bestSched = sched; }
    }

    for (size_t i = 0; i < bestSched.size(); i++)
        printf("%lld\n", bestSched[i]);
    return 0;
}
