// TIER: strong
// Active-schedule generation (Giffler-Thompson): at each step find the ready stop with the
// minimum completion time, take its machine, and resolve the machine conflict set by a
// priority rule. Runs several deterministic rules (SPT / LPT / min-completion) plus many
// seeded randomised conflict resolutions, and keeps the lowest-makespan timetable found.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int M, J;
vector<int> L;
vector<vector<int>> mach, p;
ll total;

// mode: 0 SPT, 1 LPT, 2 min-completion, 3 random (rng used only for mode 3)
ll build(int mode, mt19937_64& rng, vector<vector<ll>>& st) {
    vector<ll> machFree(M + 1, 0), jobReady(J, 0);
    vector<int> nxt(J, 0);
    for (int j = 0; j < J; j++) st[j].assign(L[j], 0);
    ll done = 0, makespan = 0;
    while (done < total) {
        // find ready op with minimum completion time -> its machine is the "critical" machine
        ll minComp = LLONG_MAX; int cm = -1;
        for (int j = 0; j < J; j++) {
            if (nxt[j] >= L[j]) continue;
            int i = nxt[j], m = mach[j][i];
            ll est = max(jobReady[j], machFree[m]);
            ll comp = est + p[j][i];
            if (comp < minComp) { minComp = comp; cm = m; }
        }
        // conflict set: ready ops on machine cm that could start before minComp
        vector<int> cand;
        for (int j = 0; j < J; j++) {
            if (nxt[j] >= L[j]) continue;
            int i = nxt[j];
            if (mach[j][i] != cm) continue;
            ll est = max(jobReady[j], machFree[cm]);
            if (est < minComp) cand.push_back(j);
        }
        // choose one convoy from the conflict set by the active rule
        int pick = cand[0];
        if (mode == 3) {
            pick = cand[(size_t)(rng() % cand.size())];
        } else {
            ll bestKey = 0; int bj = -1;
            for (int j : cand) {
                int i = nxt[j];
                ll est = max(jobReady[j], machFree[cm]);
                ll key;
                if (mode == 0) key = p[j][i];                 // SPT
                else if (mode == 1) key = -(ll)p[j][i];       // LPT
                else key = est + p[j][i];                     // min completion
                if (bj == -1 || key < bestKey || (key == bestKey && j < bj)) {
                    bestKey = key; bj = j;
                }
            }
            pick = bj;
        }
        int i = nxt[pick], m = mach[pick][i];
        ll est = max(jobReady[pick], machFree[m]);
        st[pick][i] = est;
        machFree[m] = est + p[pick][i];
        jobReady[pick] = est + p[pick][i];
        nxt[pick]++;
        done++;
        makespan = max(makespan, machFree[m]);
    }
    return makespan;
}

int main() {
    if (scanf("%d %d", &M, &J) != 2) return 0;
    L.resize(J); mach.resize(J); p.resize(J);
    total = 0;
    for (int j = 0; j < J; j++) {
        scanf("%d", &L[j]);
        mach[j].resize(L[j]); p[j].resize(L[j]);
        for (int i = 0; i < L[j]; i++) { scanf("%d %d", &mach[j][i], &p[j][i]); total++; }
    }
    mt19937_64 rng(0x9E3779B97F4A7C15ULL);   // fixed seed -> deterministic
    vector<vector<ll>> best, cur(J);
    ll bestMk = LLONG_MAX;

    for (int mode = 0; mode < 3; mode++) {
        ll mk = build(mode, rng, cur);
        if (mk < bestMk) { bestMk = mk; best = cur; }
    }
    int restarts = 600;
    for (int r = 0; r < restarts; r++) {
        ll mk = build(3, rng, cur);
        if (mk < bestMk) { bestMk = mk; best = cur; }
    }
    for (int j = 0; j < J; j++) {
        for (int i = 0; i < L[j]; i++) printf("%lld ", best[j][i]);
        printf("\n");
    }
    return 0;
}
