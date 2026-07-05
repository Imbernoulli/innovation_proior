// TIER: strong
// Multi-order constructive search.  We simulate the same "grab p_j cheapest free steps in
// window, commit if power cost < penalty" packing under SEVERAL task orderings:
//   (0) penalty v_j descending          -- same criterion as the greedy baseline
//   (1) value-density v_j/p_j descending -- spend scarce solar steps on efficient tasks
//   (2) required steps p_j ascending     -- clear many small backlogs before big ones
//   (3) profit-density (v_j - lower-bound power)/p_j descending
// Each ordering yields a full schedule; we keep the one with the lowest total cost F.  Because
// ordering (0) reproduces the greedy schedule, the result is never worse than greedy, and it is
// strictly better whenever a different ordering packs the solar steps more cheaply.  A final
// fill pass tries to clear any still-profitable dropped task in the leftover free steps.
#include <bits/stdc++.h>
using namespace std;

int T, N;
vector<int> A; vector<long long> CS, CD;
vector<int> R, D, P; vector<long long> V;

static inline long long stepCost(int t){ return A[t] ? CS[t] : CD[t]; }

// Simulate a packing for a given task order.  Returns F; fills job/mode.
long long simulate(const vector<int>& ord, vector<int>& job, vector<int>& mode) {
    fill(job.begin(), job.end(), 0);
    fill(mode.begin(), mode.end(), 0);
    vector<char> occ(T + 1, 0);
    long long power = 0;
    long long cleared = 0;                 // sum of v of cleared tasks
    vector<pair<long long,int>> cand;
    for (int j : ord) {
        cand.clear();
        for (int t = R[j]; t <= D[j]; t++)
            if (!occ[t]) cand.push_back({stepCost(t), t});
        if ((int)cand.size() < P[j]) continue;
        nth_element(cand.begin(), cand.begin() + P[j], cand.end());
        long long cost = 0;
        for (int k = 0; k < P[j]; k++) cost += cand[k].first;
        if (cost >= V[j]) continue;
        for (int k = 0; k < P[j]; k++) {
            int t = cand[k].second;
            occ[t] = 1; job[t] = j; mode[t] = A[t] ? 0 : 1;
        }
        power += cost; cleared += V[j];
    }
    // final fill pass: attempt any still-profitable dropped task (a second cheap sweep)
    for (int j = 1; j <= N; j++) {
        // is task j cleared?  count its committed steps
        // (cheap check: recompute occupancy count in window is expensive; instead track via a set)
        // We approximate by re-scanning the window once.
        int have = 0;
        for (int t = R[j]; t <= D[j]; t++) if (job[t] == j) have++;
        if (have >= P[j]) continue;
        // if partially assigned steps exist for j they are already committed to j; free steps:
        cand.clear();
        for (int t = R[j]; t <= D[j]; t++)
            if (job[t] == 0) cand.push_back({stepCost(t), t});
        int need = P[j] - have;
        if ((int)cand.size() < need) continue;
        nth_element(cand.begin(), cand.begin() + need, cand.end());
        long long cost = 0;
        for (int k = 0; k < need; k++) cost += cand[k].first;
        if (cost >= V[j]) continue;
        for (int k = 0; k < need; k++) {
            int t = cand[k].second;
            job[t] = j; mode[t] = A[t] ? 0 : 1;
        }
        power += cost; cleared += V[j];
    }
    long long totalV = 0; for (int j = 1; j <= N; j++) totalV += V[j];
    long long penalty = totalV - cleared;
    return power + penalty;
}

int main() {
    if (scanf("%d %d", &T, &N) != 2) return 0;
    A.assign(T + 1, 0); CS.assign(T + 1, 0); CD.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) scanf("%d %lld %lld", &A[t], &CS[t], &CD[t]);
    R.assign(N + 1, 0); D.assign(N + 1, 0); P.assign(N + 1, 0); V.assign(N + 1, 0);
    for (int j = 1; j <= N; j++) scanf("%d %d %d %lld", &R[j], &D[j], &P[j], &V[j]);

    vector<int> base(N);
    for (int j = 0; j < N; j++) base[j] = j + 1;

    // lower-bound power for profit-density: p_j cheapest solar/diesel-independent estimate = p_j * min cost seen
    // Use per-task minimum achievable step cost in window for a rough profit estimate.
    vector<long long> lb(N + 1, 0);
    for (int j = 1; j <= N; j++) {
        long long mn = LLONG_MAX;
        for (int t = R[j]; t <= D[j]; t++) mn = min(mn, stepCost(t));
        if (mn == LLONG_MAX) mn = 0;
        lb[j] = mn * P[j];
    }

    vector<vector<int>> orders;
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){ return V[x] > V[y]; }); orders.push_back(o); }
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){
        return (double)V[x]/P[x] > (double)V[y]/P[y]; }); orders.push_back(o); }
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){
        if (P[x] != P[y]) return P[x] < P[y]; return V[x] > V[y]; }); orders.push_back(o); }
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){
        return (double)(V[x]-lb[x])/P[x] > (double)(V[y]-lb[y])/P[y]; }); orders.push_back(o); }

    vector<int> job(T + 1), mode(T + 1), bestJob(T + 1, 0), bestMode(T + 1, 0);
    long long bestF = LLONG_MAX;
    for (auto& o : orders) {
        long long f = simulate(o, job, mode);
        if (f < bestF) { bestF = f; bestJob = job; bestMode = mode; }
    }

    string out; out.reserve((size_t)T * 4);
    char buf[32];
    for (int t = 1; t <= T; t++) {
        int len = sprintf(buf, "%d %d\n", bestJob[t], bestMode[t]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
