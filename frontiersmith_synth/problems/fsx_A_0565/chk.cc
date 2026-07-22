#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Quay Cranes on One Rail (monotone matching + staggering).
// Minimization.  Reads K,M,S,alpha,gamma; standby powers p[]; jobs (pos,work).
// Participant output: M crane indices a_j (input order) then K start offsets s_c.
// Validates: a_j in [1,K]; monotone (a non-decreasing when jobs sorted by position);
// inside each congested stretch (consecutive non-empty zones with bay gap < S) the busy
// intervals [s_c, s_c+D_c) are pairwise disjoint.
// Objective F = T + gamma * sum_c p_c*(T - D_c),  T = max non-empty (s_c + D_c),
//   D_c = work_zone + alpha*span_zone.
// Baseline B = F of the even-count partition (K equal-count contiguous zones, minimal
//   staggered offsets).  ratio = min(1, (B/max(1,F))/10).

struct Job { long long pos; long long work; };

static int K, M;
static long long S, ALPHA, GAMMA;
static vector<long long> P;           // standby powers, size K
static vector<Job> jobs;              // sorted by position

// Given a zone assignment zoneOf[i] in [0,K) for each sorted job (non-decreasing) with the
// listed crane indices, compute D per crane and the MINIMAL staggered offsets, returning F.
// Empty cranes have D=0.  Used both for the baseline and (indirectly) nowhere else.
static long long evalMinimalF(const vector<int>& zoneOf) {
    vector<long long> D(K, 0), W(K, 0), mn(K, LLONG_MAX), mx(K, LLONG_MIN);
    vector<char> nonempty(K, 0);
    for (int i = 0; i < M; i++) {
        int c = zoneOf[i];
        nonempty[c] = 1;
        W[c] += jobs[i].work;
        mn[c] = min(mn[c], jobs[i].pos);
        mx[c] = max(mx[c], jobs[i].pos);
    }
    for (int c = 0; c < K; c++)
        if (nonempty[c]) D[c] = W[c] + ALPHA * (mx[c] - mn[c]);
    // order non-empty cranes left to right (index order == position order under monotone)
    vector<int> order;
    for (int c = 0; c < K; c++) if (nonempty[c]) order.push_back(c);
    vector<long long> s(K, 0);
    long long runClock = 0, prevMax = LLONG_MIN;
    bool first = true;
    for (int oi = 0; oi < (int)order.size(); oi++) {
        int c = order[oi];
        long long gap = first ? (S + 1) : (mn[c] - prevMax);
        if (first || gap >= S) { s[c] = 0; runClock = D[c]; }
        else { s[c] = runClock; runClock += D[c]; }
        prevMax = mx[c];
        first = false;
    }
    long long T = 0;
    for (int c = 0; c < K; c++) if (nonempty[c]) T = max(T, s[c] + D[c]);
    long long idle = 0;
    for (int c = 0; c < K; c++) idle += P[c] * (T - D[c]);
    long long F = T + GAMMA * idle;
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    K = inf.readInt();
    M = inf.readInt();
    S = inf.readInt();
    ALPHA = inf.readInt();
    GAMMA = inf.readInt();
    P.resize(K);
    for (int c = 0; c < K; c++) P[c] = inf.readInt();
    vector<Job> raw(M);
    for (int i = 0; i < M; i++) { raw[i].pos = inf.readInt(); raw[i].work = inf.readInt(); }

    // sorted order by position
    vector<int> ordIdx(M);
    for (int i = 0; i < M; i++) ordIdx[i] = i;
    sort(ordIdx.begin(), ordIdx.end(), [&](int a, int b){ return raw[a].pos < raw[b].pos; });
    jobs.resize(M);
    for (int r = 0; r < M; r++) jobs[r] = raw[ordIdx[r]];

    // ---- read participant assignment (input order) ----
    vector<int> aIn(M);
    for (int i = 0; i < M; i++) aIn[i] = ouf.readInt(1, K, "a") - 1;   // 0-based crane
    // start offsets
    vector<long long> s(K);
    for (int c = 0; c < K; c++) s[c] = ouf.readLong(0LL, 4000000000LL, "s");
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the offsets");

    // map assignment to sorted order and check monotonicity
    vector<int> zoneOf(M);
    for (int r = 0; r < M; r++) zoneOf[r] = aIn[ordIdx[r]];
    for (int r = 1; r < M; r++)
        if (zoneOf[r] < zoneOf[r - 1])
            quitf(_wa, "non-monotone assignment: a crossing occurs at position rank %d", r);

    // ---- per-crane duration ----
    vector<long long> D(K, 0), W(K, 0), mn(K, LLONG_MAX), mx(K, LLONG_MIN);
    vector<char> nonempty(K, 0);
    for (int r = 0; r < M; r++) {
        int c = zoneOf[r];
        nonempty[c] = 1;
        W[c] += jobs[r].work;
        mn[c] = min(mn[c], jobs[r].pos);
        mx[c] = max(mx[c], jobs[r].pos);
    }
    for (int c = 0; c < K; c++)
        if (nonempty[c]) D[c] = W[c] + ALPHA * (mx[c] - mn[c]);

    // ---- congested-stretch feasibility on participant offsets ----
    vector<int> order;
    for (int c = 0; c < K; c++) if (nonempty[c]) order.push_back(c);
    // build congested stretches (consecutive non-empty zones with gap < S)
    int nz = (int)order.size();
    int i0 = 0;
    while (i0 < nz) {
        int j0 = i0;
        while (j0 + 1 < nz) {
            int cL = order[j0], cR = order[j0 + 1];
            long long gap = mn[cR] - mx[cL];
            if (gap < S) j0++;
            else break;
        }
        // stretch order[i0..j0] must have pairwise-disjoint busy intervals
        for (int u = i0; u <= j0; u++) {
            for (int v = u + 1; v <= j0; v++) {
                int cu = order[u], cv = order[v];
                long long au = s[cu], bu = s[cu] + D[cu];
                long long av = s[cv], bv = s[cv] + D[cv];
                bool disjoint = (bu <= av) || (bv <= au);
                if (!disjoint)
                    quitf(_wa, "cranes %d and %d overlap in a congested stretch (must stagger)",
                          cu + 1, cv + 1);
            }
        }
        i0 = j0 + 1;
    }

    // ---- makespan + objective ----
    long long T = 0;
    for (int c = 0; c < K; c++) if (nonempty[c]) T = max(T, s[c] + D[c]);
    long long idle = 0;
    for (int c = 0; c < K; c++) idle += P[c] * (T - D[c]);
    long long F = T + GAMMA * idle;
    if (F < 1) F = 1;

    // ---- baseline B: even-count partition over all K cranes ----
    vector<int> baseZone(M);
    for (int r = 0; r < M; r++) {
        // contiguous equal-count buckets
        int c = (int)((long long)r * K / M);
        if (c >= K) c = K - 1;
        baseZone[r] = c;
    }
    long long B = evalMinimalF(baseZone);
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
