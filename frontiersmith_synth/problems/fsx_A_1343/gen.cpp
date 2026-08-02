#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Generator for "Certifying the Losing Positions" (family: nim-variant-invariant).
// Emits: N, then per pile a move-menu S_i, then NQ query positions.
// Uses the SAME ground-truth DP+period machinery as chk.cc/strong.cpp so it can
// (a) balance the P/N label mix of queries (raw random draws skew ~75-80% N-position,
//     which would let a lazy "always guess N" strategy look artificially competent),
// and (b) plant TRAP queries: a pile size that is an exact multiple of (max(S_i)+1),
//     the period the classic "remove 1..C" formula would (wrongly) predict for a
//     general/sparse move menu -- this is exactly where a solver who misapplies that
//     textbook formula instead of discovering the pile's real period goes wrong.

static const int A_PROBE = 8000;
static const int TAIL = 4000;
static const int PMAX_SEARCH = 1500;
static const int R_START = A_PROBE - TAIL;

vector<int> grundyDP(const vector<int>& S) {
    vector<int> g(A_PROBE, 0);
    for (int a = 1; a < A_PROBE; a++) {
        bool seenArr[16] = {false};
        for (int s : S) {
            if (s <= a) {
                int v = g[a - s];
                if (v >= 0 && v < 16) seenArr[v] = true;
            }
        }
        int m = 0;
        while (m < 16 && seenArr[m]) m++;
        g[a] = m;
    }
    return g;
}

int findPeriod(const vector<int>& g) {
    for (int p = 1; p <= PMAX_SEARCH; p++) {
        bool ok = true;
        for (int a = R_START + p; a < A_PROBE; a++) {
            if (g[a] != g[a - p]) { ok = false; break; }
        }
        if (ok) return p;
    }
    return -1;
}

inline int trueGrundy(ll a, const vector<int>& g, int p) {
    if (a < A_PROBE) return g[(int)a];
    ll r = R_START + (a - R_START) % p;
    return g[(int)r];
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const int    Nc[11]        = {0, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10};
    static const int    Kmaxc[11]     = {0, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6};
    static const int    maxSc[11]     = {0, 6, 8, 10, 12, 14, 16, 18, 20, 20, 20};
    static const int    NQc[11]       = {0, 60, 120, 220, 350, 500, 700, 1000, 1500, 2200, 3000};
    static const ll     valHi[11]     = {0, 80, 2000, (ll)1e6, (ll)1e8, (ll)1e9, (ll)1e10,
                                          (ll)1e11, (ll)1e12, (ll)1e13, (ll)4e15};
    static const double trapFrac[11]  = {0, 0.0, 0.0, 0.05, 0.15, 0.25, 0.35, 0.4, 0.45, 0.5, 0.5};
    static const double forcedPFrac[11] = {0, 0.35, 0.35, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4};

    int N = Nc[testId];
    int Kmax = Kmaxc[testId];
    int maxS = maxSc[testId];
    int NQ = NQc[testId];
    ll VH = valHi[testId];
    double trapF = trapFrac[testId];
    double forcedF = forcedPFrac[testId];

    vector<vector<int>> S(N);
    vector<vector<int>> garr(N);
    vector<int> per(N);
    vector<int> maxSi(N);
    for (int i = 0; i < N; i++) {
        int K = rnd.next(3, Kmax);
        set<int> chosen;
        while ((int)chosen.size() < K) chosen.insert(rnd.next(1, maxS));
        S[i].assign(chosen.begin(), chosen.end());
        maxSi[i] = S[i].back();
        garr[i] = grundyDP(S[i]);
        per[i] = findPeriod(garr[i]);
        if (per[i] == -1) per[i] = 1;
    }

    printf("%d\n", N);
    for (int i = 0; i < N; i++) {
        printf("%d", (int)S[i].size());
        for (int v : S[i]) printf(" %d", v);
        printf("\n");
    }
    printf("%d\n", NQ);

    // small-value ceiling used for the "organic small" flavor of query within a test
    ll smallHi = min(VH, (ll)2000);

    for (int q = 0; q < NQ; q++) {
        vector<ll> a(N);
        double roll = rnd.next(0.0, 1.0);
        if (roll < forcedF) {
            // forced-P: pick piles 0..N-2 (mix small/large), then solve pile N-1 exactly.
            int need = -1;
            int target = 0;
            for (int i = 0; i + 1 < N; i++) {
                ll v = (rnd.next(0, 1) == 0) ? rnd.next(0LL, smallHi) : rnd.next(0LL, VH);
                a[i] = v;
                target ^= trueGrundy(v, garr[i], per[i]);
            }
            int last = N - 1;
            int p = per[last];
            // search the canonical periodic block for a residue hitting `target`
            int foundOff = -1;
            for (int off = 0; off < p; off++) {
                if (garr[last][R_START + off] == target) { foundOff = off; break; }
            }
            if (foundOff == -1) {
                a[last] = (rnd.next(0, 1) == 0) ? rnd.next(0LL, smallHi) : rnd.next(0LL, VH);
            } else {
                ll M = rnd.next(0LL, VH);
                ll base = R_START + foundOff;
                if (M <= base) a[last] = base;
                else {
                    ll kk = (M - base) / p;
                    a[last] = base + kk * (ll)p;
                }
            }
            (void)need;
        } else if (roll < forcedF + (1.0 - forcedF) * trapF) {
            // trap: one pile's size is an exact multiple of (max(S_i)+1) -- the period
            // the classic "remove 1..C" formula would predict, generally wrong here.
            int j = rnd.next(0, N - 1);
            for (int i = 0; i < N; i++) {
                if (i == j) {
                    ll M = rnd.next(0LL, VH);
                    ll modv = maxSi[i] + 1;
                    ll t = max(2LL, M / modv);
                    a[i] = t * modv;
                } else {
                    a[i] = (rnd.next(0, 1) == 0) ? rnd.next(0LL, smallHi) : rnd.next(0LL, VH);
                }
            }
        } else {
            // organic random
            for (int i = 0; i < N; i++) {
                a[i] = (rnd.next(0, 2) == 0) ? rnd.next(0LL, smallHi) : rnd.next(0LL, VH);
            }
        }
        for (int i = 0; i < N; i++) printf("%lld%c", a[i], i + 1 == N ? '\n' : ' ');
    }
    return 0;
}
