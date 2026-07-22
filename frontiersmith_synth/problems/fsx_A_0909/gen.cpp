// gen.cpp -- generator for "Slitting Coils When Knives Crawl, Not Jump"
#include "testlib.h"
#include <vector>
#include <algorithm>
#include <cstdio>
using namespace std;
typedef long long ll;

// Print one test case: W K m / S_1..S_K / n / (width_i qty_i)*n / maxSetups setupCost penalty
static void emit(ll W, int K, ll m, vector<ll> S, vector<pair<ll,ll>> types,
                  ll maxSetups, ll setupCost, ll penalty) {
    printf("%lld %d %lld\n", W, K, m);
    for (int j = 0; j < K; j++) printf("%lld%c", S[j], j + 1 == K ? '\n' : ' ');
    printf("%d\n", (int)types.size());
    for (auto &t : types) printf("%lld %lld\n", t.first, t.second);
    printf("%lld %lld %lld\n", maxSetups, setupCost, penalty);
}

// number of copies of `width` that a K-knife pattern can usefully hold (mirrors chk.cc)
static ll usefulCopies(ll W, int K, ll width) {
    return min((ll)K, (W - 1) / width);
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Tiny hand-verifiable case (matches the statement's worked example exactly).
        emit(60, 2, 25, {20, 40}, {{15, 3}, {25, 2}}, 10, 50, 6);
        return 0;
    }

    if (testId == 5 || testId == 6 || testId == 7) {
        // PLANTED interleaved trap: a LEFT cluster (small width, K copies/run) and a
        // RIGHT cluster (larger width, K-1 copies/run) both near-maximal per-run value,
        // but far apart in knife-position space. Values are constructed to interleave
        // strictly L,R,L,R,... so a value-only strategy crosses the width gap 7 times;
        // a reachability-aware strategy groups same-cluster visits and crosses once.
        int K = 3;
        ll W = 900 + (ll)(testId - 5) * 150;
        ll m = max((ll)6, W / 45);
        vector<ll> S;
        for (int j = 0; j < K; j++) S.push_back((W * (j + 1)) / (K + 1));
        double scale = (double)W / 900.0;
        ll lw[8] = {299, 293, 286, 280, 274, 268, 262, 256};
        ll rw[8] = {445, 435, 425, 415, 405, 395, 385, 375};
        vector<pair<ll, ll>> types;
        for (int i = 0; i < 8; i++) {
            ll w = max((ll)3, (ll)llround(lw[i] * scale));
            ll q = 24 + i * 2 + (testId - 5) * 3;
            types.push_back({w, q});
            ll w2 = max((ll)3, (ll)llround(rw[i] * scale));
            ll q2 = 14 + i + (testId - 5) * 2;
            types.push_back({w2, q2});
        }
        ll setupCost = max((ll)10, (ll)llround(W * 0.5));
        ll penalty = 6;
        ll totalQty = 0;
        for (auto &t : types) totalQty += t.second;
        ll maxSetups = totalQty + (ll)types.size() * 20 + 40;
        emit(W, K, m, S, types, maxSetups, setupCost, penalty);
        return 0;
    }

    // Default: organic randomized instances, growing with testId (2,3,4,8,9,10).
    // Roughly scale W, K, n with testId so later tests fill the constraint envelope.
    int rank = testId <= 4 ? testId : testId - 3;      // 2,3,4 -> 2,3,4 ; 8,9,10 -> 5,6,7
    ll W = 120 + (ll)rank * 130 + rnd.next(-20, 40);
    W = max((ll)80, W);
    int K = 2 + rnd.next(0, min(4, 1 + rank / 2));
    ll m = max((ll)4, W / (ll)rnd.next(8, 22));
    vector<ll> S;
    ll cur = 0;
    for (int j = 0; j < K; j++) {
        ll lo = max((ll)1, cur + 1);
        ll hi = W - 1 - (K - j - 1);
        if (hi < lo) hi = lo;
        cur = rnd.next(lo, hi);
        S.push_back(cur);
    }
    int n = min((int)8, 3 + rank);
    vector<ll> pool;
    for (ll w = 3; w <= W - K - 2; w++) pool.push_back(w);
    if ((int)pool.size() < n) n = max(1, (int)pool.size());
    vector<int> idx(pool.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
    for (int i = 0; i < n; i++) {
        int j = rnd.next(i, (int)idx.size() - 1);
        swap(idx[i], idx[j]);
    }
    vector<pair<ll, ll>> types;
    for (int i = 0; i < n; i++) {
        ll w = pool[idx[i]];
        ll q = rnd.next(4, 10 + rank * 6);
        types.push_back({w, q});
    }
    // On the large "needle" tests (9,10), inject one high-quantity, awkward-width type
    // to keep the constraint envelope filled and the adversarial regime present.
    if (testId == 9 || testId == 10) {
        ll w = max((ll)5, W - K - 3 - rnd.next(0, 5));
        ll q = 15 + rnd.next(0, 20);
        types.push_back({w, q});
        n++;
    }
    ll setupCost = max((ll)5, (ll)llround(W * (0.35 + 0.3 * rnd.next(0, 100) / 100.0)));
    ll penalty = 6;
    ll totalQty = 0;
    for (auto &t : types) totalQty += t.second;
    ll maxSetups = totalQty + (ll)n * 15 + 30;
    emit(W, K, m, S, types, maxSetups, setupCost, penalty);
    return 0;
}
