#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

/*
 * Balanced minimum k-cut (balanced graph partitioning) generator.
 *
 * Plants k communities, each of size n/k, whose membership is a RANDOM PERMUTATION
 * of the hive indices (so the community structure is uncorrelated with the index
 * blocks the checker uses as its reference).  Heavy "drift" records connect hives in
 * the SAME planted community; light noise records connect hives in DIFFERENT planted
 * communities.  The optimum recovers the communities (leaving only noise crossing
 * between yards); the index-block reference cut is mediocre.
 *
 * testId 1 is a tiny example-scale instance; sizes grow to a large, dense instance
 * by testId 10, sweeping k in {2..6} and the heavy/noise mix.
 */

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- difficulty / structure ladder -------------------------------------
    // n (rounded to a multiple of k), k, target #records, p_heavy, weight bands.
    int nBase, k, mTarget;
    double pHeavy;
    int hLo, hHi, gLo, gHi; // heavy band [hLo,hHi], noise band [gLo,gHi]
    switch (testId) {
        case 1:  nBase=6;    k=2; mTarget=6;      pHeavy=0.85; hLo=15; hHi=25; gLo=1; gHi=3;  break;
        case 2:  nBase=60;   k=2; mTarget=400;    pHeavy=0.80; hLo=40; hHi=100; gLo=1; gHi=6;  break;
        case 3:  nBase=180;  k=3; mTarget=2500;   pHeavy=0.75; hLo=30; hHi=90; gLo=1; gHi=8;  break;
        case 4:  nBase=400;  k=4; mTarget=9000;   pHeavy=0.72; hLo=20; hHi=80; gLo=1; gHi=10; break;
        case 5:  nBase=800;  k=2; mTarget=30000;  pHeavy=0.70; hLo=20; hHi=70; gLo=1; gHi=12; break;
        case 6:  nBase=1200; k=3; mTarget=60000;  pHeavy=0.68; hLo=15; hHi=60; gLo=1; gHi=12; break;
        case 7:  nBase=1800; k=5; mTarget=110000; pHeavy=0.66; hLo=15; hHi=55; gLo=1; gHi=14; break;
        case 8:  nBase=2400; k=6; mTarget=160000; pHeavy=0.64; hLo=12; hHi=50; gLo=1; gHi=15; break;
        case 9:  nBase=3000; k=4; mTarget=190000; pHeavy=0.62; hLo=12; hHi=45; gLo=1; gHi=16; break;
        default: nBase=3000; k=6; mTarget=200000; pHeavy=0.60; hLo=10; hHi=40; gLo=1; gHi=18; break;
    }

    // round n up to a multiple of k
    int n = ((nBase + k - 1) / k) * k;
    int s = n / k; // hives per community / yard

    // cap m by the number of available distinct pairs
    long long maxPairs = (long long)n * (n - 1) / 2;
    long long m = min<long long>(mTarget, maxPairs);
    m = min<long long>(m, 200000);

    // ---- planted community membership: a random permutation -----------------
    // comm[hive] in [0,k); each community has exactly s members.
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i; // 0-based hive ids
    shuffle(perm.begin(), perm.end());
    vector<int> comm(n);
    // members of each community (0-based hive ids)
    vector<vector<int>> members(k);
    for (int i = 0; i < n; i++) {
        int c = i / s;            // block of the permutation -> community
        int h = perm[i];          // hive id (0-based)
        comm[h] = c;
        members[c].push_back(h);
    }

    // ---- sample records without duplicates -----------------------------------
    set<pair<int,int>> used;
    vector<array<long long,3>> edges; // (u,v,w) with u<v, 1-based
    edges.reserve((size_t)m);

    long long heavyTarget = (long long)llround((double)m * pHeavy);
    long long heavyCount = 0, noiseCount = 0;
    long long attempts = 0;
    long long attemptCap = m * 40 + 1000;

    auto tryAdd = [&](int a, int b, int wlo, int whi) -> bool {
        if (a == b) return false;
        if (a > b) swap(a, b);
        auto key = make_pair(a, b);
        if (used.count(key)) return false;
        used.insert(key);
        int w = rnd.next(wlo, whi);
        edges.push_back({(long long)(a + 1), (long long)(b + 1), (long long)w});
        return true;
    };

    while ((long long)edges.size() < m && attempts < attemptCap) {
        attempts++;
        bool wantHeavy = (heavyCount < heavyTarget);
        // if we've hit the heavy target, force noise; and vice versa near the end
        if (!wantHeavy) {
            // noise: two hives from different communities
            if (k < 2) continue;
            int c1 = rnd.next(0, k - 1);
            int c2 = rnd.next(0, k - 1);
            if (c1 == c2) continue;
            int a = members[c1][rnd.next(0, (int)members[c1].size() - 1)];
            int b = members[c2][rnd.next(0, (int)members[c2].size() - 1)];
            if (tryAdd(a, b, gLo, gHi)) noiseCount++;
        } else {
            // heavy: two hives from the same community
            int c = rnd.next(0, k - 1);
            if ((int)members[c].size() < 2) continue;
            int ia = rnd.next(0, (int)members[c].size() - 1);
            int ib = rnd.next(0, (int)members[c].size() - 1);
            int a = members[c][ia];
            int b = members[c][ib];
            if (tryAdd(a, b, hLo, hHi)) heavyCount++;
        }
    }

    // If we couldn't reach m (dense small case), that's fine: emit what we have,
    // but guarantee at least one record so the instance is non-empty.
    if (edges.empty()) {
        // extremely small fallback: connect hive1-hive2 if possible
        if (n >= 2) edges.push_back({1, 2, (long long)rnd.next(hLo, hHi)});
    }

    long long realM = (long long)edges.size();

    // shuffle record order so it isn't grouped by community
    shuffle(edges.begin(), edges.end());

    // ---- output --------------------------------------------------------------
    printf("%d %lld %d\n", n, realM, k);
    for (auto &e : edges) {
        printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    }
    return 0;
}
