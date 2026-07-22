#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Starship refinery: typed module catalog, footprint budget, byproduct self-loops.
// Types 0..K-1 form a fixed conversion chain (0 = raw feedstock, K-1 = target product Z).
// Every module converts type t -> type t+1 with a ratio, costs footprint, and has a
// per-type instance limit. Some modules ALSO emit a byproduct back into their OWN input
// type t (an autocatalytic self-loop): routing that byproduct back multiplies the
// module's effective throughput by 1/(1-g) where g is the byproduct ratio.

struct Mod {
    int inT, outT, outR, byT, byR, fp, cnt;
};

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- difficulty ladder ----
    int K;               // number of resource types (chain length K-1 hops)
    int nAnchors;         // number of hops that get a "loop" module option
    int gainLoLo, gainLoHi, gainHiLo, gainHiHi; // gain ranges: first anchor uses Hi range (trap), rest alternate
    int noiseCount;

    switch (testId) {
        case 1:  K = 3; nAnchors = 1; gainHiLo = 60; gainHiHi = 70; gainLoLo = 20; gainLoHi = 30; noiseCount = 0;  break;
        case 2:  K = 4; nAnchors = 1; gainHiLo = 20; gainHiHi = 35; gainLoLo = 10; gainLoHi = 20; noiseCount = 2;  break;
        case 3:  K = 4; nAnchors = 1; gainHiLo = 62; gainHiHi = 72; gainLoLo = 15; gainLoHi = 25; noiseCount = 3;  break;
        case 4:  K = 5; nAnchors = 1; gainHiLo = 45; gainHiHi = 55; gainLoLo = 10; gainLoHi = 20; noiseCount = 5;  break;
        case 5:  K = 5; nAnchors = 2; gainHiLo = 60; gainHiHi = 70; gainLoLo = 15; gainLoHi = 25; noiseCount = 8;  break;
        case 6:  K = 6; nAnchors = 1; gainHiLo = 62; gainHiHi = 72; gainLoLo = 10; gainLoHi = 20; noiseCount = 15; break;
        case 7:  K = 6; nAnchors = 2; gainHiLo = 48; gainHiHi = 56; gainLoLo = 45; gainLoHi = 53; noiseCount = 20; break;
        case 8:  K = 7; nAnchors = 2; gainHiLo = 52; gainHiHi = 60; gainLoLo = 28; gainLoHi = 36; noiseCount = 90;  break;
        case 9:  K = 7; nAnchors = 3; gainHiLo = 48; gainHiHi = 56; gainLoLo = 18; gainLoHi = 26; noiseCount = 180; break;
        default: K = 8; nAnchors = 3; gainHiLo = 45; gainHiHi = 53; gainLoLo = 38; gainLoHi = 46; noiseCount = 330; break;
    }

    int hops = K - 1;

    // pick distinct anchor hops
    vector<int> hopIdx(hops);
    for (int i = 0; i < hops; i++) hopIdx[i] = i;
    for (int i = hops - 1; i > 0; i--) swap(hopIdx[i], hopIdx[rnd.next(0, i)]);
    vector<char> isAnchor(hops, 0);
    vector<int> anchorList;
    for (int i = 0; i < min(nAnchors, hops); i++) { isAnchor[hopIdx[i]] = 1; anchorList.push_back(hopIdx[i]); }
    sort(anchorList.begin(), anchorList.end());

    vector<Mod> mods;
    vector<int> cheapFp(hops), loopFp(hops, 0);
    bool hasLoop[64] = {false};
    int loopGain[64] = {0};

    for (int t = 0; t < hops; t++) {
        int cheapR = rnd.next(45, 66);
        int cheapF = rnd.next(2, 4);
        int cheapC = rnd.next(4, 8);
        mods.push_back({t, t + 1, cheapR, -1, 0, cheapF, cheapC});
        cheapFp[t] = cheapF;

        int premR = rnd.next(66, 72);
        int premF = cheapF + rnd.next(6, 12);
        int premC = rnd.next(2, 5);
        mods.push_back({t, t + 1, premR, -1, 0, premF, premC});

        if (isAnchor[t]) {
            int rank = 0;
            for (int a : anchorList) if (a == t) break; else rank++;
            int lo, hi;
            if (rank == 0) { lo = gainHiLo; hi = gainHiHi; }
            else if (rank % 2 == 1) { lo = gainLoLo; hi = gainLoHi; }
            else { lo = gainHiLo; hi = gainHiHi; }
            int g = rnd.next(lo, hi);
            int loopR = rnd.next(40, 68);
            int loopF = cheapF + rnd.next(10, 20);
            int loopC = rnd.next(1, 2);
            mods.push_back({t, t + 1, loopR, t, g, loopF, loopC});
            hasLoop[t] = true;
            loopGain[t] = g;
            loopFp[t] = loopF;
        }
    }

    // noise modules: redundant, DELIBERATELY dominated candidates (ratio ceiling
    // stays below the cheap/premium/loop options at every hop) appended AFTER
    // the structural modules, so they pad the catalog (needle-in-haystack search
    // cost) without ever being worth building -- and, crucially, they can never
    // out-scale the strong solution regardless of how many are generated.
    for (int i = 0; i < noiseCount; i++) {
        int t = rnd.next(0, hops - 1);
        int r = rnd.next(20, 48);
        int f = rnd.next(2, 28);
        int c = rnd.next(1, 10);
        if (rnd.next(0, 99) < 12) {
            int by = t; // occasional weak decoy self-loop
            int g = rnd.next(5, 28);
            mods.push_back({t, t + 1, r, by, g, f, c});
        } else {
            mods.push_back({t, t + 1, r, -1, 0, f, c});
        }
    }

    int M = (int)mods.size();

    // footprint budget: cover the cheap spine plus room for exactly ONE loop
    // investment (the single best-footprint-adjusted-gain candidate), plus a
    // little slack for one premium swap elsewhere. When >=2 anchors exist the
    // budget can fund only one, forcing a genuine selection among competing
    // loops instead of "always take every loop".
    long long base = 0;
    for (int t = 0; t < hops; t++) base += cheapFp[t];
    long long bestLoopFp = 0;
    for (int t = 0; t < hops; t++) if (hasLoop[t] && loopFp[t] > bestLoopFp) bestLoopFp = loopFp[t];
    long long slack = 6 + rnd.next(0, 6);
    long long S = base + bestLoopFp + slack;

    long long R = 600 + testId * 420 + rnd.next(0, 300);

    printf("%d %d %lld %lld\n", K, M, S, R);
    for (auto& md : mods) {
        printf("%d %d %d %d %d %d %d\n", md.inT, md.outT, md.outR, md.byT, md.byR, md.fp, md.cnt);
    }
    return 0;
}
