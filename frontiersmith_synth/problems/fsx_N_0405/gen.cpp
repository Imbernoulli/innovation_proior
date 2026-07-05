#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---------- difficulty / structure ladder ----------
    // testId 1 is tiny (example scale); sizes fill the envelope by testId 10.
    int S = 8 + (testId - 1) * 38;        // 8 .. 350
    int G = 6 + (testId - 1) * 30;        // 6 .. 276
    int T = 4 + (testId - 1) * 6;         // 4 .. 58

    int Cfull = 40, Cmin = 5;
    int Drain    = 8 + (testId % 4);      // 8 .. 11
    int Recharge = 6 + (testId % 3);      // 6 .. 8

    // adversarial flavour: 0 uniform, 1 TRAP, 2 NEEDLE, 3 PLANTED, 4 skewed
    int mode = testId % 5;

    // ---------- broad vs cheap sensor split ----------
    // A handful of "broad" sensors reach many targets but are EXPENSIVE per target;
    // the many "cheap" sensors reach few targets but are cheap per target. Every
    // target is reachable by >=2 cheap sensors, so a cheap rotating cover always
    // exists -- but a breadth-first greedy is lured into the costly broad sensors.
    int nBroad = 2 + (testId % 4);        // 2 .. 5
    if (nBroad > S - 3) nBroad = max(1, S - 3);
    int cheapStart = nBroad + 1;
    int nCheap = S - nBroad;
    if (nCheap < 2) { nBroad = S - 2; cheapStart = nBroad + 1; nCheap = 2; }

    set<pair<int,int>> incset;
    auto add = [&](int s, int t) {
        if (s >= 1 && s <= S && t >= 1 && t <= G) incset.insert({s, t});
    };

    // each target: 2..4 distinct cheap covers (rotation family)
    for (int t = 1; t <= G; t++) {
        int nc = min(nCheap, 2 + rnd.next(0, 2));
        set<int> chosen;
        int guard = 0;
        while ((int)chosen.size() < nc && guard++ < 60 * nc)
            chosen.insert(rnd.next(cheapStart, S));
        for (int s = cheapStart; s <= S && (int)chosen.size() < 2; s++) chosen.insert(s);
        for (int s : chosen) add(s, t);
    }

    // broad sensors reach a chunk of targets
    int reachBase = (mode == 1) ? max(4, G / 5) : max(4, G / 8);   // TRAP = broader lure
    for (int b = 1; b <= nBroad; b++) {
        int reach = min(G, reachBase + rnd.next(0, max(1, G / 16)));
        set<int> pts;
        int guard = 0;
        while ((int)pts.size() < reach && guard++ < 20 * reach)
            pts.insert(rnd.next(1, G));
        for (int t : pts) add(b, t);
    }
    // NEEDLE: one extra broad-but-still-pricey lure hiding among the noise
    if (mode == 2 && S > cheapStart) {
        int b = rnd.next(1, nBroad);
        int reach = min(G, max(4, G / 3));
        set<int> pts;
        int guard = 0;
        while ((int)pts.size() < reach && guard++ < 20 * reach)
            pts.insert(rnd.next(1, G));
        for (int t : pts) add(b, t);
    }

    // ---------- degrees, demands ----------
    vector<vector<int>> stg(S + 1);
    vector<int> deg(G + 1, 0);
    for (auto& pr : incset) { stg[pr.first].push_back(pr.second); deg[pr.second]++; }
    vector<long long> D(G + 1);
    for (int g = 1; g <= G; g++) D[g] = (long long)deg[g] * Cmin;   // all-on always feasible

    // ---------- costs ----------
    vector<int> c(S + 1);
    for (int s = 1; s <= S; s++) {
        if (s <= nBroad) {
            if (mode == 1)      c[s] = rnd.next(150, 200);   // TRAP: dearer broad
            else if (mode == 4) c[s] = rnd.wnext(90, 200, 2);
            else                c[s] = rnd.next(80, 180);
        } else {
            if (mode == 3)      c[s] = rnd.next(1, 5);        // PLANTED: very cheap partition
            else                c[s] = rnd.next(1, 10);
        }
        if (c[s] < 1) c[s] = 1;
        if (c[s] > 200) c[s] = 200;
    }

    // ---------- emit ----------
    printf("%d %d %d\n", S, G, T);
    printf("%d %d %d %d\n", Cfull, Cmin, Drain, Recharge);
    for (int g = 1; g <= G; g++) printf("%lld%c", D[g], g == G ? '\n' : ' ');
    for (int s = 1; s <= S; s++) {
        shuffle(stg[s].begin(), stg[s].end());
        printf("%d %d", c[s], (int)stg[s].size());
        for (int t : stg[s]) printf(" %d", t);
        printf("\n");
    }
    return 0;
}
