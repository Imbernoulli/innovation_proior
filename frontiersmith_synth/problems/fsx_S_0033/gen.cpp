#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Museum audio-beacon channel planning: weighted minimum-conflict k-coloring.
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example-scale sanity)
//   testId 10 -> large, dense, heavy-tailed annoyance weights.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, k, avgdeg;
    if (testId == 1) {
        n = 8; k = 3; avgdeg = 3;
    } else {
        n = 150 * testId;                 // 300 .. 1500
        k = 4 + (testId % 8);             // 4 .. 11
        avgdeg = 12 + 2 * (testId % 6);   // 12 .. 22
    }

    long long m = (long long)n * avgdeg / 2;
    if (m < 1) m = 1;

    // heavy-tailed annoyance: mostly light, occasionally a very loud pair.
    double heavyProb = 0.08 + 0.02 * (testId % 3); // 0.08 .. 0.12

    vector<array<int,3>> edges; // u, v, w
    edges.reserve(m + 8);

    auto pushEdge = [&](int u, int v) {
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb) w = rnd.next(30, 100);
        else                                w = rnd.next(1, 12);
        edges.push_back({u, v, w});
    };

    // random overlap pairs
    for (long long e = 0; e < m; e++) {
        int u = rnd.next(1, n);
        int v = rnd.next(1, n);
        if (u == v) { e--; continue; }
        pushEdge(u, v);
    }

    // Forced same-residue overlaps: galleries 1, 1+k, 1+2k, ... all fall on the
    // SAME baseline channel (residue 0), so the channel-cycling baseline B is
    // guaranteed positive and clearly improvable. Heavy weights make it matter.
    int prev = 1, added = 0;
    for (int j = 1; added < 6; j++) {
        int g = 1 + j * k;
        if (g > n) break;
        edges.push_back({prev, g, rnd.next(40, 100)});
        prev = g;
        added++;
    }
    // Fallback (e.g. n < 1+k): connect two galleries that share residue 1.
    if (added == 0 && n >= 2) {
        // gallery 2 and gallery 2+k if it exists, else 1 and any (still positive risk),
        // but with n>=2 and k>=... just force a duplicate of an existing pair on same res.
        // Safest: connect gallery 1 to itself-adjacent same-channel partner if none exists,
        // fall back to a heavy self-consistent pair using ids with residue 0 modulo k.
        for (int g = 1 + k; g <= n && added < 1; g += k) {
            edges.push_back({1, g, rnd.next(40, 100)});
            added++;
        }
    }

    // shuffle so edge order carries no structural hint
    shuffle(edges.begin(), edges.end());

    int M = (int)edges.size();
    printf("%d %d %d\n", n, k, M);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
