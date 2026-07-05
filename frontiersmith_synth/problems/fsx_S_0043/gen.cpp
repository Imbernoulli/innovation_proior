#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // Difficulty / structure ladder: N grows with testId.
    int Ns[10] = {5, 12, 24, 48, 80, 130, 200, 300, 430, 580};
    int idx = testId;
    if (idx < 1) idx = 1;
    if (idx > 10) idx = 10;
    int N = Ns[idx - 1];

    // Collar caps. Mixed distribution on every test (25% cap 2, 40% cap 3, 35% cap 4)
    // so a degree-bounded spanning mesh always exists comfortably (no all-tight test),
    // while the cap-2 nodes still bite. The bias shifts slightly with testId for variety.
    vector<int> cap(N + 1);
    int p2 = 20 + (testId % 3) * 5;        // 20..30 % of nodes get cap 2
    for (int i = 1; i <= N; i++) {
        int r = rnd.next(0, 99);
        if (r < p2) cap[i] = 2;
        else if (r < p2 + 40) cap[i] = 3;
        else cap[i] = 4;
    }

    // Weight ranges: guaranteed backbone links are EXPENSIVE so the chain baseline is
    // weak; extra terrain links span a broad, cheaper band so much better meshes exist.
    // Both B and F scale linearly with N, so the achievable ratio is N-independent
    // (no saturation on large tests). The cheap band varies with testId to diversify.
    int cheapLo = 30 + (testId % 4) * 15;  // 30 .. 75
    int cheapHi = 500 + (testId % 5) * 60; // 500 .. 740
    int backLo = 700, backHi = 1200;

    // Build the guaranteed backbone chain (i, i+1) with expensive costs.
    set<pair<int,int>> used;
    vector<tuple<int,int,int>> edges;
    for (int i = 1; i <= N - 1; i++) {
        int w = rnd.next(backLo, backHi);
        edges.push_back(make_tuple(i, i + 1, w));
        used.insert(make_pair(i, i + 1));
    }

    // Extra cheap terrain links. Aim for a moderate average degree.
    long long maxPairs = (long long)N * (N - 1) / 2 - (N - 1);
    int factor = rnd.next(3, 6);
    long long target = (long long)factor * N;
    if (target > maxPairs) target = maxPairs;

    long long added = 0, tries = 0, tryCap = target * 40 + 1000;
    while (added < target && tries < tryCap) {
        tries++;
        int u = rnd.next(1, N);
        int v = rnd.next(1, N);
        if (u == v) continue;
        if (u > v) swap(u, v);
        if (used.count(make_pair(u, v))) continue;
        used.insert(make_pair(u, v));
        int w = rnd.next(cheapLo, cheapHi);
        edges.push_back(make_tuple(u, v, w));
        added++;
    }

    // Shuffle the edge list so the backbone is not trivially identifiable by position.
    shuffle(edges.begin(), edges.end());

    int M = (int)edges.size();
    printf("%d %d\n", N, M);
    for (int i = 1; i <= N; i++) {
        printf("%d%c", cap[i], i == N ? '\n' : ' ');
    }
    for (auto &e : edges) {
        printf("%d %d %d\n", get<0>(e), get<1>(e), get<2>(e));
    }
    return 0;
}
