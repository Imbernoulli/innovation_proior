#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int u, v, c, l, r, a, b;
};

static int modp(int x, int p) {
    x %= p;
    if (x < 0) x += p;
    return x;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int Ns[10] = {12, 24, 42, 70, 105, 145, 190, 245, 305, 350};
    const int Ts[10] = {8, 10, 14, 18, 24, 30, 36, 42, 48, 50};
    const int Ps[10] = {11, 11, 13, 13, 17, 19, 19, 23, 29, 31};
    int N = Ns[testId - 1];
    int T = Ts[testId - 1];
    int P = Ps[testId - 1];

    int alpha = 2 + (testId * 3) % (P - 3);
    int beta = rnd.next(0, P - 1);
    vector<int> h(T);
    for (int t = 0; t < T; t++) h[t] = modp(beta + alpha * t, P);

    int O = 140 + 4 * testId;
    int R = 28 + testId;
    vector<int> w(N + 1, 1);
    for (int i = 2; i <= N; i++) w[i] = rnd.next(1, 3);

    vector<Edge> e;
    auto addEdge = [&](int u, int v, int c, int l, int r, int a, int b) {
        if (u == v) return;
        if (l < 0) l = 0;
        if (r >= T) r = T - 1;
        if (l > r) swap(l, r);
        e.push_back({u, v, c, l, r, modp(a, P), modp(b, P)});
    };

    // TRAP: the first N-1 links are a cost-1 connected fallback star. Their
    // residue is exactly h_t in every slot, so the cheapest connected subgraph
    // resonates everywhere.
    for (int v = 2; v <= N; v++) {
        addEdge(1, v, 1, 0, T - 1, beta, alpha);
    }

    vector<int> nodes;
    for (int v = 2; v <= N; v++) nodes.push_back(v);
    shuffle(nodes.begin(), nodes.end());

    int K = min((int)nodes.size(), max(2, min(18, 2 + testId + N / 85)));
    vector<vector<int>> cluster(K);
    for (int i = 0; i < (int)nodes.size(); i++) cluster[i % K].push_back(nodes[i]);
    for (int c = 0; c < K; c++) {
        if (cluster[c].empty()) continue;
        w[cluster[c][0]] += 2 + (c % 3);
    }

    int needle = -1;
    if (testId >= 5) {
        needle = nodes[rnd.next(0, (int)nodes.size() - 1)];
        w[needle] = 18 + 3 * testId;
    }

    // PLANTED modules: a few safe HQ bridges cover complementary time windows,
    // and zero-residue internal trees spread the clean residue through each
    // cluster. A strategy must buy both parts to clean many station-time pairs.
    for (int c = 0; c < K; c++) {
        if (cluster[c].empty()) continue;
        int root = cluster[c][0];
        int segs = 2 + (testId >= 4) + (testId >= 8);
        for (int s = 0; s < segs; s++) {
            int l = (s * T) / segs;
            int r = ((s + 1) * T) / segs - 1;
            if (s > 0) l--;
            if (s + 1 < segs) r++;
            int off = 2 + ((c * 5 + s * 3) % (P - 3));
            int cost = rnd.next(55, 105) + 6 * segs + (int)cluster[c].size() / 5;
            addEdge(1, root, cost, l, r, beta + off, alpha);
        }

        for (int i = 1; i < (int)cluster[c].size(); i++) {
            int parent = cluster[c][(i - 1) / 2];
            int v = cluster[c][i];
            int cost = rnd.next(22, 50);
            addEdge(parent, v, cost, 0, T - 1, 0, 0);
        }

        int chords = (int)cluster[c].size() / 5;
        for (int j = 0; j < chords; j++) {
            int u = cluster[c][rnd.next(0, (int)cluster[c].size() - 1)];
            int v = cluster[c][rnd.next(0, (int)cluster[c].size() - 1)];
            int off = (j % 2 == 0) ? 0 : (2 + rnd.next(0, P - 4));
            addEdge(u, v, rnd.next(28, 80), 0, T - 1, off, 0);
        }
    }

    // NEEDLE: one high-importance station has expensive clean relief links hidden
    // beside cheaper resonant decoys.
    if (needle != -1) {
        int segs = 2 + (testId >= 8);
        for (int s = 0; s < segs; s++) {
            int l = (s * T) / segs;
            int r = ((s + 1) * T) / segs - 1;
            int off = 3 + ((s + testId) % (P - 4));
            addEdge(1, needle, rnd.next(140, 230), l, r, beta + off, alpha);
        }
        addEdge(1, needle, 4, 0, T - 1, beta + 1, alpha);
    }

    int targetM = N - 1 + 3 * N + 45 * testId;
    if (testId >= 8) targetM += 180;
    targetM = min(targetM, 2780);

    // TRAP/NOISE fill: cheap full-window bad links, temporal random links, and
    // occasional useful short-window bridges. Large tests fill the stated envelope.
    while ((int)e.size() < targetM) {
        int typ = rnd.next(0, 99);
        if (typ < 28) {
            int v = rnd.next(2, N);
            int badOff = rnd.next(0, 1);
            addEdge(1, v, rnd.next(2, 9), 0, T - 1, beta + badOff, alpha);
        } else if (typ < 52) {
            int c = rnd.next(0, K - 1);
            if (cluster[c].empty()) continue;
            int root = cluster[c][0];
            int len = max(1, T / rnd.next(3, 6));
            int l = rnd.next(0, T - len);
            int off = 2 + rnd.next(0, P - 4);
            addEdge(1, root, rnd.next(80, 180), l, l + len - 1, beta + off, alpha);
        } else if (typ < 78) {
            int u = rnd.next(2, N);
            int v = rnd.next(2, N);
            if (u == v) continue;
            int len = rnd.next(1, T);
            int l = rnd.next(0, T - len);
            addEdge(u, v, rnd.next(25, 160), l, l + len - 1, rnd.next(0, P - 1), rnd.next(0, P - 1));
        } else {
            int c1 = rnd.next(0, K - 1);
            int c2 = rnd.next(0, K - 1);
            if (cluster[c1].empty() || cluster[c2].empty()) continue;
            int u = cluster[c1][rnd.next(0, (int)cluster[c1].size() - 1)];
            int v = cluster[c2][rnd.next(0, (int)cluster[c2].size() - 1)];
            if (u == v) continue;
            int off = rnd.next(0, P - 1);
            addEdge(u, v, rnd.next(18, 130), 0, T - 1, off, rnd.next(0, P - 1));
        }
    }

    printf("%d %d %d %d\n", N, (int)e.size(), T, P);
    printf("%d %d\n", O, R);
    for (int i = 1; i <= N; i++) {
        if (i > 1) printf(" ");
        printf("%d", w[i]);
    }
    printf("\n");
    for (int t = 0; t < T; t++) {
        if (t) printf(" ");
        printf("%d", h[t]);
    }
    printf("\n");
    for (const Edge& x : e) {
        printf("%d %d %d %d %d %d %d\n", x.u, x.v, x.c, x.l, x.r, x.a, x.b);
    }
    return 0;
}
