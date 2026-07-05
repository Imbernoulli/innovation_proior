// Generator for "Grid Hardening Attack" (interdiction-effective-resistance).
// testId is a difficulty/structure ladder: 1 = tiny worked example; then meshes, random
// grids, and adversarial structures (TRAP parallel-shortcut, PLANTED snake, NEEDLE detour),
// scaling up to the constraint envelope (n<=600, m<=4000) on the largest tests.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct E { int u, v, r, c; };

static int N;
static vector<E> es;

static void addE(int u, int v, int r, int c) {
    if (u == v) return;
    es.push_back({u, v, r, c});
}
static void addRnd(int u, int v) { addE(u, v, rnd.next(1, 9), rnd.next(1, 9)); }

// W x H mesh, node id = row*W + col + 1 (1-indexed), row in [0,H), col in [0,W).
static void mesh(int W, int H) {
    N = W * H;
    auto id = [&](int r, int c) { return r * W + c + 1; };
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            if (c + 1 < W) addRnd(id(r, c), id(r, c + 1));
            if (r + 1 < H) addRnd(id(r, c), id(r + 1, c));
        }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    N = 0; es.clear();
    int s = 1, t = 1;
    double budgetFrac = 0.30;

    if (testId == 1) {
        // Fixed tiny worked example from the statement.
        N = 4; s = 1; t = 4;
        addE(1, 2, 1, 1); addE(2, 4, 1, 1); addE(1, 3, 1, 1);
        addE(3, 4, 1, 1); addE(1, 4, 1, 1);
        long long B = 2;
        printf("%d %d %d %d %lld\n", N, (int)es.size(), s, t, B);
        for (auto& e : es) printf("%d %d %d %d\n", e.u, e.v, e.r, e.c);
        return 0;
    } else if (testId == 2) {
        mesh(4, 4); s = 1; t = N;                        // small mesh
    } else if (testId == 3) {
        // small random connected graph (spanning path + chords)
        N = 30;
        for (int i = 1; i < N; i++) addRnd(i, i + 1);
        for (int k = 0; k < 40; k++) addRnd(rnd.next(1, N), rnd.next(1, N));
        s = 1; t = N;
    } else if (testId == 4) {
        // TRAP: a bundle of cheap low-resistance direct shortcuts in parallel with an
        // expensive high-resistance backbone path. Cutting the "shortest" (a single
        // shortcut) barely moves R; the win is thinning the whole bundle to force current
        // onto the long backbone -- but shortcuts and backbone are hidden among noise chords.
        int L = 40;                                       // backbone length
        N = L;
        for (int i = 1; i < L; i++) addE(i, i + 1, 6, 8); // expensive backbone
        for (int k = 0; k < 12; k++) addE(1, L, 1, 1);    // cheap parallel shortcuts
        for (int k = 0; k < 30; k++) addRnd(rnd.next(1, N), rnd.next(1, N)); // noise
        s = 1; t = L; budgetFrac = 0.25;
    } else if (testId == 5) {
        // PLANTED: a mesh with a good hidden structure; medium size.
        mesh(8, 8); s = 1; t = N;
        for (int k = 0; k < 20; k++) addRnd(rnd.next(1, N), rnd.next(1, N)); // extra chords
    } else if (testId == 6) {
        // NEEDLE: a low-resistance mesh whose only high-resistance detour is one long
        // chain of r=9 lines; routing current through the needle maximizes R.
        int W = 9, H = 9;
        mesh(W, H); s = 1; t = N;
        int prev = t;                                     // build a detour hanging off t
        for (int i = 0; i < 20; i++) {
            int nn = ++N;
            addE(prev, nn, 9, 2);
            prev = nn;
        }
        addE(prev, s, 9, 2);                              // close the needle loop back to s
    } else if (testId == 7) {
        mesh(12, 12); s = 1; t = N;
        for (int k = 0; k < 40; k++) addRnd(rnd.next(1, N), rnd.next(1, N));
    } else if (testId == 8) {
        // dense-ish random graph, medium n
        N = 200;
        for (int i = 1; i < N; i++) addRnd(i, i + 1);
        for (int k = 0; k < 900; k++) addRnd(rnd.next(1, N), rnd.next(1, N));
        s = 1; t = N;
    } else if (testId == 9) {
        mesh(20, 20); s = 1; t = N;                       // n=400
        for (int k = 0; k < 120; k++) addRnd(rnd.next(1, N), rnd.next(1, N));
    } else {
        // testId 10: largest -- fill the envelope. 24x25 mesh (n=600) plus random chords
        // and a planted parallel bundle across the source corner.
        mesh(24, 25); s = 1; t = N;                       // n=600
        for (int k = 0; k < 800; k++) addRnd(rnd.next(1, N), rnd.next(1, N));
        for (int k = 0; k < 20; k++) addE(1, 2, 1, 1);    // planted cheap bundle near s
    }

    // Guarantee s and t are connected: overlay a random spanning tree via union-find and
    // add any missing tree edges (cheap, high-ish resistance) so the base graph is connected.
    // (All the structures above are already connected, but this makes it robust.)
    {
        vector<int> par(N + 1);
        for (int i = 0; i <= N; i++) par[i] = i;
        function<int(int)> find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
        for (auto& e : es) { int a = find(e.u), b = find(e.v); if (a != b) par[a] = b; }
        // link everything into one component if needed
        for (int i = 2; i <= N; i++) {
            int a = find(i - 1), b = find(i);
            if (a != b) { addE(i - 1, i, rnd.next(3, 9), rnd.next(3, 9)); par[a] = b; }
        }
    }

    // clamp edge count to the envelope
    if ((int)es.size() > 4000) es.resize(4000);

    long long total = 0;
    for (auto& e : es) total += e.c;
    long long B = (long long)floor(total * budgetFrac);
    if (B >= total) B = total - 1;
    if (B < 1) B = 1;

    printf("%d %d %d %d %lld\n", N, (int)es.size(), s, t, B);
    for (auto& e : es) printf("%d %d %d %d\n", e.u, e.v, e.r, e.c);
    return 0;
}
