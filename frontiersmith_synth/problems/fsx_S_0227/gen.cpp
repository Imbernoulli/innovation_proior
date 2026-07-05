#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- generator for Metro Busker Permit Scheduling (max-weight independent set on a
// general conflict graph). testId 1..10 is a difficulty/structure ladder:
//   1  tiny example-scale sanity (planted, few conflicts)
//   2  small sparse random
//   3  medium sparse random
//   4  dense random
//   5  larger dense random
//   6  union-of-cliques clusters (interchange spots)
//   7  larger union-of-cliques clusters, skewed cluster values
//   8  planted heavy independent set hidden inside a dense conflict graph
//   9  power-law skewed weights on a dense graph
//   10 large mixed adversarial (clusters + dense noise + power-law weights)
// Conflicts printed as an unordered edge list; duplicates are allowed by the statement but
// we avoid trivial self-loops.  Every .in stays well under 20 MB.

static const long long MAXM = 60000;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int N;
    vector<int> w;                 // 1-indexed via w[i-1]? we keep 0-indexed internal, print i+1
    vector<pair<int,int>> edges;

    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        edges.push_back({a, b});
    };

    // helper: dense-ish random edges avoiding self loops (duplicates allowed)
    auto randomEdges = [&](int n, long long m, function<bool(int,int)> allowed) {
        long long added = 0, guard = 0;
        long long cap = m * 8 + 1000;
        while (added < m && guard < cap) {
            guard++;
            int a = rnd.next(0, n - 1);
            int b = rnd.next(0, n - 1);
            if (a == b) continue;
            if (!allowed(a, b)) continue;
            addEdge(a, b);
            added++;
        }
    };

    if (t == 1) {
        // tiny planted: a small independent set of medium pitches vs one hub pitch
        N = 6;
        w.assign(N, 0);
        w[0] = 10;                 // hub
        for (int i = 1; i < N; i++) w[i] = 6;
        for (int i = 1; i < N; i++) addEdge(0, i);  // hub conflicts with all others
    } else if (t == 2 || t == 3) {
        // sparse random
        N = (t == 2) ? 200 : 700;
        long long M = (t == 2) ? 600 : 4000;
        w.assign(N, 0);
        for (int i = 0; i < N; i++) w[i] = rnd.next(1, 1000);
        randomEdges(N, M, [](int,int){ return true; });
    } else if (t == 4 || t == 5) {
        // dense random
        N = (t == 4) ? 900 : 1400;
        long long M = (t == 4) ? 35000 : MAXM;
        w.assign(N, 0);
        for (int i = 0; i < N; i++) w[i] = rnd.next(1, 1000);
        randomEdges(N, M, [](int,int){ return true; });
    } else if (t == 6 || t == 7) {
        // union of cliques: each clique = an interchange spot; pick one per clique
        N = (t == 6) ? 1500 : 2400;
        int g = (t == 6) ? 10 : 12;             // clique size
        w.assign(N, 0);
        if (t == 6) {
            for (int i = 0; i < N; i++) w[i] = rnd.next(1, 1000);
        } else {
            // skewed: some cliques carry much higher potential than others
            for (int i = 0; i < N; i++) {
                int hi = (rnd.next(0, 3) == 0) ? 1000 : 300;
                w[i] = rnd.next(1, hi);
            }
        }
        for (int base = 0; base < N; base += g) {
            int end = min(N, base + g);
            for (int a = base; a < end; a++)
                for (int b = a + 1; b < end; b++)
                    addEdge(a, b);
            if ((long long)edges.size() > MAXM) break;
        }
        // trim to cap if needed
        if ((long long)edges.size() > MAXM) edges.resize(MAXM);
    } else if (t == 8) {
        // planted heavy independent set hidden inside a dense conflict graph
        N = 2500;
        int K = N / 8;                          // size of hidden independent set
        w.assign(N, 0);
        vector<char> inS(N, 0);
        // choose first K indices (after a shuffle) as the planted independent set
        vector<int> perm(N);
        for (int i = 0; i < N; i++) perm[i] = i;
        shuffle(perm.begin(), perm.end());
        for (int i = 0; i < K; i++) inS[perm[i]] = 1;
        for (int i = 0; i < N; i++)
            w[i] = inS[i] ? rnd.next(800, 1000) : rnd.next(1, 250);
        // dense edges everywhere EXCEPT never inside the planted set
        randomEdges(N, MAXM, [&](int a, int b){ return !(inS[a] && inS[b]); });
    } else if (t == 9) {
        // power-law skewed weights on a dense graph
        N = 3000;
        w.assign(N, 0);
        for (int i = 0; i < N; i++) {
            // heavy-tailed: mostly small, a few large
            double u = rnd.next(1, 1000000) / 1000000.0;
            int val = (int)floor(1.0 / (u + 0.001));
            val = max(1, min(1000, val));
            w[i] = val;
        }
        randomEdges(N, MAXM, [](int,int){ return true; });
    } else {
        // t == 10: large mixed adversarial
        N = 5000;
        w.assign(N, 0);
        for (int i = 0; i < N; i++) {
            double u = rnd.next(1, 1000000) / 1000000.0;
            int val = (int)floor(1.0 / (u + 0.002));
            val = max(1, min(1000, val));
            w[i] = val;
        }
        // clusters over the first half
        int half = N / 2;
        int g = 14;
        for (int base = 0; base < half; base += g) {
            int end = min(half, base + g);
            for (int a = base; a < end; a++)
                for (int b = a + 1; b < end; b++) {
                    addEdge(a, b);
                    if ((long long)edges.size() > MAXM * 3 / 5) goto donecl;
                }
        }
        donecl:;
        // dense random noise over the whole graph up to the cap
        {
            long long remaining = MAXM - (long long)edges.size();
            if (remaining > 0)
                randomEdges(N, remaining, [](int,int){ return true; });
        }
        if ((long long)edges.size() > MAXM) edges.resize(MAXM);
    }

    // safety: guarantee at least one edge and at most MAXM edges
    if (edges.empty()) addEdge(0, 1 % N);
    if ((long long)edges.size() > MAXM) edges.resize(MAXM);

    // shuffle edge order so structure is not trivially readable
    shuffle(edges.begin(), edges.end());

    long long M = (long long)edges.size();
    printf("%d %lld\n", N, M);
    for (int i = 0; i < N; i++) {
        printf("%d", w[i]);
        putchar(i + 1 == N ? '\n' : ' ');
    }
    for (auto& e : edges)
        printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
