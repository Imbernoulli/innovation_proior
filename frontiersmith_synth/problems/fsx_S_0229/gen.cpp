#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

/*
  Museum Gallery Tour -- weighted-budget doorway interdiction.
  Grid museum floor plan; a few CHEAP "guided tour" staircase routes carry the
  short walk; all other doorways are slow/expensive.  Each doorway also has a
  CLOSURE COST (staffing to rope it off).  The curator may close doorways with
  total closure cost <= K to lengthen the shortest Entrance->Star-Exhibit walk,
  keeping the two connected.  (Budgeted -> knapsack-flavoured interdiction.)

  testId is a difficulty/structure ladder: testId 1 tiny (example scale),
  growing to a large, denser museum by testId 10.
*/

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int side = 3 + 2 * (testId - 1);   // 3,5,7,...,21
    int R = side, C = side;
    int n = R * C;
    auto node = [&](int i, int j) { return i * C + j + 1; };

    // key for a grid doorway (unordered pair)
    auto key = [&](int a, int b) -> long long {
        if (a > b) swap(a, b);
        return (long long)a * 1000000LL + b;
    };

    // collect grid doorways (right + down)
    vector<pair<int,int>> gridE;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) gridE.push_back({node(i, j), node(i, j + 1)});
            if (i + 1 < R) gridE.push_back({node(i, j), node(i + 1, j)});
        }

    // mark a few cheap monotone (right/down) staircase tour routes s->t
    set<long long> cheap;
    int Q = 2 + (testId % 3);           // 2..4 cheap routes
    for (int q = 0; q < Q; q++) {
        int i = 0, j = 0;
        while (i != R - 1 || j != C - 1) {
            int ni = i, nj = j;
            bool canR = (j + 1 < C), canD = (i + 1 < R);
            if (canR && (!canD || rnd.next(0, 1) == 0)) nj = j + 1;
            else ni = i + 1;
            cheap.insert(key(node(i, j), node(ni, nj)));
            i = ni; j = nj;
        }
    }

    int expLo = 10, expHi = 25;         // slow doorways
    int cheapLo = 1, cheapHi = 3;       // guided-tour doorways

    struct E { int u, v, w, c; };
    vector<E> edges;
    for (auto& p : gridE) {
        int w;
        if (cheap.count(key(p.first, p.second)))
            w = rnd.next(cheapLo, cheapHi);
        else
            w = rnd.next(expLo, expHi);
        int c = rnd.next(1, 4);         // closure cost
        edges.push_back({p.first, p.second, w, c});
    }

    // a few extra (slow) shortcut doorways add parallel choices / open-endedness
    int extra = testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(expLo, expHi);
        int c = rnd.next(1, 4);
        edges.push_back({a, b, w, c});
    }

    int s = node(0, 0);                 // Grand Entrance
    int t = node(R - 1, C - 1);         // Star Exhibit
    int K = 2 * side;                   // closure-cost budget

    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, K);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e.u, e.v, e.w, e.c);
    return 0;
}
