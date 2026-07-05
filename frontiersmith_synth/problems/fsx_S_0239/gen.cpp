#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Salmon Ladder Watch: maximum-weight independent set on a GENERAL conflict graph.
// Structure ladder over testId (1..10):
//   - n grows from tiny (example scale) to ~120.
//   - conflict density varies (sparse early, denser later).
//   - "hub" sites carry high data value AND high conflict degree, so naive
//     weight-greedy is tempted into low-yield choices -> genuinely open-ended.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 8 + 12 * (testId - 1);          // 8, 20, 32, ..., 116
    if (n > 120) n = 120;

    // base edge probability climbs with the ladder (sparse -> dense)
    double p = 0.10 + 0.028 * testId;       // ~0.13 .. ~0.38

    // fraction of "hub" (high-value, high-conflict) sites
    double hubFrac = 0.12 + 0.01 * (testId % 5);

    vector<int> w(n + 1);
    vector<char> isHub(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        if (rnd.next(0.0, 1.0) < hubFrac) {
            isHub[i] = 1;
            w[i] = rnd.next(70, 100);       // lucrative but conflict-prone
        } else {
            w[i] = rnd.next(1, 35);
        }
    }

    // build a simple general graph
    set<pair<int,int>> es;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        es.insert({a, b});
    };

    // Erdos-Renyi backbone
    for (int a = 1; a <= n; a++)
        for (int b = a + 1; b <= n; b++)
            if (rnd.next(0.0, 1.0) < p) addEdge(a, b);

    // extra conflicts incident to hubs: make the tempting sites expensive to take
    for (int a = 1; a <= n; a++) {
        if (!isHub[a]) continue;
        int extra = rnd.next(2, max(2, n / 6));
        for (int e = 0; e < extra; e++) {
            int b = rnd.next(1, n);
            addEdge(a, b);
        }
    }

    // guarantee at least one edge (needed for a meaningful conflict instance)
    if (es.empty()) addEdge(1, min(2, n));

    // relabel vertices with a random permutation so index != structural role
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) {
        int j = rnd.next(1, i);
        swap(perm[i], perm[j]);
    }

    vector<int> w2(n + 1);
    for (int i = 1; i <= n; i++) w2[perm[i]] = w[i];

    vector<pair<int,int>> edges;
    for (auto& pr : es) {
        int a = perm[pr.first], b = perm[pr.second];
        if (a > b) swap(a, b);
        edges.push_back({a, b});
    }
    // shuffle edge output order
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++) {
        printf("%d%c", w2[i], i == n ? '\n' : ' ');
    }
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
