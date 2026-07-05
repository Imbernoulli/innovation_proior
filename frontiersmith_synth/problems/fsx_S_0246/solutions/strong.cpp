// TIER: strong
// Density-greedy construction (profit per docent-minute) followed by first-improving
// local search: adds, single moves, and pairwise gallery swaps until no improvement.
#include <bits/stdc++.h>
using namespace std;
int V, G;
vector<vector<int>> a, p;
vector<int> C, rem, assignAt; // assignAt[i] = gallery (0 = unassigned)

int main() {
    if (scanf("%d %d", &V, &G) != 2) return 0;
    C.assign(G + 1, 0);
    for (int j = 1; j <= G; j++) scanf("%d", &C[j]);
    a.assign(V + 1, vector<int>(G + 1));
    p.assign(V + 1, vector<int>(G + 1));
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++) scanf("%d %d", &a[i][j], &p[i][j]);

    rem = C;
    assignAt.assign(V + 1, 0);

    // ---- density-greedy construction ----
    struct T { double d; int prof, i, j; };
    vector<T> tours;
    tours.reserve((size_t)V * G);
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++)
            tours.push_back({(double)p[i][j] / (double)a[i][j], p[i][j], i, j});
    sort(tours.begin(), tours.end(), [](const T& x, const T& y) {
        if (x.d != y.d) return x.d > y.d;
        return x.prof > y.prof;
    });
    for (auto& t : tours) {
        if (assignAt[t.i]) continue;
        if (a[t.i][t.j] <= rem[t.j]) {
            assignAt[t.i] = t.j;
            rem[t.j] -= a[t.i][t.j];
        }
    }

    // ---- local search ----
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 200) {
        improved = false;

        // 1) ADD: place an unassigned group in its best feasible gallery
        for (int i = 1; i <= V && !improved; i++) {
            if (assignAt[i]) continue;
            int bj = 0, bp = 0;
            for (int j = 1; j <= G; j++)
                if (a[i][j] <= rem[j] && p[i][j] > bp) { bp = p[i][j]; bj = j; }
            if (bj) { assignAt[i] = bj; rem[bj] -= a[i][bj]; improved = true; }
        }
        if (improved) continue;

        // 2) MOVE: relocate an assigned group to a more profitable gallery
        for (int i = 1; i <= V && !improved; i++) {
            int cj = assignAt[i];
            if (!cj) continue;
            for (int j = 1; j <= G; j++) {
                if (j == cj) continue;
                if (a[i][j] <= rem[j] && p[i][j] > p[i][cj]) {
                    rem[cj] += a[i][cj];
                    assignAt[i] = j;
                    rem[j] -= a[i][j];
                    improved = true;
                    break;
                }
            }
        }
        if (improved) continue;

        // 3) SWAP: exchange galleries of two assigned groups if total profit rises
        for (int x = 1; x <= V && !improved; x++) {
            int jx = assignAt[x];
            if (!jx) continue;
            for (int y = x + 1; y <= V && !improved; y++) {
                int jy = assignAt[y];
                if (!jy || jy == jx) continue;
                // feasibility after swap: x->jy, y->jx
                int remJx = rem[jx] + a[x][jx] - a[y][jx];
                int remJy = rem[jy] + a[y][jy] - a[x][jy];
                if (remJx < 0 || remJy < 0) continue;
                int delta = (p[x][jy] + p[y][jx]) - (p[x][jx] + p[y][jy]);
                if (delta > 0) {
                    rem[jx] = remJx;
                    rem[jy] = remJy;
                    assignAt[x] = jy;
                    assignAt[y] = jx;
                    improved = true;
                }
            }
        }
    }

    vector<pair<int,int>> asn;
    for (int i = 1; i <= V; i++)
        if (assignAt[i]) asn.push_back({i, assignAt[i]});
    printf("%d\n", (int)asn.size());
    for (auto& e : asn) printf("%d %d\n", e.first, e.second);
    return 0;
}
