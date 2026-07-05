// TIER: greedy
// Profit-greedy: consider all (group,gallery) tours sorted by profit descending;
// assign a tour if the group is still free and the gallery has enough budget.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int V, G;
    if (scanf("%d %d", &V, &G) != 2) return 0;
    vector<int> C(G + 1);
    for (int j = 1; j <= G; j++) scanf("%d", &C[j]);
    vector<vector<int>> a(V + 1, vector<int>(G + 1)), p(V + 1, vector<int>(G + 1));
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++) scanf("%d %d", &a[i][j], &p[i][j]);

    struct T { int prof, i, j; };
    vector<T> tours;
    tours.reserve((size_t)V * G);
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++)
            tours.push_back({p[i][j], i, j});
    sort(tours.begin(), tours.end(), [](const T& x, const T& y) {
        return x.prof > y.prof;
    });

    vector<char> used(V + 1, 0);
    vector<int> rem(G + 1);
    for (int j = 1; j <= G; j++) rem[j] = C[j];

    vector<pair<int,int>> asn;
    for (auto& t : tours) {
        if (used[t.i]) continue;
        if (a[t.i][t.j] <= rem[t.j]) {
            used[t.i] = 1;
            rem[t.j] -= a[t.i][t.j];
            asn.push_back({t.i, t.j});
        }
    }

    printf("%d\n", (int)asn.size());
    for (auto& e : asn) printf("%d %d\n", e.first, e.second);
    return 0;
}
