// TIER: invalid
// Deliberately INFEASIBLE: cut both fiber links incident to the corner hub s, isolating the
// hub from the whole array so every science antenna becomes unreachable. Must score 0 (the
// checker's connectivity gate rejects it).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, s, q, k;
    scanf("%d %d %d %d %d", &n, &m, &s, &q, &k);
    vector<int> targets(q);
    for (int i = 0; i < q; i++) scanf("%d", &targets[i]);
    vector<int> incident;
    for (int i = 1; i <= m; i++) {
        int u, v; ll w;
        scanf("%d %d %lld", &u, &v, &w);
        if (u == s || v == s) incident.push_back(i);
    }
    // Cut every link touching the hub (bounded by k), guaranteeing the hub is isolated.
    int r = min((int)incident.size(), k);
    printf("%d\n", r);
    for (int i = 0; i < r; i++) printf("%d\n", incident[i]);
    return 0;
}
