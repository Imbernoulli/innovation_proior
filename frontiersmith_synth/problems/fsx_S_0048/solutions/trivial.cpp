// TIER: trivial
// Reactive schedule x[j][t] = h[j][t] = exactly the checker's internal baseline B -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int Z, T, C, s;
    scanf("%d %d %d %d", &Z, &T, &C, &s);
    vector<int> p(T);
    for (int t = 0; t < T; t++) scanf("%d", &p[t]);
    vector<int> Cap(Z), init(Z);
    for (int j = 0; j < Z; j++) scanf("%d", &Cap[j]);
    for (int j = 0; j < Z; j++) scanf("%d", &init[j]);
    vector<vector<int>> h(Z, vector<int>(T));
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++) scanf("%d", &h[j][t]);

    for (int j = 0; j < Z; j++) {
        for (int t = 0; t < T; t++)
            printf("%d%c", h[j][t], t == T - 1 ? '\n' : ' ');
    }
    return 0;
}
