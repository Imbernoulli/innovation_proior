// TIER: invalid
// Deliberately infeasible: the first cooling value exceeds the plant capacity C (out of the
// checker's [0,C] range), so the output is rejected and scores 0.
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
    // (heat trace not needed)

    for (int j = 0; j < Z; j++) {
        for (int t = 0; t < T; t++) {
            int v = (j == 0 && t == 0) ? (C + 1) : 0;   // out-of-range token
            printf("%d%c", v, t == T - 1 ? '\n' : ' ');
        }
    }
    return 0;
}
