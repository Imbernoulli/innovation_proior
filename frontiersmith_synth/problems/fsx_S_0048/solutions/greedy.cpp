// TIER: greedy
// Price-threshold bang-bang: on cheap (sub-median) steps pre-cool aggressively toward the
// floor (store coolth) using spare plant capacity; on expensive steps deliver only the
// minimum cooling required to avoid overheating (coast). Always feasible by construction.
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

    // median price as the cheap/expensive threshold
    vector<int> sp = p;
    sort(sp.begin(), sp.end());
    int med = sp[T / 2];

    vector<vector<int>> x(Z, vector<int>(T, 0));
    vector<int> theta(init.begin(), init.end());

    for (int t = 0; t < T; t++) {
        bool cheap = (p[t] <= med);
        // minimum cooling to keep theta <= Cap ; maximum cooling to keep theta >= 0
        vector<int> xmin(Z), xmax(Z);
        int usedMin = 0;
        for (int j = 0; j < Z; j++) {
            int lo = theta[j] + h[j][t] - Cap[j];
            xmin[j] = lo > 0 ? lo : 0;
            xmax[j] = theta[j] + h[j][t];             // drive theta to 0
            x[j][t] = xmin[j];
            usedMin += xmin[j];
        }
        int rem = C - usedMin;                          // >= 0 (guaranteed feasible)
        if (rem < 0) rem = 0;
        if (cheap) {
            // spend spare capacity pre-cooling zones (index order)
            for (int j = 0; j < Z && rem > 0; j++) {
                int add = min(rem, xmax[j] - x[j][t]);
                if (add < 0) add = 0;
                x[j][t] += add;
                rem -= add;
            }
        }
        // advance temperatures
        for (int j = 0; j < Z; j++)
            theta[j] = theta[j] + h[j][t] - x[j][t];
    }

    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++)
            printf("%d%c", x[j][t], t == T - 1 ? '\n' : ' ');
    return 0;
}
