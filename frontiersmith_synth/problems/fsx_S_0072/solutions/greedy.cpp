// TIER: greedy
// One-pass spot-chasing heuristic: for each ship (input order) pick the w_j
// cheapest-spot hours in its window; take a shared spot slot when one remains,
// else fall back to shore. Ignores gantry re-mobilisation -> fragments runs.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, J, G;
    if (scanf("%d %d %d", &T, &J, &G) != 3) return 0;
    vector<int> sp(T), od(T), C(T);
    for (int t = 0; t < T; t++) scanf("%d %d %d", &sp[t], &od[t], &C[t]);
    vector<int> rem = C; // remaining shared spot capacity per hour
    for (int j = 0; j < J; j++) {
        int a, b, w; scanf("%d %d %d", &a, &b, &w);
        vector<int> hrs;
        for (int t = a; t < b; t++) hrs.push_back(t);
        sort(hrs.begin(), hrs.end(), [&](int x, int y){
            if (sp[x] != sp[y]) return sp[x] < sp[y];
            return x < y;
        });
        // choose w cheapest-spot hours
        vector<int> chosen(hrs.begin(), hrs.begin() + w);
        sort(chosen.begin(), chosen.end());
        for (int t : chosen) {
            int m;
            if (rem[t] > 0) { rem[t]--; m = 0; }
            else            { m = 1; }
            printf("%d %d ", t, m);
        }
        printf("\n");
    }
    return 0;
}
