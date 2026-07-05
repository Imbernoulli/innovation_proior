// TIER: greedy
// Value-greedy: process targets in decreasing order of their best available value,
// and place each on the fitting telescope that gives the highest value.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T;
    scanf("%d %d", &N, &T);
    vector<long long> C(T);
    for (int j = 0; j < T; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> cost(N, vector<long long>(T)), val(N, vector<long long>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) scanf("%lld %lld", &cost[i][j], &val[i][j]);

    // order targets by max value across telescopes, descending (ties by index)
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    vector<long long> bestv(N, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) bestv[i] = max(bestv[i], val[i][j]);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (bestv[a] != bestv[b]) return bestv[a] > bestv[b];
        return a < b;
    });

    vector<long long> rem = C;
    vector<int> assign(N, 0);
    for (int i : order) {
        long long best = -1;
        int bj = 0;
        for (int j = 0; j < T; j++) {
            if (cost[i][j] <= rem[j] && val[i][j] > best) {
                best = val[i][j];
                bj = j + 1;
            }
        }
        if (bj > 0) {
            assign[i] = bj;
            rem[bj - 1] -= cost[i][bj - 1];
        }
    }
    for (int i = 0; i < N; i++) printf("%d\n", assign[i]);
    return 0;
}
