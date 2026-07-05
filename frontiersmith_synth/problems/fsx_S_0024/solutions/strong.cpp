// TIER: strong
// Value-aware allocation:
//   1) for each tower use spot only on its cheapest available steps whose price < lambda_i,
//      up to R_i units (covering a unit by spot only when it beats paying the penalty);
//   2) for towers still short, if lambda_i > D allocate the scarce shared ranger crew,
//      prioritising the highest-liability towers, respecting the per-step cap K;
//   3) otherwise accept the shortfall penalty.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, T, K, D;
    scanf("%d %d %d %d", &N, &T, &K, &D);
    vector<int> spot(T);
    for (int j = 0; j < T; j++) scanf("%d", &spot[j]);
    vector<int> R(N), lam(N);
    for (int i = 0; i < N; i++) scanf("%d", &R[i]);
    for (int i = 0; i < N; i++) scanf("%d", &lam[i]);
    vector<vector<char>> avail(N, vector<char>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) { int a; scanf("%d", &a); avail[i][j] = (char)a; }

    vector<vector<int>> m(N, vector<int>(T, 0));
    vector<int> work(N, 0);
    vector<int> crew(T, 0);

    // Step 1: worthwhile spot units (cheapest first) up to R_i.
    for (int i = 0; i < N; i++) {
        vector<int> cand;
        for (int j = 0; j < T; j++)
            if (avail[i][j] && spot[j] < lam[i]) cand.push_back(j);
        sort(cand.begin(), cand.end(),
             [&](int a, int b) { return spot[a] < spot[b]; });
        for (int idx = 0; idx < (int)cand.size() && work[i] < R[i]; idx++) {
            m[i][cand[idx]] = 1;
            work[i]++;
        }
    }

    // Step 2: allocate rangers to still-short towers, highest lambda first.
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int a, int b) { return lam[a] > lam[b]; });
    for (int oi = 0; oi < N; oi++) {
        int i = order[oi];
        if (lam[i] <= D) continue;                 // penalty cheaper than a ranger
        for (int j = 0; j < T && work[i] < R[i]; j++) {
            if (m[i][j] == 0 && crew[j] < K) {
                m[i][j] = 2;
                crew[j]++;
                work[i]++;
            }
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) printf("%d%c", m[i][j], j + 1 == T ? '\n' : ' ');
    return 0;
}
