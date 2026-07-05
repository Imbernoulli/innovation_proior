// TIER: trivial
// First-fit in input order over telescopes in index order.
// This reproduces the checker's internal baseline exactly -> ratio ~= 0.1.
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

    vector<long long> rem = C;
    vector<int> assign(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T; j++) {
            if (cost[i][j] <= rem[j]) {
                rem[j] -= cost[i][j];
                assign[i] = j + 1;
                break;
            }
        }
    }
    for (int i = 0; i < N; i++) printf("%d\n", assign[i]);
    return 0;
}
