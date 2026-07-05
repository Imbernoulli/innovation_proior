// TIER: trivial
// First-fit in input / pool-index order -- exactly the checker's reference B.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, P;
    scanf("%d %d", &N, &P);
    vector<long long> C(P);
    for (int j = 0; j < P; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> v(N, vector<long long>(P)), w(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) scanf("%lld %lld", &v[i][j], &w[i][j]);

    vector<long long> rem = C;
    vector<int> a(N, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            if (w[i][j] <= rem[j]) { rem[j] -= w[i][j]; a[i] = j + 1; break; }

    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
