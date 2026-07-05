// TIER: trivial
// First-fit in index order -- exactly the checker's baseline construction, so ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> v(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    vector<long long> C(M);
    for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> d(N, vector<long long>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) scanf("%lld", &d[i][j]);

    vector<long long> rem(C);
    vector<int> a(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (rem[j] >= d[i][j]) { rem[j] -= d[i][j]; a[i] = j + 1; break; }
        }
    }
    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 < N ? ' ' : '\n');
    return 0;
}
