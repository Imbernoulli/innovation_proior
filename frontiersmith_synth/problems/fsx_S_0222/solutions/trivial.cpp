// TIER: trivial
// First-fit in input / gate-index order -- exactly reproduces the checker baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, P;
    if (scanf("%d %d", &N, &P) != 2) return 0;
    vector<long long> C(P), K(P);
    for (int j = 0; j < P; j++) scanf("%lld %lld", &C[j], &K[j]);
    vector<vector<long long>> val(N, vector<long long>(P)), vol(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            scanf("%lld %lld", &val[i][j], &vol[i][j]);

    vector<long long> remVol = C, remCnt = K;
    vector<int> a(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            if (remCnt[j] >= 1 && vol[i][j] <= remVol[j]) {
                remVol[j] -= vol[i][j];
                remCnt[j] -= 1;
                a[i] = j + 1;
                break;
            }
        }
    }
    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
