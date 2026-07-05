// TIER: greedy
// Value-aware one pass: process zones in input order, route each zone through the
// currently-feasible gate with the highest service value (ties -> smallest volume).
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
        int best = -1;
        long long bestVal = -1, bestVol = 0;
        for (int j = 0; j < P; j++) {
            if (remCnt[j] >= 1 && vol[i][j] <= remVol[j]) {
                if (val[i][j] > bestVal || (val[i][j] == bestVal && vol[i][j] < bestVol)) {
                    bestVal = val[i][j];
                    bestVol = vol[i][j];
                    best = j;
                }
            }
        }
        if (best >= 0) {
            remVol[best] -= vol[i][best];
            remCnt[best] -= 1;
            a[i] = best + 1;
        }
    }
    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
