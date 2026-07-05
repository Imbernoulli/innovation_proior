// TIER: greedy
// Baseline OD-EDF schedule, then swap any on-demand step for spot at the same slot
// whenever spot is at least as productive (cap>=D) and strictly cheaper (sc<OD).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T; long long D, OD;
    scanf("%d %d %lld %lld", &N, &T, &D, &OD);
    vector<long long> cap(T + 1), sc(T + 1);
    for (int s = 1; s <= T; s++) scanf("%lld %lld", &cap[s], &sc[s]);
    vector<long long> W(N + 1), d(N + 1), k(N + 1);
    for (int i = 1; i <= N; i++) {
        scanf("%lld %lld", &W[i], &d[i]);
        k[i] = (W[i] + D - 1) / D;
    }

    vector<int> mode(T + 1, 0), sec(T + 1, 0);
    int st = 1;
    for (int i = 1; i <= N; i++)
        for (long long c = 0; c < k[i]; c++) { mode[st] = 2; sec[st] = i; st++; }

    for (int s = 1; s <= T; s++) {
        if (mode[s] == 2 && cap[s] >= D && sc[s] < OD) {
            mode[s] = 1;   // spot delivers >= D units, keeps section feasible, cheaper
        }
    }
    for (int s = 1; s <= T; s++) printf("%d %d\n", mode[s], sec[s]);
    return 0;
}
