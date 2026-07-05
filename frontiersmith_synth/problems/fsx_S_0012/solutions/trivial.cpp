// TIER: trivial
// All on-demand, earliest-deadline-first. Reproduces the checker's baseline B exactly.
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
    // Sections are already in deadline order; give each k_i consecutive OD steps.
    for (int i = 1; i <= N; i++) {
        for (long long c = 0; c < k[i]; c++) {
            mode[st] = 2; sec[st] = i; st++;
        }
    }
    for (int s = 1; s <= T; s++) printf("%d %d\n", mode[s], sec[s]);
    return 0;
}
