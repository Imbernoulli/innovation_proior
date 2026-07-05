// TIER: invalid
// Pauses every step: no section ever completes, so every deadline is missed => score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T; long long D, OD;
    scanf("%d %d %lld %lld", &N, &T, &D, &OD);
    vector<long long> cap(T + 1), sc(T + 1);
    for (int s = 1; s <= T; s++) scanf("%lld %lld", &cap[s], &sc[s]);
    long long w, dd;
    for (int i = 1; i <= N; i++) scanf("%lld %lld", &w, &dd);
    for (int s = 1; s <= T; s++) printf("0 0\n");
    return 0;
}
