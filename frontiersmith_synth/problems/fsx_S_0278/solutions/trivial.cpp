// TIER: trivial
// Switch EVERY sensor on -> feasible net, cost == checker baseline B -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    for (int i = 0; i < N; i++) { long long a, b; scanf("%lld %lld", &a, &b); }
    for (int j = 0; j < M; j++) { long long a, b, c, d; scanf("%lld %lld %lld %lld", &a, &b, &c, &d); }
    printf("%d\n", M);
    for (int j = 1; j <= M; j++) printf("%d%c", j, j == M ? '\n' : ' ');
    return 0;
}
