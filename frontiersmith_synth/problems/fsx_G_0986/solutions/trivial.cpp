// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Do-nothing baseline: assign zero coupling to every candidate edge. Every
// oscillator then evolves completely independently. This is exactly the
// checker's internal reference construction B, so this solution always
// scores ratio ~= 0.1 by definition.

int main() {
    int N, M; scanf("%d %d", &N, &M);
    double R, C; int T, W;
    scanf("%lf %lf %d %d", &R, &C, &T, &W);
    for (int i = 0; i < N; i++) { double x; scanf("%lf", &x); }
    for (int e = 0; e < M; e++) { int u, v; double cap; scanf("%d %d %lf", &u, &v, &cap); }
    for (int e = 0; e < M; e++) printf("0%c", e + 1 == M ? '\n' : ' ');
    return 0;
}
