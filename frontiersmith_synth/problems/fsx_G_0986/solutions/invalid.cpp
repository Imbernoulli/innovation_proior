// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: claim every candidate edge at its full cap,
// ignoring both the coupling-degree cap and the total budget. Every node in
// this family has several incident candidate edges, so its coupling-degree
// sum blows past 1.0 immediately -- the checker must reject this with score 0.

int main() {
    int N, M; scanf("%d %d", &N, &M);
    double R, C; int T, W;
    scanf("%lf %lf %d %d", &R, &C, &T, &W);
    for (int i = 0; i < N; i++) { double x; scanf("%lf", &x); }
    vector<double> cap(M);
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d %lf", &u, &v, &cap[e]); }
    for (int e = 0; e < M; e++) printf("%.9f%c", cap[e], e + 1 == M ? '\n' : ' ');
    return 0;
}
