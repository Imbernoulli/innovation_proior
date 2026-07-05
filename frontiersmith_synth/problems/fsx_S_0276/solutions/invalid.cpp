// TIER: invalid
// Deliberately infeasible for every input: it prints T lines but with mode = 2, which is out
// of the allowed range [0,1].  The checker's bounded read of mode rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N;
    if (scanf("%d %d", &T, &N) != 2) return 0;
    for (int t = 0; t < T; t++) { int x, y, z; scanf("%d %d %d", &x, &y, &z); }
    for (int j = 0; j < N; j++) { int a, b, c; long long d; scanf("%d %d %d %lld", &a, &b, &c, &d); }
    for (int t = 0; t < T; t++) printf("0 2\n");   // mode 2 is illegal
    return 0;
}
