// TIER: invalid
// Deliberately infeasible: prints an out-of-range rotor phase (9, outside the
// required [0,3]), which the checker's bounded read must reject -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int L, N, sx, sy;
    scanf("%d %d", &L, &N);
    scanf("%d %d", &sx, &sy);
    for (int i = 0; i < N; i++) { int x, y; scanf("%d %d", &x, &y); }
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) printf("9%c", x + 1 < L ? ' ' : '\n');
    }
    return 0;
}
