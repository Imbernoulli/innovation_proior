// TIER: invalid
// Deliberately infeasible: territory id 1 is assigned to the two opposite
// corners only (never adjacent), so it is split into two disconnected
// pieces -- the checker's edge-connectivity feasibility check must reject
// this with ratio 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    long long total = (long long)R * C * 2;
    for (long long i = 0; i < total; i++) { long long tmp; scanf("%lld", &tmp); }

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            bool corner = (r == 0 && c == 0) || (r == R - 1 && c == C - 1);
            int id = corner ? 1 : 2;
            printf("%d%c", id, c + 1 == C ? '\n' : ' ');
        }
    }
    return 0;
}
