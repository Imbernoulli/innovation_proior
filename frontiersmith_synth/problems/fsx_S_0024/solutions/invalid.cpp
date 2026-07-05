// TIER: invalid
// Deliberately infeasible: put every tower in ranger mode at every step, which
// exceeds the shared per-step crew cap K whenever N > K. Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, T, K, D;
    scanf("%d %d %d %d", &N, &T, &K, &D);
    for (int j = 0; j < T; j++) { int x; scanf("%d", &x); }
    for (int i = 0; i < N; i++) { int x; scanf("%d", &x); }
    for (int i = 0; i < N; i++) { int x; scanf("%d", &x); }
    for (int i = 0; i < N; i++) for (int j = 0; j < T; j++) { int x; scanf("%d", &x); }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) printf("%d%c", 2, j + 1 == T ? '\n' : ' ');
    return 0;
}
