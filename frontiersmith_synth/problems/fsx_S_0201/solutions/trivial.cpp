// TIER: trivial
// All stations on one channel -> every overlap is co-channel -> F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, C;
    if (scanf("%d %d %d", &N, &M, &C) != 3) return 0;
    for (int i = 0; i < M; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 0; i < N; i++) printf("1\n");
    return 0;
}
