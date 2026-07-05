// TIER: trivial
// Chain baseline: build the guaranteed backbone links 1-2, ..., (N-1)-N.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    for (int i = 1; i <= N; i++) { int c; scanf("%d", &c); }
    // ignore the edge list; the chain links are guaranteed to exist
    printf("%d\n", N - 1);
    for (int i = 1; i <= N - 1; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
