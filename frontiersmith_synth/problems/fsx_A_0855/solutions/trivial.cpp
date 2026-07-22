// TIER: trivial
// Punch the jobs in their raw listing (row-major) order -- exactly the
// checker's own internal baseline construction B. No reasoning about tools
// or geometry at all.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, N, T;
    scanf("%d %d %d %d", &R, &C, &N, &T);
    for (int r = 0; r < R; r++) { char buf[64]; scanf("%s", buf); }
    for (int i = 0; i < N; i++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); }
    for (int i = 1; i <= N; i++) printf("%d\n", i);
    return 0;
}
