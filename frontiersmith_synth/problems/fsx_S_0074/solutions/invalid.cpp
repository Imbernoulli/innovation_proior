// TIER: invalid
// Deliberately INFEASIBLE: install no relays, leaving every habitable sector cold -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    for (int i = 0; i < H; i++) { char buf[64]; scanf("%s", buf); }
    int n; scanf("%d", &n);
    // read and ignore the rest; then emit an empty relay set (infeasible when n>=1)
    printf("0\n\n");
    return 0;
}
