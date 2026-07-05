// TIER: invalid
// Switches nothing on -> every target has coverage 0 < demand -> infeasible -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int S, G, T;
    if (scanf("%d %d %d", &S, &G, &T) != 3) return 0;
    for (int r = 0; r < T; r++) printf("0\n");
    return 0;
}
