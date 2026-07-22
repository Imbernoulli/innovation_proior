// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Repeat the smallest-index self-loop tile (N==S && E==W) everywhere: the
// "do nothing" baseline construction the checker itself measures as B.
int main() {
    int R, W, T;
    scanf("%d %d %d", &R, &W, &T);
    vector<array<int,4>> tiles(T);
    for (int i = 0; i < T; i++)
        scanf("%d %d %d %d", &tiles[i][0], &tiles[i][1], &tiles[i][2], &tiles[i][3]);

    int loopIdx = 0;
    for (int i = 0; i < T; i++) {
        if (tiles[i][0] == tiles[i][2] && tiles[i][1] == tiles[i][3]) { loopIdx = i; break; }
    }
    int id = loopIdx + 1;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) printf(j ? " %d" : "%d", id);
        printf("\n");
    }
    return 0;
}
