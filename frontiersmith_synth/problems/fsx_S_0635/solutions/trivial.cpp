// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, B, sr, sc;
    scanf("%d %d %d", &R, &C, &B);
    scanf("%d %d", &sr, &sc);
    vector<string> grid(R);
    for (int i = 0; i < R; i++) { char buf[210]; scanf("%s", buf); grid[i] = buf; }
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) { int x; scanf("%d", &x); }
    // Exactly the checker's own baseline construction: DR, zero mirrors.
    printf("DR\n0\n");
    return 0;
}
