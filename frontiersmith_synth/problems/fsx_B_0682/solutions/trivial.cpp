// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, K;
    long long LAMBDA;
    scanf("%d %d %d", &H, &W, &K);
    scanf("%lld", &LAMBDA);
    vector<string> grid(H);
    for (int r = 0; r < H; r++) {
        char buf[100005];
        scanf("%s", buf);
        grid[r] = buf;
    }
    int fr = -1, fc = -1;
    for (int r = 0; r < H && fr < 0; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] != '#') { fr = r; fc = c; break; }
    printf("1\n%d %d 0\n", fr, fc);
    return 0;
}
