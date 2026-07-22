// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Reproduces exactly the checker's own internal baseline: cover only the
// even rows with fixed-phase horizontal pairs (0,1),(2,3),...; the single
// required 1x1 mat goes on the first free cell found scanning the odd rows.
// No thought given to orientation mix, junction shape, or symmetry at all.

int main() {
    int H, W;
    scanf("%d %d", &H, &W);
    vector<string> g(H);
    for (int r = 0; r < H; r++) { char buf[512]; scanf("%s", buf); g[r] = buf; }

    vector<vector<int>> matId(H, vector<int>(W, -1));
    vector<tuple<char,int,int>> out;
    int mid = 0;
    for (int r = 0; r < H; r += 2) {
        for (int c = 0; c + 1 < W; c += 2) {
            if (g[r][c] == '.' && g[r][c + 1] == '.') {
                matId[r][c] = mid; matId[r][c + 1] = mid; mid++;
                out.push_back({'H', r, c});
            }
        }
    }
    bool placed = false;
    for (int r = 1; r < H && !placed; r += 2)
        for (int c = 0; c < W; c++)
            if (g[r][c] == '.') { out.push_back({'S', r, c}); placed = true; break; }
    if (!placed)
        for (int r = 0; r < H && !placed; r++)
            for (int c = 0; c < W; c++)
                if (g[r][c] == '.' && matId[r][c] == -1) { out.push_back({'S', r, c}); placed = true; break; }
    if (!placed) {
        int cr = (H - 1) / 2, cc = (W - 1) / 2;
        vector<tuple<char,int,int>> filtered;
        for (auto &t : out) {
            auto [ty, r, c] = t;
            if (ty == 'H' && r == cr && (c == cc || c + 1 == cc)) continue;
            filtered.push_back(t);
        }
        out = filtered;
        out.push_back({'S', cr, cc});
    }

    printf("%d\n", (int)out.size());
    for (auto &t : out) {
        auto [ty, r, c] = t;
        printf("%c %d %d\n", ty, r, c);
    }
    return 0;
}
