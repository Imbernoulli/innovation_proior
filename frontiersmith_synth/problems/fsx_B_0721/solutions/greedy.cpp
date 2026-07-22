// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The "obvious" competent approach: a dense running-bond (brick) tiling.
// Every row is filled with horizontal dominoes at a FIXED column phase that
// alternates by row parity (even rows start at column 0, odd rows at column
// 1) -- exactly the real-world trick for avoiding four-mat corners with a
// single pass, and it covers almost the whole court. But it never uses a
// vertical mat (orientation entropy stays at 0) and it drops the single
// required 1x1 mat wherever the scan happens to first run out of pairs, with
// no regard for the court's centre or its mirror symmetry.

int main() {
    int H, W;
    scanf("%d %d", &H, &W);
    vector<string> g(H);
    for (int r = 0; r < H; r++) { char buf[512]; scanf("%s", buf); g[r] = buf; }

    vector<vector<int>> matId(H, vector<int>(W, -1));
    vector<tuple<char,int,int>> out;
    vector<pair<int,int>> orphans;
    int mid = 0;
    for (int r = 0; r < H; r++) {
        int start = (r % 2 == 0) ? 0 : 1;
        for (int c = start; c + 1 < W; c += 2) {
            if (g[r][c] == '.' && g[r][c + 1] == '.') {
                matId[r][c] = mid; matId[r][c + 1] = mid; mid++;
                out.push_back({'H', r, c});
            } else {
                if (g[r][c] == '.' && matId[r][c] == -1) orphans.push_back({r, c});
                if (g[r][c + 1] == '.' && matId[r][c + 1] == -1) orphans.push_back({r, c + 1});
            }
        }
        // columns skipped by the fixed phase at the row ends
        if (start == 1 && g[r][0] == '.') orphans.push_back({r, 0});
        if ((W - start) % 2 == 1 && g[r][W - 1] == '.' && matId[r][W - 1] == -1)
            orphans.push_back({r, W - 1});
    }

    // exactly one required S mat: use the first true orphan (an uncovered free
    // cell), arbitrary order -- no attempt to pick the room's centre.
    bool placed = false;
    for (auto &p : orphans) {
        if (matId[p.first][p.second] == -1) { out.push_back({'S', p.first, p.second}); placed = true; break; }
    }
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
