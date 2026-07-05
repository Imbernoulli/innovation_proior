// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

int H, W, P;
vector<string> grid;
vector<vector<pair<int,int>>> base;
vector<int> cnt;

vector<pair<int,int>> orient(const vector<pair<int,int>>& b, int o) {
    vector<pair<int,int>> r;
    for (auto &p : b) {
        int a = p.first, c = p.second, na = 0, nc = 0;
        switch (o) {
            case 0: na =  a; nc =  c; break;
            case 1: na =  c; nc = -a; break;
            case 2: na = -a; nc = -c; break;
            case 3: na = -c; nc =  a; break;
            case 4: na =  a; nc = -c; break;
            case 5: na = -a; nc =  c; break;
            case 6: na =  c; nc =  a; break;
            case 7: na = -c; nc = -a; break;
        }
        r.push_back({na, nc});
    }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto &p : r) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto &p : r) { p.first -= mr; p.second -= mc; }
    return r;
}

int main() {
    scanf("%d %d %d", &H, &W, &P);
    grid.resize(H);
    for (int i = 0; i < H; i++) { char buf[64]; scanf("%s", buf); grid[i] = buf; }
    base.resize(P); cnt.resize(P);
    for (int t = 0; t < P; t++) {
        int s; scanf("%d %d", &s, &cnt[t]);
        for (int k = 0; k < s; k++) { int a,b; scanf("%d %d", &a, &b); base[t].push_back({a,b}); }
    }
    vector<vector<char>> cov(H, vector<char>(W, 0));
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) if (grid[i][j] == '#') cov[i][j] = 1;

    // precompute normalized orientations
    vector<vector<vector<pair<int,int>>>> oris(P);
    for (int t = 0; t < P; t++) for (int o = 0; o < 8; o++) oris[t].push_back(orient(base[t], o));

    vector<int> order(P);
    for (int t = 0; t < P; t++) order[t] = t;
    sort(order.begin(), order.end(), [&](int a, int b){ return base[a].size() > base[b].size(); });

    vector<array<int,4>> placements;
    for (int t : order) {
        int copies = cnt[t];
        while (copies > 0) {
            bool placed = false;
            for (int r = 0; r < H && !placed; r++)
                for (int c = 0; c < W && !placed; c++)
                    for (int o = 0; o < 8 && !placed; o++) {
                        auto &sh = oris[t][o];
                        bool ok = true;
                        for (auto &p : sh) {
                            int rr = r + p.first, cc = c + p.second;
                            if (rr >= H || cc >= W || cov[rr][cc]) { ok = false; break; }
                        }
                        if (ok) {
                            for (auto &p : sh) cov[r + p.first][c + p.second] = 1;
                            placements.push_back({t + 1, o, r, c});
                            placed = true;
                        }
                    }
            if (!placed) break;
            copies--;
        }
    }
    printf("%d\n", (int)placements.size());
    for (auto &pl : placements) printf("%d %d %d %d\n", pl[0], pl[1], pl[2], pl[3]);
    return 0;
}
