// TIER: trivial
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
    // largest-area type first
    vector<int> order(P);
    for (int t = 0; t < P; t++) order[t] = t;
    sort(order.begin(), order.end(), [&](int a, int b){ return base[a].size() > base[b].size(); });
    for (int t : order) {
        for (int o = 0; o < 8; o++) {
            auto sh = orient(base[t], o);
            int Mr = 0, Mc = 0;
            for (auto &p : sh) { Mr = max(Mr, p.first); Mc = max(Mc, p.second); }
            for (int r = 0; r + Mr < H; r++)
                for (int c = 0; c + Mc < W; c++) {
                    bool ok = true;
                    for (auto &p : sh) if (grid[r+p.first][c+p.second] == '#') { ok = false; break; }
                    if (ok) { printf("1\n%d %d %d %d\n", t + 1, o, r, c); return 0; }
                }
        }
    }
    printf("0\n");
    return 0;
}
