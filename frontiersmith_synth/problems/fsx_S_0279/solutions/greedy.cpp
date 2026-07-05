// TIER: greedy
// First-fit-by-area: scan empty cells row-major, and at each one drop the largest array
// (over types, by area) whose renormalized footprint anchored at that cell fits and is
// clear. Ignores hazard weights entirely -> fills area but not necessarily the hot cells.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, T;
vector<vector<int>> wgt;
vector<vector<pair<int,int>>> shape;
vector<int> avail, area;

vector<pair<int,int>> orient(vector<pair<int,int>> cs, int o) {
    if (o >= 4) { for (auto& p : cs) p.second = -p.second; o -= 4; }
    for (int i = 0; i < o; i++)
        for (auto& p : cs) { int r = p.first, c = p.second; p.first = c; p.second = -r; }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cs) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cs) { p.first -= mr; p.second -= mc; }
    return cs;
}

int main() {
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    wgt.assign(H, vector<int>(W, 0));
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) scanf("%d", &wgt[r][c]);
    shape.assign(T + 1, {}); avail.assign(T + 1, 0); area.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        int k, c; scanf("%d %d", &k, &c);
        avail[t] = c; area[t] = k;
        for (int i = 0; i < k; i++) { int rr, cc; scanf("%d %d", &rr, &cc); shape[t].push_back({rr, cc}); }
    }
    // precompute the 8 oriented footprints per type
    vector<vector<vector<pair<int,int>>>> ors(T + 1);
    for (int t = 1; t <= T; t++)
        for (int o = 0; o < 8; o++) ors[t].push_back(orient(shape[t], o));

    // types sorted by area descending
    vector<int> order;
    for (int t = 1; t <= T; t++) order.push_back(t);
    sort(order.begin(), order.end(), [&](int a, int b){ return area[a] > area[b]; });

    vector<vector<char>> occ(H, vector<char>(W, 0));
    vector<int> used(T + 1, 0);
    vector<array<int,4>> out;

    auto fits = [&](vector<pair<int,int>>& cs, int r0, int c0) -> bool {
        for (auto& p : cs) {
            int R = r0 + p.first, C = c0 + p.second;
            if (R < 0 || R >= H || C < 0 || C >= W) return false;
            if (occ[R][C]) return false;
        }
        return true;
    };

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            if (occ[r][c]) continue;
            bool done = false;
            for (int t : order) {
                if (used[t] >= avail[t]) continue;
                for (int o = 0; o < 8 && !done; o++) {
                    auto& cs = ors[t][o];
                    if (fits(cs, r, c)) {
                        for (auto& p : cs) occ[r + p.first][c + p.second] = 1;
                        used[t]++; out.push_back({t, o, r, c});
                        done = true;
                    }
                }
                if (done) break;
            }
        }

    printf("%d\n", (int)out.size());
    for (auto& p : out) printf("%d %d %d %d\n", p[0], p[1], p[2], p[3]);
    return 0;
}
