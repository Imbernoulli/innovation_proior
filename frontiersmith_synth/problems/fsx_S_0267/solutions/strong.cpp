// TIER: strong
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

    vector<vector<vector<pair<int,int>>>> oris(P);
    for (int t = 0; t < P; t++) for (int o = 0; o < 8; o++) oris[t].push_back(orient(base[t], o));

    // types sorted by area descending (largest-first preference)
    vector<int> areaOrder(P);
    for (int t = 0; t < P; t++) areaOrder[t] = t;
    sort(areaOrder.begin(), areaOrder.end(), [&](int a, int b){ return base[a].size() > base[b].size(); });

    vector<pair<int,int>> anchors;
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) anchors.push_back({r, c});

    mt19937 rng(12345u);
    vector<array<int,4>> bestPlace;
    long long bestCov = -1;
    int restarts = 120;

    for (int it = 0; it < restarts; it++) {
        vector<vector<char>> cov(H, vector<char>(W, 0));
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) if (grid[i][j] == '#') cov[i][j] = 1;
        vector<int> rem(cnt.begin(), cnt.end());

        vector<pair<int,int>> order = anchors;
        if (it > 0) shuffle(order.begin(), order.end(), rng);

        vector<array<int,4>> place;
        long long covered = 0;
        // one maximal sweep: at each anchor, place the largest feasible section (best orientation)
        for (auto &an : order) {
            int r = an.first, c = an.second;
            int bestT = -1, bestO = -1; size_t bestArea = 0;
            // randomize type tie order slightly on restarts, else strict largest-first
            for (int ti = 0; ti < P; ti++) {
                int t = areaOrder[ti];
                if (rem[t] <= 0) continue;
                if (base[t].size() <= bestArea) continue;
                for (int o = 0; o < 8; o++) {
                    auto &sh = oris[t][o];
                    bool ok = true;
                    for (auto &p : sh) {
                        int rr = r + p.first, cc = c + p.second;
                        if (rr >= H || cc >= W || cov[rr][cc]) { ok = false; break; }
                    }
                    if (ok) { bestT = t; bestO = o; bestArea = base[t].size(); break; }
                }
            }
            if (bestT >= 0) {
                for (auto &p : oris[bestT][bestO]) cov[r + p.first][c + p.second] = 1;
                place.push_back({bestT + 1, bestO, r, c});
                rem[bestT]--;
                covered += (long long)base[bestT].size();
            }
        }
        // second sweep to fill leftover gaps with any feasible section
        for (auto &an : order) {
            int r = an.first, c = an.second;
            for (int ti = 0; ti < P; ti++) {
                int t = areaOrder[ti];
                if (rem[t] <= 0) continue;
                bool done = false;
                for (int o = 0; o < 8 && !done; o++) {
                    auto &sh = oris[t][o];
                    bool ok = true;
                    for (auto &p : sh) {
                        int rr = r + p.first, cc = c + p.second;
                        if (rr >= H || cc >= W || cov[rr][cc]) { ok = false; break; }
                    }
                    if (ok) {
                        for (auto &p : sh) cov[r + p.first][c + p.second] = 1;
                        place.push_back({t + 1, o, r, c});
                        rem[t]--; covered += (long long)base[t].size();
                        done = true;
                    }
                }
            }
        }
        if (covered > bestCov) { bestCov = covered; bestPlace = place; }
    }

    printf("%d\n", (int)bestPlace.size());
    for (auto &pl : bestPlace) printf("%d %d %d %d\n", pl[0], pl[1], pl[2], pl[3]);
    return 0;
}
