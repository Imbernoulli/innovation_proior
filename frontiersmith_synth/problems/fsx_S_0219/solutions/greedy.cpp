// TIER: greedy
// Value-density greedy: scan tiles row-major; at each free tile place the largest-area
// piece that fits in its DEFAULT orientation (o=0) subject to remaining supply. One pass.
#include <bits/stdc++.h>
using namespace std;

int H, W, D;
vector<vector<pair<int,int>>> base;
vector<int> supply;
vector<vector<char>> occ; // blocked or used

int main() {
    scanf("%d %d %d", &H, &W, &D);
    base.resize(D); supply.assign(D, 0);
    for (int i = 0; i < D; i++) {
        int A; scanf("%d %d", &A, &supply[i]);
        for (int j = 0; j < A; j++) { int dr, dc; scanf("%d %d", &dr, &dc); base[i].push_back({dr, dc}); }
    }
    int Q; scanf("%d", &Q);
    occ.assign(H, vector<char>(W, 0));
    for (int j = 0; j < Q; j++) { int r, c; scanf("%d %d", &r, &c); occ[r][c] = 1; }

    // shapes ordered by area descending
    vector<int> order(D);
    for (int i = 0; i < D; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){ return base[a].size() > base[b].size(); });

    vector<int> rem = supply;
    vector<array<int,4>> out; // i,o,r,c

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            if (occ[r][c]) continue;
            for (int idx : order) {
                if (rem[idx] <= 0) continue;
                bool fit = true;
                for (auto& p : base[idx]) {
                    int rr = r + p.first, cc = c + p.second;
                    if (rr < 0 || rr >= H || cc < 0 || cc >= W || occ[rr][cc]) { fit = false; break; }
                }
                if (!fit) continue;
                for (auto& p : base[idx]) occ[r + p.first][c + p.second] = 1;
                rem[idx]--;
                out.push_back({idx, 0, r, c});
                break;
            }
        }
    }

    printf("%d\n", (int)out.size());
    for (auto& a : out) printf("%d %d %d %d\n", a[0], a[1], a[2], a[3]);
    return 0;
}
