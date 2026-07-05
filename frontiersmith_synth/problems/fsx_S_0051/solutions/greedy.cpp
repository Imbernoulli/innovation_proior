// TIER: greedy
// Largest-area shapes first, all 8 orientations, single first-fit pass.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> P;

static vector<vector<P>> gen_orients(const vector<P>& base) {
  set<vector<P>> uniq;
  for (int k = 0; k < 8; k++) {
    vector<P> v;
    for (auto& p : base) {
      int r = p.first, c = p.second, nr, nc;
      switch (k) {
        case 0: nr = r;  nc = c;  break;
        case 1: nr = c;  nc = -r; break;
        case 2: nr = -r; nc = -c; break;
        case 3: nr = -c; nc = r;  break;
        case 4: nr = r;  nc = -c; break;
        case 5: nr = -r; nc = c;  break;
        case 6: nr = c;  nc = r;  break;
        default: nr = -c; nc = -r; break;
      }
      v.push_back({nr, nc});
    }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : v) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : v) { p.first -= mr; p.second -= mc; }
    sort(v.begin(), v.end());
    uniq.insert(v);
  }
  return vector<vector<P>>(uniq.begin(), uniq.end());
}

int main() {
  int R, C, T;
  if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
  vector<vector<char>> block(R, vector<char>(C, 0));
  for (int i = 0; i < R; i++) { char buf[256]; scanf("%s", buf); for (int j = 0; j < C; j++) block[i][j] = (buf[j] == '#'); }
  vector<int> sz(T), supply(T);
  vector<vector<vector<P>>> orients(T);
  for (int s = 0; s < T; s++) {
    int m; scanf("%d %d", &m, &supply[s]);
    vector<P> cells(m);
    int mr = INT_MAX, mc = INT_MAX;
    for (int k = 0; k < m; k++) { scanf("%d %d", &cells[k].first, &cells[k].second); mr = min(mr, cells[k].first); mc = min(mc, cells[k].second); }
    for (auto& p : cells) { p.first -= mr; p.second -= mc; }
    sort(cells.begin(), cells.end());
    sz[s] = m; orients[s] = gen_orients(cells);
  }

  vector<int> order(T);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int a, int b){ return sz[a] > sz[b]; });

  vector<vector<char>> used(R, vector<char>(C, 0));
  vector<pair<int,vector<P>>> res;
  for (int oi = 0; oi < T; oi++) {
    int s = order[oi];
    for (int i = 0; i < R; i++)
      for (int j = 0; j < C; j++) {
        if (supply[s] <= 0) break;
        for (auto& o : orients[s]) {
          bool ok = true;
          for (auto& p : o) {
            int ni = i + p.first, nj = j + p.second;
            if (ni < 0 || ni >= R || nj < 0 || nj >= C || block[ni][nj] || used[ni][nj]) { ok = false; break; }
          }
          if (ok) {
            vector<P> pc;
            for (auto& p : o) { used[i + p.first][j + p.second] = 1; pc.push_back({i + p.first, j + p.second}); }
            res.push_back({s, pc});
            supply[s]--;
            break;
          }
        }
      }
  }

  printf("%d\n", (int)res.size());
  for (auto& r : res) { printf("%d", r.first + 1); for (auto& p : r.second) printf(" %d %d", p.first, p.second); printf("\n"); }
  return 0;
}
