// TIER: trivial
// Reproduces the checker's internal baseline: the FIRST footprint only,
// orientation 0 (no rotation/reflection), first-fit scan. Scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> P;

int main() {
  int R, C, T;
  if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
  vector<vector<char>> block(R, vector<char>(C, 0));
  for (int i = 0; i < R; i++) {
    char buf[256];
    scanf("%s", buf);
    for (int j = 0; j < C; j++) block[i][j] = (buf[j] == '#');
  }
  vector<vector<P>> base(T);
  vector<int> sz(T), supply(T);
  for (int s = 0; s < T; s++) {
    int m; scanf("%d %d", &m, &supply[s]);
    vector<P> cells(m);
    int mr = INT_MAX, mc = INT_MAX;
    for (int k = 0; k < m; k++) { scanf("%d %d", &cells[k].first, &cells[k].second); mr = min(mr, cells[k].first); mc = min(mc, cells[k].second); }
    for (auto& p : cells) { p.first -= mr; p.second -= mc; }
    sort(cells.begin(), cells.end());
    base[s] = cells; sz[s] = m;
  }

  vector<vector<char>> used(R, vector<char>(C, 0));
  vector<pair<int,vector<P>>> res;
  for (int s = 0; s < 1; s++) {      // FIRST footprint only, matches baseline B
    int sup = supply[s];
    auto& cells = base[s];
    while (sup > 0) {
      bool placed = false;
      for (int i = 0; i < R && !placed; i++)
        for (int j = 0; j < C && !placed; j++) {
          bool ok = true;
          for (auto& p : cells) {
            int ni = i + p.first, nj = j + p.second;
            if (ni < 0 || ni >= R || nj < 0 || nj >= C || block[ni][nj] || used[ni][nj]) { ok = false; break; }
          }
          if (ok) {
            vector<P> pc;
            for (auto& p : cells) { used[i + p.first][j + p.second] = 1; pc.push_back({i + p.first, j + p.second}); }
            res.push_back({s, pc});
            sup--; placed = true;
          }
        }
      if (!placed) break;
    }
  }

  printf("%d\n", (int)res.size());
  for (auto& r : res) {
    printf("%d", r.first + 1);
    for (auto& p : r.second) printf(" %d %d", p.first, p.second);
    printf("\n");
  }
  return 0;
}
