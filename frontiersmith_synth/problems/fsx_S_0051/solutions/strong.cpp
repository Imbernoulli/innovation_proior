// TIER: strong
// Multi-restart randomized greedy (shuffled shape/orientation order) + a
// hole-filling pass with the smallest shapes; keep the best coverage found.
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

int R, C, T;
vector<vector<char>> block;
vector<int> sz, supply0;
vector<vector<vector<P>>> orients;

struct Res { long long cov; vector<pair<int,vector<P>>> pl; };

Res run(mt19937& rng, bool holefill) {
  vector<vector<char>> used(R, vector<char>(C, 0));
  vector<int> supply = supply0;
  vector<pair<int,vector<P>>> pl;
  long long cov = 0;

  // shape order: largest first, but permute within to add variety
  vector<int> order(T);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int a, int b){ return sz[a] > sz[b]; });
  for (int a = 0; a + 1 < T; a++) if (rng() % 3 == 0) swap(order[a], order[a + 1]);

  for (int oi = 0; oi < T; oi++) {
    int s = order[oi];
    // randomized orientation preference order
    vector<int> op(orients[s].size());
    iota(op.begin(), op.end(), 0);
    shuffle(op.begin(), op.end(), rng);
    int startI = R ? (int)(rng() % R) : 0;
    for (int di = 0; di < R; di++) {
      int i = (startI + di) % R;
      for (int j = 0; j < C; j++) {
        if (supply[s] <= 0) { di = R; break; }
        for (int oo : op) {
          auto& o = orients[s][oo];
          bool ok = true;
          for (auto& p : o) {
            int ni = i + p.first, nj = j + p.second;
            if (ni < 0 || ni >= R || nj < 0 || nj >= C || block[ni][nj] || used[ni][nj]) { ok = false; break; }
          }
          if (ok) {
            vector<P> pc;
            for (auto& p : o) { used[i + p.first][j + p.second] = 1; pc.push_back({i + p.first, j + p.second}); }
            pl.push_back({s, pc});
            supply[s]--; cov += sz[s];
            break;
          }
        }
      }
    }
  }

  if (holefill) {
    // greedily fill remaining holes with the smallest shapes first
    vector<int> order2(T);
    iota(order2.begin(), order2.end(), 0);
    sort(order2.begin(), order2.end(), [&](int a, int b){ return sz[a] < sz[b]; });
    for (int oi = 0; oi < T; oi++) {
      int s = order2[oi];
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
              pl.push_back({s, pc});
              supply[s]--; cov += sz[s];
              break;
            }
          }
        }
    }
  }
  return Res{cov, pl};
}

int main() {
  if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
  block.assign(R, vector<char>(C, 0));
  for (int i = 0; i < R; i++) { char buf[256]; scanf("%s", buf); for (int j = 0; j < C; j++) block[i][j] = (buf[j] == '#'); }
  sz.assign(T, 0); supply0.assign(T, 0); orients.assign(T, {});
  for (int s = 0; s < T; s++) {
    int m; scanf("%d %d", &m, &supply0[s]);
    vector<P> cells(m);
    int mr = INT_MAX, mc = INT_MAX;
    for (int k = 0; k < m; k++) { scanf("%d %d", &cells[k].first, &cells[k].second); mr = min(mr, cells[k].first); mc = min(mc, cells[k].second); }
    for (auto& p : cells) { p.first -= mr; p.second -= mc; }
    sort(cells.begin(), cells.end());
    sz[s] = m; orients[s] = gen_orients(cells);
  }

  mt19937 rng(987654321u);
  Res best; best.cov = -1;
  for (int r = 0; r < 14; r++) {
    Res cur = run(rng, true);
    if (cur.cov > best.cov) best = move(cur);
  }

  printf("%d\n", (int)best.pl.size());
  for (auto& r : best.pl) { printf("%d", r.first + 1); for (auto& p : r.second) printf(" %d %d", p.first, p.second); printf("\n"); }
  return 0;
}
