#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

typedef pair<int,int> P;

// generate the (up to 8) normalized orientations of a base polyomino
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

int main(int argc, char* argv[]) {
  registerTestlibCmd(argc, argv);

  int R = inf.readInt();
  int C = inf.readInt();
  int T = inf.readInt();
  vector<vector<char>> block(R, vector<char>(C, 0));
  for (int i = 0; i < R; i++) {
    string row = inf.readToken();
    for (int j = 0; j < C; j++) block[i][j] = (row[j] == '#') ? 1 : 0;
  }
  vector<vector<P>> base0(T);      // normalized identity orientation
  vector<int> sz(T), supply(T);
  vector<vector<vector<P>>> orients(T);
  for (int s = 0; s < T; s++) {
    int m = inf.readInt();
    supply[s] = inf.readInt();
    vector<P> cells(m);
    for (int k = 0; k < m; k++) { int r = inf.readInt(); int c = inf.readInt(); cells[k] = {r, c}; }
    // normalize base
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cells) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cells) { p.first -= mr; p.second -= mc; }
    sort(cells.begin(), cells.end());
    base0[s] = cells;
    sz[s] = m;
    orients[s] = gen_orients(cells);
  }

  // ---------- internal baseline B: the FIRST footprint (input order) ONLY,
  //            orientation 0 (no rotation/reflection), top-left first-fit.
  //            Deliberately weak: one irregular shape, one orientation. --------
  vector<vector<char>> ub(R, vector<char>(C, 0));
  long long B = 0;
  {
    int s = 0;
    int sup = supply[s];
    auto& cells = base0[s];
    while (sup > 0) {
      bool placed = false;
      for (int i = 0; i < R && !placed; i++)
        for (int j = 0; j < C && !placed; j++) {
          bool ok = true;
          for (auto& p : cells) {
            int ni = i + p.first, nj = j + p.second;
            if (ni < 0 || ni >= R || nj < 0 || nj >= C || block[ni][nj] || ub[ni][nj]) { ok = false; break; }
          }
          if (ok) {
            for (auto& p : cells) ub[i + p.first][j + p.second] = 1;
            B += (int)cells.size();
            sup--; placed = true;
          }
        }
      if (!placed) break;
    }
  }
  if (B < 1) B = 1;

  // ---------- read + validate participant output ----------
  vector<vector<char>> used(R, vector<char>(C, 0));
  vector<int> left = supply;
  long long F = 0;
  int K = ouf.readInt(0, R * C, "K");
  for (int idx = 0; idx < K; idx++) {
    int s = ouf.readInt(1, T, "shape_id") - 1;
    if (left[s] <= 0) quitf(_wa, "placement %d: shape %d exceeds supply", idx, s + 1);
    int m = sz[s];
    vector<P> cells(m);
    for (int k = 0; k < m; k++) {
      int i = ouf.readInt(0, R - 1, "cell_row");
      int j = ouf.readInt(0, C - 1, "cell_col");
      cells[k] = {i, j};
    }
    // feasibility of cells: free + not already used
    for (auto& p : cells) {
      if (block[p.first][p.second]) quitf(_wa, "placement %d covers an obstruction at (%d,%d)", idx, p.first, p.second);
      if (used[p.first][p.second]) quitf(_wa, "placement %d overlaps a used cell at (%d,%d)", idx, p.first, p.second);
    }
    // shape-match: normalized cells must equal one of shape s's orientations
    vector<P> nc = cells;
    int mr = INT_MAX, mcc = INT_MAX;
    for (auto& p : nc) { mr = min(mr, p.first); mcc = min(mcc, p.second); }
    for (auto& p : nc) { p.first -= mr; p.second -= mcc; }
    sort(nc.begin(), nc.end());
    bool match = false;
    for (auto& o : orients[s]) if (o == nc) { match = true; break; }
    if (!match) quitf(_wa, "placement %d is not a valid orientation of shape %d", idx, s + 1);
    // commit
    for (auto& p : cells) used[p.first][p.second] = 1;
    left[s]--;
    F += m;
  }
  if (!ouf.seekEof()) quitf(_wa, "trailing output");

  double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
  quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
  return 0;
}
