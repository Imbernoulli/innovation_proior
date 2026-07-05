#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---- polyomino pool (normalized, min row/col = 0) ----
// Chosen to be mostly IRREGULAR (poor packers without rotation) so that a
// no-rotation baseline is genuinely weak vs rotation-aware heuristics.
static vector<vector<pair<int,int>>> POOL = {
  {{0,1},{0,2},{1,0},{1,1},{2,1}},          // 0 F-pentomino-ish
  {{0,0},{1,0},{2,0},{3,0},{3,1}},          // 1 L-pentomino
  {{0,0},{0,1},{1,0},{1,1},{2,0}},          // 2 P-pentomino
  {{0,0},{0,2},{1,0},{1,1},{1,2}},          // 3 U-pentomino
  {{0,1},{1,0},{1,1},{1,2},{2,1}},          // 4 plus-pentomino
  {{0,0},{1,0},{2,0},{2,1}},                // 5 L-tetromino
  {{0,0},{0,1},{0,2},{1,1}},                // 6 T-tetromino
  {{0,1},{0,2},{1,0},{1,1}},                // 7 S-tetromino
};

int main(int argc, char* argv[]) {
  registerGen(argc, argv, 1);
  int t = atoi(argv[1]);

  // board size grows with testId
  int R = 6 + (t - 1) * 5;   // 6 .. 51
  int C = 6 + (t - 1) * 5;
  R = min(R, 52); C = min(C, 52);

  // obstruction density (structural pillars / installed equipment)
  double dens = (t <= 1) ? 0.08 : 0.12 + 0.008 * t;   // ~0.12..0.20
  vector<string> grid(R, string(C, '.'));
  for (int i = 0; i < R; i++)
    for (int j = 0; j < C; j++)
      if (rnd.next(0.0, 1.0) < dens) grid[i][j] = '#';

  int freec = 0;
  for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) if (grid[i][j] == '.') freec++;
  // guarantee some free space exists
  if (freec < 10) { for (int i=0;i<R;i++) for(int j=0;j<C;j++) grid[i][j]='.'; freec=R*C; }

  // Build catalog: 3 pentominoes + 3 tetrominoes, all IRREGULAR (no small
  // filler) so that packing is genuinely hard and rotation-aware heuristics
  // diverge from a no-rotation baseline. First footprint = a pentomino.
  vector<int> pent = {0,1,2,3,4};
  shuffle(pent.begin(), pent.end());
  vector<int> tet = {5,6,7};
  shuffle(tet.begin(), tet.end());

  vector<vector<pair<int,int>>> cat;
  cat.push_back(POOL[pent[0]]);
  cat.push_back(POOL[pent[1]]);
  cat.push_back(POOL[pent[2]]);
  cat.push_back(POOL[tet[0]]);
  cat.push_back(POOL[tet[1]]);
  cat.push_back(POOL[tet[2]]);

  int T = (int)cat.size();

  // supply: over-supply (~1.3x free area) so the board is packing-constrained.
  long long sumArea = 0;
  for (auto& s : cat) sumArea += (long long)s.size();
  double budget = 1.3 * (double)freec;
  vector<int> supply(T);
  for (int s = 0; s < T; s++) {
    int a = (int)cat[s].size();
    int sup = (int)llround(budget / ((double)T * a));
    sup = max(2, min(sup, 45));
    supply[s] = sup;
  }

  // ---- print ----
  printf("%d %d %d\n", R, C, T);
  for (int i = 0; i < R; i++) printf("%s\n", grid[i].c_str());
  for (int s = 0; s < T; s++) {
    printf("%d %d\n", (int)cat[s].size(), supply[s]);
    for (auto& p : cat[s]) printf("%d %d\n", p.first, p.second);
  }
  return 0;
}
