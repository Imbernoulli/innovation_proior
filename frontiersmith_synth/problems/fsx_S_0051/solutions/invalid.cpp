// TIER: invalid
// Deliberately infeasible: emits shape 1's cells all at the same location,
// which cannot form a valid orientation -> checker rejects (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
  int R, C, T;
  if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
  for (int i = 0; i < R; i++) { char buf[256]; scanf("%s", buf); }
  int m1 = 0;
  for (int s = 0; s < T; s++) {
    int m, sup; scanf("%d %d", &m, &sup);
    for (int k = 0; k < m; k++) { int r, c; scanf("%d %d", &r, &c); }
    if (s == 0) m1 = m;
  }
  printf("1\n1");
  for (int k = 0; k < m1; k++) printf(" 0 0");
  printf("\n");
  return 0;
}
