// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
  int N, M, K;
  if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
  for (int i = 0; i < M; i++) { int u, v, w, s; scanf("%d %d %d %d", &u, &v, &w, &s); }
  // deliberately out-of-range channel (K+1) -> infeasible, must score 0
  printf("%d", K + 1);
  for (int i = 1; i < N; i++) printf(" 1");
  printf("\n");
  return 0;
}
