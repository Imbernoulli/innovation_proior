// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
  int N, M, K;
  if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
  for (int i = 0; i < M; i++) { int u, v, w, s; scanf("%d %d %d %d", &u, &v, &w, &s); }
  for (int i = 0; i < N; i++) printf("1%c", i + 1 == N ? '\n' : ' ');
  return 0;
}
