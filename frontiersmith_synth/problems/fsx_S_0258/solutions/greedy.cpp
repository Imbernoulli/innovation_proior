// TIER: greedy
// One pass in input order: give each debris to the feasible satellite of highest hazard weight.
#include <bits/stdc++.h>
using namespace std;
int main() {
  int M, N;
  if (scanf("%d %d", &M, &N) != 2) return 0;
  vector<long long> rem(M);
  for (int i = 0; i < M; i++) scanf("%lld", &rem[i]);
  vector<int> ans(N, 0);
  for (int j = 0; j < N; j++) {
    vector<long long> v(M), c(M);
    for (int i = 0; i < M; i++) scanf("%lld %lld", &v[i], &c[i]);
    int best = -1; long long bv = -1, bc = 0;
    for (int i = 0; i < M; i++) {
      if (c[i] <= rem[i] && (v[i] > bv || (v[i] == bv && c[i] < bc))) {
        bv = v[i]; bc = c[i]; best = i;
      }
    }
    if (best >= 0) { rem[best] -= c[best]; ans[j] = best + 1; }
  }
  for (int j = 0; j < N; j++) printf("%d\n", ans[j]);
  return 0;
}
