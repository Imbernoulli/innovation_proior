// TIER: trivial
// Replicates the judge's round-robin reference plan exactly -> scores the 0.1 baseline.
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
    int i = j % M;
    if (c[i] <= rem[i]) { rem[i] -= c[i]; ans[j] = i + 1; }
  }
  for (int j = 0; j < N; j++) printf("%d\n", ans[j]);
  return 0;
}
