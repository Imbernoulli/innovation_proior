// TIER: strong
// Density-ordered constructive assignment + local reassignment search.
#include <bits/stdc++.h>
using namespace std;
int main() {
  int M, N;
  if (scanf("%d %d", &M, &N) != 2) return 0;
  vector<long long> cap(M), rem(M);
  for (int i = 0; i < M; i++) { scanf("%lld", &cap[i]); rem[i] = cap[i]; }
  vector<vector<long long>> v(N, vector<long long>(M)), c(N, vector<long long>(M));
  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++) scanf("%lld %lld", &v[j][i], &c[j][i]);

  // order debris by best value-density -> prioritize high-yield, cheap captures
  vector<int> order(N);
  iota(order.begin(), order.end(), 0);
  vector<double> pot(N);
  for (int j = 0; j < N; j++) {
    double best = 0;
    for (int i = 0; i < M; i++) best = max(best, (double)v[j][i] / (double)c[j][i]);
    pot[j] = best;
  }
  sort(order.begin(), order.end(), [&](int a, int b) { return pot[a] > pot[b]; });

  vector<int> ans(N, 0);  // 0 = unassigned, else 1..M
  // greedy pass: assign each debris to the feasible satellite of max value
  for (int idx = 0; idx < N; idx++) {
    int j = order[idx];
    int best = -1; long long bv = -1, bc = 0;
    for (int i = 0; i < M; i++) {
      if (c[j][i] <= rem[i] && (v[j][i] > bv || (v[j][i] == bv && c[j][i] < bc))) {
        bv = v[j][i]; bc = c[j][i]; best = i;
      }
    }
    if (best >= 0) { rem[best] -= c[j][best]; ans[j] = best + 1; }
  }

  // local search: repeatedly move/insert a debris to a satellite that raises total value
  for (int pass = 0; pass < 6; pass++) {
    bool changed = false;
    for (int j = 0; j < N; j++) {
      int a = ans[j];                       // current satellite (1-indexed) or 0
      long long curVal = a ? v[j][a - 1] : 0;
      int bestI = a; long long bestVal = curVal;
      for (int i = 0; i < M; i++) {
        long long room = rem[i] + (a == i + 1 ? c[j][i] : 0);
        if (c[j][i] <= room && v[j][i] > bestVal) { bestVal = v[j][i]; bestI = i + 1; }
      }
      if (bestI != a) {
        if (a) rem[a - 1] += c[j][a - 1];
        rem[bestI - 1] -= c[j][bestI - 1];
        ans[j] = bestI;
        changed = true;
      }
    }
    if (!changed) break;
  }

  for (int j = 0; j < N; j++) printf("%d\n", ans[j]);
  return 0;
}
