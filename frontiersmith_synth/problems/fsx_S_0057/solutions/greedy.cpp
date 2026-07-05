// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
int main() {
  int N, M, K;
  if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
  vector<int> eu(M), ev(M), ew(M), es(M);
  vector<long long> wdeg(N + 1, 0);
  vector<vector<int>> adj(N + 1);
  for (int i = 0; i < M; i++) {
    scanf("%d %d %d %d", &eu[i], &ev[i], &ew[i], &es[i]);
    adj[eu[i]].push_back(i);
    adj[ev[i]].push_back(i);
    wdeg[eu[i]] += ew[i];
    wdeg[ev[i]] += ew[i];
  }
  vector<int> order(N);
  for (int i = 0; i < N; i++) order[i] = i + 1;
  sort(order.begin(), order.end(), [&](int a, int b) { return wdeg[a] > wdeg[b]; });
  vector<int> c(N + 1, 0);  // 0 = not yet tuned
  for (int idx = 0; idx < N; idx++) {
    int u = order[idx];
    long long best = LLONG_MAX;
    int bestc = 1;
    for (int ch = 1; ch <= K; ch++) {
      long long cost = 0;
      for (int e : adj[u]) {
        int o = (eu[e] == u ? ev[e] : eu[e]);
        if (c[o] == 0) continue;  // ignore not-yet-tuned neighbors
        int d = abs(ch - c[o]);
        int pen = es[e] - d;
        if (pen > 0) cost += (long long)ew[e] * pen;
      }
      if (cost < best) { best = cost; bestc = ch; }
    }
    c[u] = bestc;
  }
  for (int i = 1; i <= N; i++) printf("%d%c", c[i], i == N ? '\n' : ' ');
  return 0;
}
