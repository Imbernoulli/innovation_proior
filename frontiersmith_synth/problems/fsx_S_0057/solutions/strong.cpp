// TIER: strong
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

  auto bestChannel = [&](int u, const vector<int>& c) -> int {
    long long best = LLONG_MAX;
    int bc = c[u] ? c[u] : 1;
    for (int ch = 1; ch <= K; ch++) {
      long long cost = 0;
      for (int e : adj[u]) {
        int o = (eu[e] == u ? ev[e] : eu[e]);
        int co = c[o];
        if (co == 0) continue;
        int d = abs(ch - co);
        int pen = es[e] - d;
        if (pen > 0) cost += (long long)ew[e] * pen;
      }
      if (cost < best) { best = cost; bc = ch; }
    }
    return bc;
  };

  // greedy initialization in weighted-degree order
  vector<int> c(N + 1, 0);
  for (int idx = 0; idx < N; idx++) {
    int u = order[idx];
    c[u] = bestChannel(u, c);
  }
  // iterated local search: re-tune each lift given ALL current neighbors
  for (int sweep = 0; sweep < 30; sweep++) {
    bool improved = false;
    for (int idx = 0; idx < N; idx++) {
      int u = order[idx];
      int nc = bestChannel(u, c);
      if (nc != c[u]) { c[u] = nc; improved = true; }
    }
    if (!improved) break;
  }
  for (int i = 1; i <= N; i++) printf("%d%c", c[i], i == N ? '\n' : ' ');
  return 0;
}
