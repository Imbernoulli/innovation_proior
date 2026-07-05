#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
  registerTestlibCmd(argc, argv);

  int M = inf.readInt();
  int N = inf.readInt();
  vector<long long> cap(M);
  for (int i = 0; i < M; i++) cap[i] = inf.readLong();
  vector<vector<long long>> v(N, vector<long long>(M)), c(N, vector<long long>(M));
  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++) {
      v[j][i] = inf.readLong();
      c[j][i] = inf.readLong();
    }

  // ---- read + strictly validate participant assignment ----
  vector<long long> used(M, 0);
  long long F = 0;
  for (int j = 0; j < N; j++) {
    int a = ouf.readInt(0, M, "assign");
    if (a >= 1) {
      used[a - 1] += c[j][a - 1];
      F += v[j][a - 1];
    }
  }
  if (!ouf.seekEof()) quitf(_wa, "trailing output");
  for (int i = 0; i < M; i++)
    if (used[i] > cap[i])
      quitf(_wa, "satellite %d over budget: %lld > %lld", i + 1, used[i], cap[i]);

  // ---- internal baseline B: round-robin reference plan ----
  vector<long long> rem = cap;
  long long B = 0;
  for (int j = 0; j < N; j++) {
    int i = j % M;                 // (j-1 mod M)+1 in 1-indexed == j%M in 0-indexed for j=0..
    if (c[j][i] <= rem[i]) {
      rem[i] -= c[j][i];
      B += v[j][i];
    }
  }
  if (B <= 0) B = 1;

  double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
  quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
  return 0;
}
