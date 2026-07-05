#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
  registerTestlibCmd(argc, argv);
  int N = inf.readInt();
  int M = inf.readInt();
  int K = inf.readInt();
  vector<int> eu(M), ev(M), ew(M), es(M);
  long long B = 0;  // baseline: every lift on channel 1 -> every pair pays w*s
  for (int i = 0; i < M; i++) {
    eu[i] = inf.readInt();
    ev[i] = inf.readInt();
    ew[i] = inf.readInt();
    es[i] = inf.readInt();
    B += (long long)ew[i] * es[i];
  }
  // read participant channel assignment
  vector<int> c(N + 1);
  for (int i = 1; i <= N; i++) c[i] = ouf.readInt(1, K, "channel");
  if (!ouf.seekEof()) quitf(_wa, "trailing output after N channels");

  long long F = 0;
  for (int i = 0; i < M; i++) {
    int d = abs(c[eu[i]] - c[ev[i]]);
    int pen = es[i] - d;
    if (pen > 0) F += (long long)ew[i] * pen;
  }
  double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
  quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
  return 0;
}
