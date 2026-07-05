#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
  registerGen(argc, argv, 1);
  int t = atoi(argv[1]);
  long long N, M, K, wmax, sHi;
  if (t <= 1) {
    // tiny, example-scale sanity case
    N = 6; M = 8; K = 3; wmax = 5; sHi = 2;
  } else {
    N = 200 + 150LL * t;                 // 500 .. 1700
    M = 8 * N;                           // average degree ~16
    K = max(5LL, 16 - (long long)t);     // 14 .. 6 (fewer channels = harder)
    wmax = 100;
    sHi = min(K, 2 + (long long)t / 3);  // 2 .. 5 separation
  }
  if (M > 20000) M = 20000;
  printf("%lld %lld %lld\n", N, M, K);
  for (long long i = 0; i < M; i++) {
    int u = rnd.next(1, (int)N);
    int v = rnd.next(1, (int)N);
    while (v == u) v = rnd.next(1, (int)N);
    int w = rnd.next(1, (int)wmax);
    int s = rnd.next(1, (int)sHi);
    printf("%d %d %d %d\n", u, v, w, s);
  }
  return 0;
}
