// TIER: invalid
// Deliberately emits an out-of-range satellite index -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
  int M, N;
  if (scanf("%d %d", &M, &N) != 2) return 0;
  printf("%d\n", M + 1);                 // out of [0, M]
  for (int j = 1; j < N; j++) printf("0\n");
  return 0;
}
