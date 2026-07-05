#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
  registerGen(argc, argv, 1);
  int t = atoi(argv[1]);

  // difficulty / scale ladder
  int M, N;
  if (t <= 1)      { M = 2;  N = 3;    }
  else if (t == 2) { M = 4;  N = 30;   }
  else if (t == 3) { M = 6;  N = 100;  }
  else if (t == 4) { M = 8;  N = 300;  }
  else if (t == 5) { M = 10; N = 600;  }
  else if (t == 6) { M = 15; N = 1000; }
  else if (t == 7) { M = 20; N = 1500; }
  else if (t == 8) { M = 25; N = 2000; }
  else if (t == 9) { M = 30; N = 2500; }
  else             { M = 40; N = 3000; }

  int mode = t % 3;  // 0 uniform, 1 value~cost correlated, 2 skewed values/budgets

  vector<vector<int>> v(N, vector<int>(M)), c(N, vector<int>(M));
  for (int j = 0; j < N; j++) {
    // a subset of debris are "high hazard" in skewed mode
    bool hot = (mode == 2) && (rnd.next(0, 4) == 0);
    for (int i = 0; i < M; i++) {
      int cost = rnd.next(1, 1000);
      int val;
      if (mode == 0) {
        val = rnd.next(1, 1000);
      } else if (mode == 1) {
        // value loosely correlated with cost -> value-greedy and density-greedy diverge
        val = min(1000, max(1, cost + rnd.next(-250, 250)));
      } else {
        val = hot ? rnd.next(700, 1000) : rnd.next(1, 300);
      }
      c[j][i] = cost;
      v[j][i] = val;
    }
  }

  // per-satellite delta-v budgets: round-robin gives ~ceil(N/M) debris per satellite,
  // average cost ~500, so a factor in [200,400] makes the budget bind but stay feasible.
  int perSat = (int)ceil((double)N / M);
  vector<long long> cap(M);
  for (int i = 0; i < M; i++) {
    long long factor = rnd.next(80, 200);
    if (mode == 2) {
      // skew budgets: some fuel-rich, some fuel-poor
      factor = (rnd.next(0, 2) == 0) ? rnd.next(40, 110) : rnd.next(160, 280);
    }
    long long budget = (long long)perSat * factor;
    // guarantee every satellite can afford at least one max-cost capture -> baseline B > 0
    cap[i] = max<long long>(budget, 1000);
  }

  printf("%d %d\n", M, N);
  for (int i = 0; i < M; i++) printf("%lld%c", cap[i], i + 1 == M ? '\n' : ' ');
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      printf("%d %d%c", v[j][i], c[j][i], i + 1 == M ? '\n' : ' ');
    }
  }
  return 0;
}
