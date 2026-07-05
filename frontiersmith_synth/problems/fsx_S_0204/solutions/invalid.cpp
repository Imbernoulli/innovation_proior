// TIER: invalid
// Deliberately infeasible: never mills anything (all Pause) while W >= 1, so the demand
// is unmet -> checker rejects -> score 0. Also emits an out-of-range token to exercise the
// checker's bounded reads.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int T; ll W; int q; ll K;
    if (scanf("%d %lld %d %lld", &T, &W, &q, &K) != 4) return 0;
    for (int t = 1; t <= T; t++) { int a, g, s, d; scanf("%d %d %d %d", &a, &g, &s, &d); }
    for (int t = 1; t <= T; t++) {
        // slot 1 gets an illegal mode (5); the rest are Pause -> demand never met anyway.
        printf("%d", t == 1 ? 5 : 0);
        putchar(t == T ? '\n' : ' ');
    }
    return 0;
}
