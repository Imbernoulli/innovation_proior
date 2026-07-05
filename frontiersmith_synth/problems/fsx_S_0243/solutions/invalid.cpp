// TIER: invalid
// Deliberately infeasible: claims two overlapping placements, so the checker's
// disjointness check rejects the output -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int W, H;
    if (scanf("%d %d", &W, &H) != 2) return 0;
    int K; for (scanf("%d", &K); K > 0; K--) { int r, c; scanf("%d %d", &r, &c); }
    int T; scanf("%d", &T);
    for (int i = 0; i < T; i++) { int sz; scanf("%d", &sz); for (int j = 0; j < sz; j++){int r,c;scanf("%d %d",&r,&c);} }
    // Two identical 2x2 squares at (0,0) -> overlap -> infeasible.
    printf("2\n");
    printf("4 0 0 0 1 1 0 1 1\n");
    printf("4 0 0 0 1 1 0 1 1\n");
    return 0;
}
