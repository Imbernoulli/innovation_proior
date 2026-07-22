// TIER: invalid
// Deliberately infeasible: a single page holding only tile 1, so the in-order walk
// does not cover 1..N. The checker must reject it (score 0).
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, B, S; scanf("%d %d %d", &N, &B, &S);
    // one page, one tile, two null children -> tiles 2..N missing
    printf("1 1\n1 1 0 0\n");
    return 0;
}
