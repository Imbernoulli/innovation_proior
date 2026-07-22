// TIER: invalid
// Deliberately infeasible: claims to seed far more cells than the budget allows.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, T, B, M; scanf("%d %d %d %d", &N, &T, &B, &M);
    char buf[300]; scanf("%299s", buf);
    // k alone already exceeds the allowed budget -- the checker rejects it before
    // reading any further tokens, so a short, cheap tail is enough.
    int k = B + 1;
    printf("%d\n", k);
    for (int i = 0; i < k; i++) printf("%d ", i % (N > 0 ? N : 1));
    printf("\n");
    return 0;
}
