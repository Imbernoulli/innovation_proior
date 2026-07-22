// TIER: trivial
// Spine-only: keep exactly the M spine links, buy no reinforcement.
// This reproduces the checker's baseline B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int n, m, M, K, D, Bbudget; ll P;
    if (scanf("%d %d %d %d %d %d %lld", &n, &m, &M, &K, &D, &Bbudget, &P) != 7) return 0;
    printf("%d\n", M);
    for (int i = 0; i < M; i++) printf("%d%c", i, i + 1 < M ? ' ' : '\n');
    return 0;
}
