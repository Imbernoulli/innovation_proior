// TIER: invalid
// Deliberately infeasible: keeps Bbudget+1 links (over budget) -> must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int n, m, M, K, D, Bbudget; ll P;
    if (scanf("%d %d %d %d %d %d %lld", &n, &m, &M, &K, &D, &Bbudget, &P) != 7) return 0;
    int q = Bbudget + 1;
    printf("%d\n", q);
    for (int i = 0; i < q; i++) printf("%d%c", i, i + 1 < q ? ' ' : '\n');
    return 0;
}
