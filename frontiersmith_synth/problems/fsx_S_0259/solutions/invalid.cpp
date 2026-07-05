// TIER: invalid
// Deliberately infeasible: install a single pipe segment. For n >= 2 this leaves the
// network disconnected, so the checker must score it 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n, m, D;
    scanf("%d %d %d", &n, &m, &D);
    printf("1\n1\n");   // one segment -> cannot span n>=3 junctions -> infeasible
    return 0;
}
