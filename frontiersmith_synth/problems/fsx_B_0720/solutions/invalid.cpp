// TIER: invalid
// Deliberately infeasible: -1 lies outside every task's [e_i, l_i] window
// (e_i >= 1 always), so the checker's first bounded read rejects it.
#include <cstdio>
int main(){
    int N; long long H, C;
    scanf("%d %lld %lld", &N, &H, &C);
    for (int i = 0; i < N; i++) printf("-1\n");
    return 0;
}
