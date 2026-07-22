// TIER: invalid
// Deliberately infeasible: braids every single component, blowing the
// replication budget R (R is always far smaller than F in this family).
#include <bits/stdc++.h>
using namespace std;

int main(){
    int F, Q, R; long long P, K;
    if (scanf("%d %d %lld %lld %d", &F, &Q, &P, &K, &R) != 5) return 0;
    for (int i = 0; i < F; i++){ long long w; scanf("%lld", &w); }
    for (int i = 1; i <= F; i++) printf("1 2\n");   // braid on every field -> budget blown
    return 0;
}
