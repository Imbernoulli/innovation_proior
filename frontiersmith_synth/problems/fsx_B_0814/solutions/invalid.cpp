// TIER: invalid
// Deliberately infeasible: emits a move token far outside [0,n], which the
// checker's bounded read rejects immediately.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n;
    ll bxp, bxm, byp, bym, C, Lmax;
    scanf("%d %lld %lld %lld %lld %lld %lld", &n, &bxp, &bxm, &byp, &bym, &C, &Lmax);
    printf("1\n");
    printf("%d\n", n + 987654);
    return 0;
}
