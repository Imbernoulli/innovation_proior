// TIER: invalid
// Deliberately infeasible: opens one facility whose printed index is m+5,
// strictly outside the valid [1,m] range, so the checker's bounded read fails
// immediately -> no Ratio printed -> scores 0.
#include <cstdio>
int main(){
    int n, m; long long Bud, QW;
    scanf("%d %d %lld %lld", &n, &m, &Bud, &QW);
    long long tmp;
    for (int i = 0; i < n; i++) scanf("%lld %lld %lld", &tmp, &tmp, &tmp);
    for (int j = 0; j < m; j++) scanf("%lld %lld %lld %lld", &tmp, &tmp, &tmp, &tmp);
    printf("1\n%d\n", m + 5);
    for (int i = 0; i < n; i++) printf("1 ");
    printf("\n");
    return 0;
}
