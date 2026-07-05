// TIER: invalid
// Deliberately infeasible: assigns every parcel to a non-existent drone index m+1
// (out of range) -> the checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    vector<long long> c(m + 1);
    for (int i = 1; i <= m; i++) scanf("%lld", &c[i]);
    long long tmp;
    for (int j = 1; j <= n; j++) {
        scanf("%lld", &tmp);
        for (int i = 1; i <= m; i++) scanf("%lld", &tmp);
    }
    for (int j = 1; j <= n; j++) printf("%d%c", m + 1, j == n ? '\n' : ' ');
    return 0;
}
