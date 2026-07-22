// TIER: invalid
// Deliberately infeasible: dumps every edge into the day shift, blowing past capDay whenever
// capDay < m (true on every generated test). Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    double tau;
    scanf("%lf", &tau);
    long long capDay, capNight;
    scanf("%lld %lld", &capDay, &capNight);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        printf("0\n");
    }
    return 0;
}
