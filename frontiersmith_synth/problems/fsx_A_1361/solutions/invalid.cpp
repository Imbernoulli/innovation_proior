// TIER: invalid
// Deliberately infeasible: claims a bidder index equal to n (out of range 0..n-1) in the
// first market's only group.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    scanf("%d", &T);
    for (int t = 0; t < T; t++) {
        int n, m;
        scanf("%d %d", &n, &m);
        vector<long long> s(m);
        for (int a = 0; a < m; a++) scanf("%lld", &s[a]);
        for (int i = 0; i < n; i++)
            for (int a = 0; a < m; a++) { long long b; scanf("%lld", &b); }
        if (t == 0) {
            printf("1\n1 %d\n", n); // n is out of range [0, n-1]
        } else {
            printf("0\n");
        }
    }
    return 0;
}
