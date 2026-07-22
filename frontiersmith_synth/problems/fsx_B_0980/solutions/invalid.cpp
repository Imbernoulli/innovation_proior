// TIER: invalid
// Deliberately infeasible: installs at segment 1 but requests an extraction far
// beyond both its hardware capacity and the Tsink physical limit. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, K, Tsink;
    scanf("%d %d %d", &T, &K, &Tsink);
    long long q, theta, eta, cap;
    scanf("%lld %lld %lld %lld", &q, &theta, &eta, &cap);
    printf("1\n1 1900000000\n");
    return 0;
}
