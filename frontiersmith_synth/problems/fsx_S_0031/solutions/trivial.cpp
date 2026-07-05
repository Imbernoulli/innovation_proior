// TIER: trivial
// Reference Morton (Z-order) chain -- identical construction to the judge baseline.
#include <bits/stdc++.h>
using namespace std;

static inline unsigned long long mortonKey(unsigned int x, unsigned int y) {
    unsigned long long k = 0;
    for (int i = 0; i < 20; i++) {
        k |= ((unsigned long long)((x >> i) & 1u)) << (2 * i);
        k |= ((unsigned long long)((y >> i) & 1u)) << (2 * i + 1);
    }
    return k;
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<int> X(n), Y(n), B(n);
    for (int i = 0; i < n; i++) scanf("%d %d %d", &X[i], &Y[i], &B[i]);

    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        unsigned long long ka = mortonKey((unsigned)X[a], (unsigned)Y[a]);
        unsigned long long kb = mortonKey((unsigned)X[b], (unsigned)Y[b]);
        if (ka != kb) return ka < kb;
        return a < b;
    });

    printf("%d\n", n - 1);
    for (int i = 1; i < n; i++)
        printf("%d %d\n", ord[i-1] + 1, ord[i] + 1);
    return 0;
}
