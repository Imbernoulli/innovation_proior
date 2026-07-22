// TIER: invalid
// Deliberately infeasible: emits one template using the digit '9', which is outside
// the required alphabet {0,1,2,3}. The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int K, Mmax, L, Rmax;
    ll Budget;
    if (scanf("%d %d %d %d %lld", &K, &Mmax, &L, &Rmax, &Budget) != 5) return 0;
    // consume the rest of the input (not strictly required, but tidy)
    for (int k = 0; k < K; k++) {
        int n;
        scanf("%d", &n);
        static char buf[64];
        for (int i = 0; i < n; i++) scanf("%s", buf);
    }
    printf("1\n");
    for (int i = 0; i < L; i++) putchar('9');
    printf(" 0\n");
    return 0;
}
