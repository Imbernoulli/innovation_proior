// TIER: trivial
// Reference first-fit: every forager to its lowest-indexed reachable patch.
// This reproduces the checker's baseline B* exactly -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int P, B;
    scanf("%d %d", &P, &B);
    vector<long long> S(P + 1);
    for (int j = 1; j <= P; j++) scanf("%lld", &S[j]);
    vector<int> ans(B + 1);
    for (int i = 1; i <= B; i++) {
        int m; scanf("%d", &m);
        int firstP = -1;
        for (int k = 0; k < m; k++) {
            int p, a; scanf("%d %d", &p, &a);
            if (k == 0) firstP = p;
        }
        ans[i] = firstP; // lowest-indexed reachable (input sorted ascending)
    }
    for (int i = 1; i <= B; i++) printf("%d%c", ans[i], i == B ? '\n' : ' ');
    return 0;
}
