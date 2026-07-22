// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Reproduces the checker's internal baseline exactly: scan edges in the order
// they appear in the input and greedily take a pair if both endpoints are still
// free. Never uses side payments.
int main() {
    int n, m, e; ll lambda;
    scanf("%d %d %d %lld", &n, &m, &e, &lambda);
    vector<int> ei(e), ej(e), ea(e), eb(e), es(e);
    for (int k = 0; k < e; k++)
        scanf("%d %d %d %d %d", &ei[k], &ej[k], &ea[k], &eb[k], &es[k]);

    vector<char> usedI(n + 1, 0), usedJ(m + 1, 0);
    vector<int> matchI, matchJ;
    for (int k = 0; k < e; k++) {
        if (!usedI[ei[k]] && !usedJ[ej[k]]) {
            usedI[ei[k]] = usedJ[ej[k]] = 1;
            matchI.push_back(ei[k]);
            matchJ.push_back(ej[k]);
        }
    }
    printf("%d\n", (int)matchI.size());
    for (size_t t = 0; t < matchI.size(); t++)
        printf("%d %d 0\n", matchI[t], matchJ[t]);
    return 0;
}
