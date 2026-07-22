// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The obvious "single greedy pass" approach: sort all edges by combined utility
// (a+b) descending and greedily add an edge to the matching whenever both
// endpoints are still free. This chases raw welfare and never spends a single
// side payment, so it never protects a matched pair from a blocking threat.
int main() {
    int n, m, e; ll lambda;
    scanf("%d %d %d %lld", &n, &m, &e, &lambda);
    vector<int> ei(e), ej(e), ea(e), eb(e), es(e);
    for (int k = 0; k < e; k++)
        scanf("%d %d %d %d %d", &ei[k], &ej[k], &ea[k], &eb[k], &es[k]);

    vector<int> order(e);
    for (int k = 0; k < e; k++) order[k] = k;
    sort(order.begin(), order.end(), [&](int x, int y) {
        int wx = ea[x] + eb[x], wy = ea[y] + eb[y];
        if (wx != wy) return wx > wy;
        return x < y;
    });

    vector<char> usedI(n + 1, 0), usedJ(m + 1, 0);
    vector<int> matchI, matchJ;
    for (int k : order) {
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
