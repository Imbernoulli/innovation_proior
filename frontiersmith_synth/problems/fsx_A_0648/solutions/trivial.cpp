// TIER: trivial
// The simplest valid plan: make every step exactly once, one lane per cycle
// (ignore the other K-1 lanes entirely), releasing a step's value the instant
// its true last use has fired (standard remaining-use bookkeeping -- no
// recompute, no attempt at parallel packing). This is exactly the checker's
// baseline construction B=n.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, R, K;
    scanf("%d %d %d %d", &n, &m, &R, &K);
    vector<int> type(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &type[i]);
    vector<vector<int>> preds(n + 1);
    vector<int> outdeg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        preds[v].push_back(u);
        outdeg[u]++;
    }
    vector<int> remaining = outdeg;

    printf("%d\n", n);
    for (int v = 1; v <= n; v++) {
        vector<int> disc;
        for (int u : preds[v]) {
            remaining[u]--;
            if (remaining[u] == 0) disc.push_back(u);
        }
        printf("1 0 %d %d", v, (int)disc.size());
        for (int u : disc) printf(" %d", u);
        printf("\n");
    }
    return 0;
}
