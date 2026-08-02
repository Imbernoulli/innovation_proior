// TIER: trivial
// Ascending-id clearing order at every junction -- exactly the checker's own
// baseline construction B. No attempt to reason about search-number or cost.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<int> par(N + 1, 0), cst(N + 1, 0);
    vector<vector<int>> children(N + 1);
    for (int i = 2; i <= N; i++){
        int p, c;
        scanf("%d %d", &p, &c);
        par[i] = p; cst[i] = c;
        children[p].push_back(i);
    }
    for (int v = 1; v <= N; v++) sort(children[v].begin(), children[v].end());

    for (int v = 1; v <= N; v++){
        if (children[v].empty()) continue;
        printf("%d %d", v, (int)children[v].size());
        for (int u : children[v]) printf(" %d", u);
        printf("\n");
    }
    return 0;
}
