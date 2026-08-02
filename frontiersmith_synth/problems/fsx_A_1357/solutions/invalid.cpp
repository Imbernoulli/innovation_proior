// TIER: invalid
// Deliberately infeasible: prints correct lines for every internal junction
// EXCEPT the first one encountered, whose declared child-count k is one less
// than the truth (one true child is silently dropped). The checker's
// declared-k-vs-true-child-count check must reject this -> score 0. (This
// works regardless of that junction's degree, including degree 1.)
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<int> par(N + 1, 0);
    vector<vector<int>> children(N + 1);
    for (int i = 2; i <= N; i++){
        int p, c;
        scanf("%d %d", &p, &c);
        par[i] = p;
        children[p].push_back(i);
    }
    for (int v = 1; v <= N; v++) sort(children[v].begin(), children[v].end());

    bool corrupted = false;
    for (int v = 1; v <= N; v++){
        if (children[v].empty()) continue;
        vector<int> kids = children[v];
        if (!corrupted){
            kids.pop_back(); // drop one true child -> k mismatch
            corrupted = true;
        }
        printf("%d %d", v, (int)kids.size());
        for (int u : kids) printf(" %d", u);
        printf("\n");
    }
    return 0;
}
