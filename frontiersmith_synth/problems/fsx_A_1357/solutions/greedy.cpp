// TIER: greedy
// Natural first idea: at every junction, clear the child branch that needs
// the MOST active sweepers (largest x_child) first, then work down. This
// correctly captures the search-number half of the recursion but is totally
// blind to sentry cost c_child -- a cheap-x, expensive-c branch has no signal
// telling greedy to move it, so it gets buried wherever its small x lands it,
// and every sibling after it inherits its full sentry cost.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N;
vector<int> par, cst;
vector<vector<int>> children;
vector<ll> x;

int main(){
    if (scanf("%d", &N) != 1) return 0;
    par.assign(N + 1, 0); cst.assign(N + 1, 0);
    children.assign(N + 1, {});
    for (int i = 2; i <= N; i++){
        int p, c;
        scanf("%d %d", &p, &c);
        par[i] = p; cst[i] = c;
        children[p].push_back(i);
    }
    for (int v = 1; v <= N; v++) sort(children[v].begin(), children[v].end());

    x.assign(N + 1, 0);
    vector<vector<int>> order(N + 1);

    for (int v = N; v >= 1; v--){
        if (children[v].empty()){ x[v] = 1; continue; }
        vector<int> kids = children[v];
        // descending by x_child only; ties broken by ascending id for determinism
        sort(kids.begin(), kids.end(), [](int a, int b){
            if (x[a] != x[b]) return x[a] > x[b];
            return a < b;
        });
        order[v] = kids;
        ll run = 0, best = 0;
        for (int u : kids){ best = max(best, x[u] + run); run += cst[u]; }
        x[v] = best;
    }

    for (int v = 1; v <= N; v++){
        if (children[v].empty()) continue;
        printf("%d %d", v, (int)order[v].size());
        for (int u : order[v]) printf(" %d", u);
        printf("\n");
    }
    return 0;
}
