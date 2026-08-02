// TIER: strong
// Insight: at each junction, ordering children u_1..u_k to minimize
//   max_i ( x_{u_i} + c_{u_1} + ... + c_{u_{i-1}} )
// is exactly the classical "minimize the maximum of (value_i + accumulated
// setup time before i)" scheduling shape. An adjacent-swap exchange argument
// gives the correct pairwise rule: place u before w iff
//   max(x_u, x_w + c_u)  <=  max(x_w, x_u + c_w)
// (swapping only changes the pair's own two contributed terms -- everything
// scheduled after the pair sees the same total prefix cost c_u+c_w either
// way). This jointly weighs each branch's search-number requirement AGAINST
// its sentry cost, unlike greedy which only looks at x. Applied bottom-up at
// every branching junction via the same recursive tree DP.
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
        sort(kids.begin(), kids.end(), [](int u, int w){
            ll a = max(x[u], x[w] + (ll)cst[u]);
            ll b = max(x[w], x[u] + (ll)cst[w]);
            if (a != b) return a < b;
            return u < w; // deterministic tie-break
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
