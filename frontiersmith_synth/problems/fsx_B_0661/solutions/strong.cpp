// TIER: strong
// Insight: the reach graph decomposes into weakly-connected components (by
// construction, well-separated in space). Within a component, compute the best
// achievable detonated value for EVERY local ignition budget k (a "value curve") --
// bounded exactly for small components (Y-structures have only 2-4 candidates:
// brute force), local marginal-gain for large ones (dense clusters/decoys). This
// curve can JUMP non-smoothly (a shielded target only lights up once every one of
// its feeder chains is lit together), which a single global marginal-gain pass
// would miss: it always prefers a cluster's smooth, always-visible one-hop gain
// over a chain entry whose payoff is invisible until a sibling entry joins it.
// Then solve a 0/1 GROUP KNAPSACK over components' curves to allocate the global
// M-budget optimally -- this is what actually captures "closure of the propagation
// graph", not just local density.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll NEG = LLONG_MIN / 4;

int N, M;
vector<ll> X, Y, V, R, T;
vector<int> C;
vector<vector<int>> outAdj;

ll simulate(const vector<int>& ignite){
    vector<char> det(N, 0);
    vector<int> cnt(N, 0);
    vector<int> q; q.reserve(N);
    size_t head = 0;
    for (int i : ignite) if (!det[i]){ det[i] = 1; q.push_back(i); }
    while (head < q.size()){
        int u = q[head++];
        for (int j : outAdj[u]){
            if (det[j]) continue;
            if (++cnt[j] >= T[j]){ det[j] = 1; q.push_back(j); }
        }
    }
    ll tot = 0;
    for (int i = 0; i < N; i++) if (det[i]) tot += V[i];
    return tot;
}

int main(){
    if (scanf("%d %d", &N, &M) != 2) return 0;
    X.assign(N, 0); Y.assign(N, 0); V.assign(N, 0); R.assign(N, 0); T.assign(N, 0);
    C.assign(N, 0);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %lld %lld %d", &X[i], &Y[i], &V[i], &R[i], &T[i], &C[i]);

    outAdj.assign(N, {});
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j){
                ll dx = X[i] - X[j], dy = Y[i] - Y[j];
                if (dx * dx + dy * dy <= R[i] * R[i]) outAdj[i].push_back(j);
            }

    // ---- weakly-connected components (union-find over directed edges, either dir) ----
    vector<int> par(N); iota(par.begin(), par.end(), 0);
    function<int(int)> find = [&](int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; };
    auto uni = [&](int a, int b){ a = find(a); b = find(b); if (a != b) par[a] = b; };
    for (int i = 0; i < N; i++) for (int j : outAdj[i]) uni(i, j);

    map<int, vector<int>> compCandidates;   // root -> ignitable node ids in that component
    for (int i = 0; i < N; i++) if (C[i] == 1) compCandidates[find(i)].push_back(i);

    // ---- per-component value curve: curve[k] = best value using exactly k ignitions
    //      from this component's candidates; pick[k] = the node set achieving it. ----
    vector<vector<ll>> curve;                 // [compIdx][k]
    vector<vector<vector<int>>> pick;         // [compIdx][k] -> node list

    for (auto& kv : compCandidates){
        vector<int>& cand = kv.second;
        int sz = (int)cand.size();
        int maxK = min(sz, M);
        vector<ll> cv(maxK + 1, 0);
        vector<vector<int>> pk(maxK + 1);

        if (sz <= 12){
            // exact: brute force every subset, keep the best per popcount.
            for (int mask = 0; mask < (1 << sz); mask++){
                int pc = __builtin_popcount((unsigned)mask);
                if (pc > maxK) continue;
                vector<int> sel;
                for (int b = 0; b < sz; b++) if (mask & (1 << b)) sel.push_back(cand[b]);
                ll val = simulate(sel);
                if (val > cv[pc]){ cv[pc] = val; pk[pc] = sel; }
            }
        } else {
            // local marginal-gain within this component only (no cross-component
            // distraction, so redundant picks -- e.g. a second clique member -- are
            // correctly recognized as zero marginal gain and never crowd out others).
            vector<int> chosen;
            vector<char> used(sz, 0);
            for (int step = 1; step <= maxK; step++){
                ll bestVal = -1; int bestB = -1;
                for (int b = 0; b < sz; b++){
                    if (used[b]) continue;
                    vector<int> trial = chosen; trial.push_back(cand[b]);
                    ll val = simulate(trial);
                    if (val > bestVal){ bestVal = val; bestB = b; }
                }
                used[bestB] = 1;
                chosen.push_back(cand[bestB]);
                cv[step] = simulate(chosen);
                pk[step] = chosen;
            }
        }
        curve.push_back(cv);
        pick.push_back(pk);
    }

    int nc = (int)curve.size();
    // ---- group knapsack: dp[k] = best total value using exactly k ignitions
    //      across all components processed so far. ----
    vector<ll> dp(M + 1, NEG);
    dp[0] = 0;
    vector<vector<int>> choiceHist(nc, vector<int>(M + 1, 0));
    for (int ci = 0; ci < nc; ci++){
        vector<ll> ndp(M + 1, NEG);
        int maxK = (int)curve[ci].size() - 1;
        for (int k = 0; k <= M; k++){
            if (dp[k] == NEG) continue;
            for (int c = 0; c <= maxK && k + c <= M; c++){
                ll val = dp[k] + curve[ci][c];
                if (val > ndp[k + c]){ ndp[k + c] = val; choiceHist[ci][k + c] = c; }
            }
        }
        dp = ndp;
    }

    // dp[M] is reachable because total candidates across components == count of
    // ignitable charges >= M (generator invariant).
    int budget = M;
    if (dp[budget] == NEG){   // extreme defensive fallback, should not trigger
        for (budget = M; budget >= 0; budget--) if (dp[budget] != NEG) break;
    }
    vector<int> result;
    int k = budget;
    for (int ci = nc - 1; ci >= 0; ci--){
        int c = choiceHist[ci][k];
        for (int node : pick[ci][c]) result.push_back(node);
        k -= c;
    }
    // pad (extremely defensive; DP should already reach exactly M) with any unused
    // ignitable charges to guarantee exactly M distinct outputs.
    if ((int)result.size() < M){
        vector<char> used(N, 0);
        for (int node : result) used[node] = 1;
        for (int i = 0; i < N && (int)result.size() < M; i++)
            if (C[i] == 1 && !used[i]){ result.push_back(i); used[i] = 1; }
    }
    for (size_t i = 0; i < result.size(); i++)
        printf("%d%c", result[i] + 1, i + 1 < result.size() ? ' ' : '\n');
    return 0;
}
