// TIER: strong
// Insight: a shared cut's per-participant cost (L_g/2 extra) grows with L_g just
// as fast as its per-participant credit, so the real bottleneck is not "biggest
// line" but QUORUM -- a line only pays off once 2+ members can jointly afford
// it, and whichever order you commit parts' budgets in is IRREVOCABLE (budgets
// only shrink). That makes the naive "biggest line first" recipe just one point
// in a space of commit-orders, with no guarantee of being a good one: locking a
// scarce heat-tolerant part into one giant line can silently starve many cheap
// lines that each only needed a sliver of that part's budget to reach quorum.
//
// Strategy: evaluate a small PORTFOLIO of commit-orders (descending cut length,
// ascending cut length, and ascending "cost per line-member" density) on
// independent budget copies, and keep whichever total saved length is largest.
// Ascending order wins whenever many cheap shared contacts are being starved by
// one dominant line (the planted trap clusters); descending still wins when a
// single line is uncontested (e.g. a "needle" with no competing claims on its
// members) or when a dense, uncorrelated random instance happens to favor early
// large claims. Comparing orders costs nothing (each pass is O(sum k_g)) and can
// never do worse than the plain greedy recipe.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<vector<int>> mem;
vector<ll> L, baseline;

// Run one greedy commit-order pass over lines `ord`, return (F, plan).
pair<ll, vector<pair<int, vector<int>>>> runOrder(const vector<ll>& Hraw, const vector<int>& ord){
    vector<ll> remain(N + 1);
    for (int i = 1; i <= N; i++) remain[i] = Hraw[i] - baseline[i];
    ll F = 0;
    vector<pair<int, vector<int>>> plan;
    for (int g : ord){
        vector<int> ok;
        for (int p : mem[g]) if (remain[p] >= L[g] / 2) ok.push_back(p);
        if ((int)ok.size() >= 2){
            for (int p : ok) remain[p] -= L[g] / 2;
            F += (ll)(ok.size() - 1) * L[g];
            plan.push_back({g, ok});
        }
    }
    return {F, plan};
}

int main(){
    scanf("%d %d", &N, &M);
    vector<ll> H(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &H[i]);
    mem.assign(M + 1, {});
    L.assign(M + 1, 0);
    baseline.assign(N + 1, 0);
    for (int g = 1; g <= M; g++){
        int k; ll Lg; scanf("%d %lld", &k, &Lg);
        L[g] = Lg; mem[g].resize(k);
        for (int j = 0; j < k; j++){ scanf("%d", &mem[g][j]); baseline[mem[g][j]] += Lg / 2; }
    }

    vector<int> ordDesc(M), ordAsc(M), ordDensity(M);
    for (int g = 1; g <= M; g++) ordDesc[g - 1] = ordAsc[g - 1] = ordDensity[g - 1] = g;
    sort(ordDesc.begin(), ordDesc.end(), [&](int a, int b){
        if (L[a] != L[b]) return L[a] > L[b]; return a < b; });
    sort(ordAsc.begin(), ordAsc.end(), [&](int a, int b){
        if (L[a] != L[b]) return L[a] < L[b]; return a < b; });
    // "density" order: cost per potential participant ascending (favors lines
    // that are cheap to fully staff, a different tie-break among small lines).
    sort(ordDensity.begin(), ordDensity.end(), [&](int a, int b){
        double da = (double)L[a] / max(1, (int)mem[a].size());
        double db = (double)L[b] / max(1, (int)mem[b].size());
        if (da != db) return da < db; return a < b; });

    auto rDesc = runOrder(H, ordDesc);
    auto rAsc = runOrder(H, ordAsc);
    auto rDen = runOrder(H, ordDensity);

    auto* best = &rDesc;
    if (rAsc.first > best->first) best = &rAsc;
    if (rDen.first > best->first) best = &rDen;

    printf("%d\n", (int)best->second.size());
    for (auto &pr : best->second){
        printf("%d %d", pr.first, (int)pr.second.size());
        for (int p : pr.second) printf(" %d", p);
        printf("\n");
    }
    return 0;
}
