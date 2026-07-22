// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e15;

// The obvious "account for popularity" approach: reserve every cabinet's own
// do-nothing memory first (so it is always feasible), split the LEFTOVER
// (surplus) budget proportionally to popularity weight W_c, then solve each
// cabinet's stride-selection DP independently within that fixed share. This
// is exactly the textbook per-trie DP applied to a budget that was silently
// pre-partitioned per subtree -- it never reconsiders the split once made,
// so it cannot move memory from a cabinet with a poor cost-per-probe-saved
// ratio to one with a good ratio, even when that ratio gap is huge.

struct DP {
    int D;
    vector<vector<ll>> dp;   // dp[depth][levels]
    vector<vector<int>> par; // par[depth][levels] = previous depth
};

static DP buildDP(int D, const vector<ll>& T) {
    DP r; r.D = D;
    r.dp.assign(D + 1, vector<ll>(D + 1, INF));
    r.par.assign(D + 1, vector<int>(D + 1, -1));
    r.dp[0][0] = 0;
    for (int d = 0; d < D; d++)
        for (int j = 0; j <= D; j++) {
            if (r.dp[d][j] >= INF) continue;
            for (int nd = d + 1; nd <= D; nd++) {
                ll cost = T[d] * (1LL << (nd - d));
                ll cand = r.dp[d][j] + cost;
                if (cand < r.dp[nd][j + 1]) { r.dp[nd][j + 1] = cand; r.par[nd][j + 1] = d; }
            }
        }
    return r;
}

static vector<int> reconstruct(const DP& r, int j) {
    vector<int> strides;
    int d = r.D, lv = j;
    while (lv > 0) {
        int pd = r.par[d][lv];
        strides.push_back(d - pd);
        d = pd; lv--;
    }
    reverse(strides.begin(), strides.end());
    return strides;
}

int main() {
    int K; ll M;
    scanf("%d %lld", &K, &M);
    vector<int> D(K);
    vector<ll> W(K), baseMem(K);
    vector<vector<ll>> T(K);
    vector<DP> dps(K);

    for (int c = 0; c < K; c++) {
        scanf("%d %lld", &D[c], &W[c]);
        T[c].resize(D[c] + 1);
        for (int d = 0; d <= D[c]; d++) scanf("%lld", &T[c][d]);
        ll bm = 0;
        for (int d = 0; d < D[c]; d++) bm += T[c][d] * 2;
        baseMem[c] = bm;
        dps[c] = buildDP(D[c], T[c]);
    }

    ll totalBase = 0, sumW = 0;
    for (int c = 0; c < K; c++) { totalBase += baseMem[c]; sumW += W[c]; }
    ll surplus = M - totalBase;
    if (surplus < 0) surplus = 0;
    if (sumW <= 0) sumW = 1;

    for (int c = 0; c < K; c++) {
        ll extra = (ll)((__int128)surplus * W[c] / sumW);
        ll budget = baseMem[c] + extra;
        int bestJ = D[c];
        for (int j = 1; j <= D[c]; j++) {
            if (dps[c].dp[D[c]][j] <= budget) { bestJ = j; break; }
        }
        vector<int> strides = reconstruct(dps[c], bestJ);
        printf("%d", (int)strides.size());
        for (int s : strides) printf(" %d", s);
        printf("\n");
    }
    return 0;
}
