// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e15;

// The insight: the K per-cabinet DP curves (memory needed for exactly j
// probes) are exact and cheap to compute -- the classic per-trie DP is not
// wrong, it is just being fed the wrong (pre-partitioned) budget. Instead of
// fixing a share per cabinet, treat "reduce cabinet c's probe count by one"
// as a purchasable step with a known memory COST and a known popularity-
// weighted BENEFIT (W_c). Repeatedly buy the cheapest available step
// (memory cost per unit of weighted benefit) across ALL cabinets until the
// shared budget is exhausted. This is a discrete Lagrangian / market-
// clearing view of the global-budget allocation: cabinets compete for the
// same pool of memory on a common "price" scale instead of each getting an
// isolated, fixed slice.

struct DP {
    int D;
    vector<vector<ll>> dp;
    vector<vector<int>> par;
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

struct Step { double ratio; ll cost; int c; int targetJ; };
struct StepCmp { bool operator()(const Step& a, const Step& b) const { return a.ratio > b.ratio; } };

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

    vector<int> curJ(K);
    ll curMem = 0;
    for (int c = 0; c < K; c++) { curJ[c] = D[c]; curMem += baseMem[c]; }

    priority_queue<Step, vector<Step>, StepCmp> pq;
    auto pushStep = [&](int c) {
        int j = curJ[c];
        if (j <= 1) return;
        ll cost = dps[c].dp[D[c]][j - 1] - dps[c].dp[D[c]][j];
        double ratio = (double)cost / (double)W[c];
        pq.push({ratio, cost, c, j - 1});
    };
    for (int c = 0; c < K; c++) pushStep(c);

    while (!pq.empty()) {
        Step st = pq.top(); pq.pop();
        if (st.targetJ != curJ[st.c] - 1) continue;  // stale (shouldn't happen, safety)
        if (curMem + st.cost <= M) {
            curMem += st.cost;
            curJ[st.c] = st.targetJ;
            pushStep(st.c);
        }
        // else: this step is unaffordable right now; drop it (other cabinets'
        // cheaper steps may still fit in the remaining budget).
    }

    for (int c = 0; c < K; c++) {
        vector<int> strides = reconstruct(dps[c], curJ[c]);
        printf("%d", (int)strides.size());
        for (int s : strides) printf(" %d", s);
        printf("\n");
    }
    return 0;
}
