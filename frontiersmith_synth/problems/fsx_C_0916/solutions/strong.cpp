// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e18;

// Insight: distance here is ONE-SIDED -- a stake at s only helps a query at
// pos >= s, and for any fixed group of trace points served by one stake, the
// best position for that stake is the group's LEFT EDGE (the smallest pos in
// the group), because every served point's cost ceil((pos-s)/W) is
// non-increasing as s grows, and s can never exceed the group's minimum
// without abandoning that point to an earlier stake. So the optimal stakes
// are always AT trace positions (or 0), and the whole problem reduces to an
// optimal weighted PARTITION of the sorted trace into <= K contiguous
// groups, each charged at its own left edge under the W-quantized probe
// cost -- solved here with a straightforward O(Q^2) + O(Q^2*K) DP.

static inline ll ceilDiv(ll a, ll b) { return (a + b - 1) / b; }

int main() {
    ll L, K, W;
    int Q;
    if (!(cin >> L >> K >> Q >> W)) return 0;
    vector<ll> pos(Q), cnt(Q);
    for (int i = 0; i < Q; i++) cin >> pos[i] >> cnt[i];

    if (Q == 0) { for (ll i = 0; i < K; i++) cout << 0 << " \n"[i + 1 == K]; return 0; }

    // cost0[i] = cost of serving pos[0..i-1] using ONLY the trailhead (0).
    vector<ll> cost0(Q + 1, 0);
    for (int i = 0; i < Q; i++) cost0[i + 1] = cost0[i] + cnt[i] * ceilDiv(pos[i], W);

    // costMat[l][r] = cost of serving pos[l..r-1] with one stake at pos[l].
    vector<vector<ll>> costMat(Q, vector<ll>(Q + 1, 0));
    for (int l = 0; l < Q; l++) {
        ll acc = 0;
        costMat[l][l] = 0;
        for (int r = l; r < Q; r++) {
            acc += cnt[r] * ceilDiv(pos[r] - pos[l], W);
            costMat[l][r + 1] = acc;
        }
    }

    int Kc = (int)min<ll>(K, Q);  // never useful to plan more stakes than points
    vector<vector<ll>> dp(Q + 1, vector<ll>(Kc + 1, INF));
    vector<vector<int>> par(Q + 1, vector<int>(Kc + 1, -2)); // -2 unset, -1 inherit(no new stake), >=0 split at l
    for (int k = 0; k <= Kc; k++) { dp[0][k] = 0; par[0][k] = -1; }
    for (int i = 1; i <= Q; i++) { dp[i][0] = cost0[i]; par[i][0] = -1; }

    for (int k = 1; k <= Kc; k++) {
        for (int i = 1; i <= Q; i++) {
            ll best = dp[i][k - 1];
            int bestPar = -1;
            for (int l = 0; l < i; l++) {
                ll cand = dp[l][k - 1] + costMat[l][i];
                if (cand < best) { best = cand; bestPar = l; }
            }
            dp[i][k] = best;
            par[i][k] = bestPar;
        }
    }

    // reconstruct the stake positions used along the optimal path dp[Q][Kc]
    vector<ll> stakes;
    int i = Q, k = Kc;
    while (i > 0) {
        int p = par[i][k];
        if (p == -1) {
            // inherit: either "served by trailhead only up to i" (k==0 base
            // case) or "same as using k-1 stakes" -- either way no new stake
            // is attributed here except at the true k==0 base row.
            if (k == 0) break;
            k -= 1;
        } else {
            stakes.push_back(pos[p]);
            i = p;
            k -= 1;
        }
    }
    while ((int)stakes.size() < (int)K) stakes.push_back(0);

    for (size_t j = 0; j < stakes.size(); j++)
        cout << stakes[j] << (j + 1 < stakes.size() ? ' ' : '\n');
    return 0;
}
