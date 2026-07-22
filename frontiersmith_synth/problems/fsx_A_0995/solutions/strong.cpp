// TIER: strong
// The insight: the leverage is in the discrete recoloring freedom, which
// reshapes the transition graph, not in touring it better. We maintain one
// GLOBAL "committed tier" (LIGHT/DARK) as we sweep layers 1..L in order.
// For every region whose tolerance class offers a color in the committed
// tier, we RECOLOR it into that tier regardless of which color is listed
// first -- this keeps the whole run on the cheap side of the purge matrix.
// A region whose class is locked to the opposite tier (an "anchor") forces
// an unavoidable pivot: we flush every same-tier region in the layer first,
// then the anchor(s) last, so the expensive transition happens AT MOST ONCE
// per forced pivot instead of being repeated by accident. After a pivot the
// committed tier updates for subsequent layers. Within a tier-block we sort
// by concrete color id, which for our matrix shape also minimizes the
// residual small same-tier cost (a cheap secondary "touring" win).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int L, C;
    scanf("%d %d", &L, &C);
    vector<vector<ll>> P(C, vector<ll>(C));
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) scanf("%lld", &P[i][j]);
    int half = C / 2; // colors < half : LIGHT(0), colors >= half : DARK(1)

    int committed = 0; // 0 = LIGHT, 1 = DARK

    for (int l = 0; l < L; l++) {
        int R; scanf("%d", &R);
        vector<vector<int>> cls(R);
        for (int r = 0; r < R; r++) {
            int k; scanf("%d", &k);
            cls[r].resize(k);
            for (int i = 0; i < k; i++) scanf("%d", &cls[r][i]);
        }

        vector<int> color(R);
        vector<int> tierOf(R); // tier actually used for this region
        for (int r = 0; r < R; r++) {
            // does this region offer a color matching the committed tier?
            int best = -1;
            for (int c : cls[r]) {
                bool isCommitted = (committed == 0) ? (c < half) : (c >= half);
                if (isCommitted) { if (best == -1 || c < best) best = c; }
            }
            if (best != -1) { color[r] = best; tierOf[r] = committed; }
            else {
                // forced to the opposite tier (anchor): pick the smallest
                // color in whichever tier its class actually offers.
                int lo = -1;
                for (int c : cls[r]) if (lo == -1 || c < lo) lo = c;
                color[r] = lo;
                tierOf[r] = (lo < half) ? 0 : 1;
            }
        }

        // order: committed-tier block first (sorted by color), then the
        // forced-opposite-tier block last (sorted by color) -- at most one
        // pivot inside this layer, placed as late as possible.
        vector<int> order(R);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (tierOf[a] != tierOf[b]) return tierOf[a] == committed; // committed block first
            if (color[a] != color[b]) return color[a] < color[b];
            return a < b;
        });

        bool anyOpposite = false;
        for (int r = 0; r < R; r++) if (tierOf[r] != committed) anyOpposite = true;
        if (anyOpposite) {
            // the layer ends in the opposite tier -> that becomes committed
            committed = 1 - committed;
        }

        for (int i = 0; i < R; i++) printf("%d%c", order[i], i + 1 == R ? '\n' : ' ');
        for (int i = 0; i < R; i++) printf("%d%c", color[order[i]], i + 1 == R ? '\n' : ' ');
    }
    return 0;
}
