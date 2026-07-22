// TIER: strong
// Insight: don't guess a tile shape -- READ it off the entropy map. A row (or column) is a
// true structural boundary iff it differs from its neighbor somewhere. Cutting exactly there
// makes every tile perfectly homogeneous: every low-entropy patch gets isolated into its own
// small tile (forced level 0 costs almost nothing extra), and every background tile -- however
// large -- can compress all the way to the max level. That shrinks the resident footprint of
// everything the trace merely sweeps through, freeing byte-capacity that lets whatever the
// trace actually revisits stay cache-resident under LRU -- the cross-region externality no
// per-tile decision can see.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, K;
    if (scanf("%d %d %d", &R, &C, &K) != 3) return 0;
    vector<long long> levelBytes(K);
    for (int k = 0; k < K; k++) scanf("%lld", &levelBytes[k]);
    long long cacheBytes; scanf("%lld", &cacheBytes);

    vector<vector<int>> ent(R, vector<int>(C));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            scanf("%d", &ent[i][j]);
    // trace is not read: the entropy-aligned layout helps regardless of what the trace does.

    vector<int> rb = {0};
    for (int i = 1; i < R; i++) {
        bool diff = false;
        for (int j = 0; j < C; j++) if (ent[i][j] != ent[i - 1][j]) { diff = true; break; }
        if (diff) rb.push_back(i);
    }
    rb.push_back(R);

    vector<int> cb = {0};
    for (int j = 1; j < C; j++) {
        bool diff = false;
        for (int i = 0; i < R; i++) if (ent[i][j] != ent[i][j - 1]) { diff = true; break; }
        if (diff) cb.push_back(j);
    }
    cb.push_back(C);

    // A homogeneous span can be arbitrarily long (e.g. background between two far-apart
    // patches) -- left as ONE tile it becomes an all-or-nothing giant that either always
    // fits or always faults for its full size. Chop any span above `cap` into near-equal
    // sub-bands (entropy inside stays uniform, so this never changes what any sub-tile is
    // allowed to compress to) -- this lets partial residency track the trace's own locality
    // instead of forcing one monolithic reload.
    const int cap = 16;
    auto refine = [&](const vector<int>& b) {
        vector<int> out;
        out.push_back(b[0]);
        for (size_t k = 1; k < b.size(); k++) {
            int lo = b[k - 1], hi = b[k];
            int parts = max(1, (hi - lo + cap - 1) / cap);
            for (int p = 1; p <= parts; p++)
                out.push_back(lo + (int)((long long)(hi - lo) * p / parts));
        }
        return out;
    };
    rb = refine(rb);
    cb = refine(cb);

    // Safety valve: if the entropy map is pathologically noisy this could over-fragment;
    // cap total tiles by coarsening (kept generous -- our inputs never approach this).
    while ((long long)(rb.size() - 1) * (cb.size() - 1) > 300000LL && rb.size() > 2) {
        vector<int> nrb;
        for (size_t k = 0; k < rb.size(); k += 2) nrb.push_back(rb[k]);
        if (nrb.back() != rb.back()) nrb.push_back(rb.back());
        rb = nrb;
    }

    int rB = (int)rb.size() - 1, cB = (int)cb.size() - 1;
    vector<int> rowBand(R), colBand(C);
    { int r = 0; for (int i = 0; i < R; i++) { while (i >= rb[r + 1]) r++; rowBand[i] = r; } }
    { int c = 0; for (int j = 0; j < C; j++) { while (j >= cb[c + 1]) c++; colBand[j] = c; } }

    vector<vector<int>> capMax(rB, vector<int>(cB, K - 1));
    for (int i = 0; i < R; i++) {
        int r = rowBand[i];
        for (int j = 0; j < C; j++) {
            int c = colBand[j];
            int& cm = capMax[r][c];
            if (ent[i][j] < cm) cm = ent[i][j];
        }
    }

    printf("%d\n", rB);
    for (int x : rb) printf("%d ", x);
    printf("\n%d\n", cB);
    for (int x : cb) printf("%d ", x);
    printf("\n");
    for (int r = 0; r < rB; r++) {
        for (int c = 0; c < cB; c++) printf("%d ", capMax[r][c]);
        printf("\n");
    }
    return 0;
}
