// TIER: strong
// Insight: boundaries form where fronts collide, so pick sites whose anisotropic-Voronoi
// bisector lands on the cheapest possible seam. We measure each row's cavity width,
// locate the region's narrow "waists" (local minima of width), and place one seed per
// resulting segment (at that segment's widest, most central row) -- so any collision the
// growth simulation produces is forced through a short waist, not across an open bulge.
// When the seed budget is too small to use every waist, we choose which ones to use by
// BALANCING the resulting segment lengths (a chain partition minimizing the longest reach
// any one seed must cover), not just by picking the narrowest waists -- merging two distant
// bulges under one seed costs far more (in peak fill time) than the boundary length saved
// by skipping a slightly wider waist. Orientation 0 aligns the crystal's fast axes with the
// grid's own N/S/E/W, which is always weakly optimal here since travel angles are 45-degree
// multiples.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, K;
    long long LAMBDA;
    scanf("%d %d %d", &H, &W, &K);
    scanf("%lld", &LAMBDA);
    vector<string> grid(H);
    for (int r = 0; r < H; r++) {
        char buf[100005];
        scanf("%s", buf);
        grid[r] = buf;
    }
    vector<int> width(H, 0);
    int rowLo = -1, rowHi = -1;
    for (int r = 0; r < H; r++) {
        int w = 0;
        for (int c = 0; c < W; c++) if (grid[r][c] != '#') w++;
        width[r] = w;
        if (w > 0) { if (rowLo < 0) rowLo = r; rowHi = r; }
    }

    // run-length-encode the width profile: the generator's shape is a chain of bulges and
    // waists, each with its OWN constant half-width, so width[] is piecewise constant and
    // a run is a waist iff it is strictly narrower than BOTH neighbouring runs (comparing
    // whole runs, not a fixed-size row window, correctly handles multi-row flat waists).
    struct Run { int w, lo, hi; };
    vector<Run> runs;
    for (int r = rowLo; r <= rowHi; r++) {
        if (!runs.empty() && runs.back().w == width[r]) runs.back().hi = r;
        else runs.push_back({width[r], r, r});
    }
    vector<int> waists; // representative row per waist run, row-sorted
    for (size_t i = 1; i + 1 < runs.size(); i++)
        if (runs[i].w < runs[i - 1].w && runs[i].w < runs[i + 1].w)
            waists.push_back((runs[i].lo + runs[i].hi) / 2);

    // When there are more waists than seed budget allows (K-1 cuts), which (K-1) waists to
    // use matters more for the FILL-TIME term than the narrowest-width-first idea suggests:
    // merging two far-apart bulges under one seed can blow up the peak solidification time
    // M far more than the boundary-length G saved by skipping a slightly wider waist. So we
    // choose cuts to balance the resulting segment lengths (minimize the longest segment a
    // single seed must reach across), restricted to legal waist positions -- a balanced
    // chain partition via binary search on the max segment length.
    int maxCuts = K - 1;
    int totalRows = rowHi - rowLo + 1;
    auto feasible = [&](int L, vector<int>* outCuts) -> bool {
        int pos = rowLo;
        int used = 0;
        if (outCuts) outCuts->clear();
        while (rowHi - pos + 1 > L) {
            int best = -1;
            for (int c : waists) if (c > pos && c <= pos + L) best = max(best, c);
            if (best < 0) return false;
            used++;
            if (used > maxCuts) return false;
            pos = best;
            if (outCuts) outCuts->push_back(best);
        }
        return true;
    };
    int loL = max(1, (totalRows + maxCuts) / (maxCuts + 1)), hiL = totalRows;
    while (loL < hiL) {
        int mid = (loL + hiL) / 2;
        if (feasible(mid, nullptr)) hiL = mid; else loL = mid + 1;
    }
    vector<int> splits;
    feasible(hiL, &splits);
    sort(splits.begin(), splits.end());

    vector<pair<int,int>> segs; // [lo,hi] row ranges
    int lo = rowLo;
    for (int s : splits) {
        if (s - 1 >= lo) segs.push_back({lo, s - 1});
        lo = s;
    }
    segs.push_back({lo, rowHi});

    vector<pair<int,int>> seeds;
    for (auto& seg : segs) {
        // Seed row = the segment's row-midpoint (balances reach to both of the segment's
        // ends -> keeps the peak fill time down), not the widest row: a widest-row pick can
        // sit right next to one boundary and leave a long one-sided reach to the other end.
        int midRow = (seg.first + seg.second) / 2;
        int bestRow = midRow, bestGap = INT_MAX;
        for (int r = seg.first; r <= seg.second; r++)
            if (width[r] > 0 && abs(r - midRow) < bestGap) { bestGap = abs(r - midRow); bestRow = r; }
        vector<int> cols;
        for (int c = 0; c < W; c++) if (grid[bestRow][c] != '#') cols.push_back(c);
        int col = cols[cols.size() / 2];
        seeds.push_back({bestRow, col});
    }
    if (seeds.empty()) {
        for (int r = 0; r < H && seeds.empty(); r++)
            for (int c = 0; c < W; c++)
                if (grid[r][c] != '#') { seeds.push_back({r, c}); break; }
    }
    // de-duplicate (shouldn't normally happen) and cap to K
    sort(seeds.begin(), seeds.end());
    seeds.erase(unique(seeds.begin(), seeds.end()), seeds.end());
    if ((int)seeds.size() > K) seeds.resize(K);

    printf("%d\n", (int)seeds.size());
    for (auto& s : seeds) printf("%d %d 0\n", s.first, s.second);
    return 0;
}
