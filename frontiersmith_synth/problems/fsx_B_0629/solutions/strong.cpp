// TIER: strong
// Two compounded insights, not one:
//  (1) DENOISE THE CENTER. A raw sample point is itself a corrupted copy of a
//      hidden motif, so a template placed AT a sample point must reach roughly
//      TWICE the corruption spread to cover a second corrupted copy (triangle
//      inequality). Instead, gather a local cluster of nearby uncovered points and
//      take the POSITION-WISE MAJORITY VOTE -- the Hamming-space geometric median.
//      With low corruption rates the majority vote reconstructs the hidden motif
//      almost exactly, so a template centered there needs only the corruption
//      radius itself (not the sum of two), covering far more demands per unit of
//      the convex (r+1)^2 budget. This is exactly "apportion budget by each
//      scenario's intrinsic covering difficulty" -- a tight cluster costs little,
//      a diffuse one costs more, and the checker rewards whichever solver notices.
//  (2) TARGET THE BOTTLENECK. The objective is a MINIMUM over wings, not a sum, so
//      each round (a) finds the CURRENT worst wing, (b) draws candidate seeds from
//      its own uncovered demands, builds each seed's denoised cluster center, and
//      (c) scores every candidate/radius pair by SIMULATING the coverage it adds to
//      EVERY wing, picking whichever raises the resulting global minimum fraction
//      the most per unit cost. A denoised template that lands on a motif SHARED by
//      several poor wings is recognized immediately (it raises several floors for
//      one cheap template), while one that only helps the wing it was drawn from
//      stops mattering once that wing is no longer the minimum.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int L;

static inline int hammingDist(const string &a, const string &b) {
    int d = 0;
    for (int i = 0; i < L; i++) if (a[i] != b[i]) d++;
    return d;
}

int main() {
    int K, Mmax, Rmax;
    ll Budget;
    if (scanf("%d %d %d %d %lld", &K, &Mmax, &L, &Rmax, &Budget) != 5) return 0;

    vector<vector<string>> demand(K);
    vector<int> n(K);
    static char buf[64];
    for (int k = 0; k < K; k++) {
        scanf("%d", &n[k]);
        demand[k].resize(n[k]);
        for (int i = 0; i < n[k]; i++) { scanf("%s", buf); demand[k][i] = string(buf); }
    }

    // flat uncovered pool across all wings, with owner wing per element
    vector<string> flat;
    vector<int> owner;
    for (int k = 0; k < K; k++)
        for (int i = 0; i < n[k]; i++) { flat.push_back(demand[k][i]); owner.push_back(k); }

    vector<int> uncovered(flat.size());
    iota(uncovered.begin(), uncovered.end(), 0);
    vector<ll> coveredCnt(K, 0);

    // per-wing index lists into `uncovered` positions are not kept persistently;
    // we rebuild the worst wing's candidate pool each round from `uncovered`.
    mt19937 rng(20260629u);
    vector<pair<string, int>> out;
    ll budgetLeft = Budget;
    const int CAND = 32;
    const int GATH_CAP = 400;                 // cap cluster size scanned for the vote
    const int GATH_R = min(2 * Rmax, L - 1);   // generous: catches same-motif noise,
                                                // far below typical cross-motif distance

    auto curFrac = [&](int k) { return (double)coveredCnt[k] / (double)n[k]; };

    // position-wise majority vote over a cluster of nearby uncovered points --
    // the Hamming-space geometric median, i.e. the denoised motif estimate.
    auto majorityVote = [&](const string &seed) -> string {
        int cnt[24][4] = {{0}};
        int taken = 0;
        for (int idx : uncovered) {
            if (hammingDist(flat[idx], seed) <= GATH_R) {
                const string &s = flat[idx];
                for (int p = 0; p < L; p++) cnt[p][s[p] - '0']++;
                if (++taken >= GATH_CAP) break;
            }
        }
        string center = seed;
        for (int p = 0; p < L; p++) {
            int bestSym = 0, bestC = -1;
            for (int sym = 0; sym < 4; sym++) if (cnt[p][sym] > bestC) { bestC = cnt[p][sym]; bestSym = sym; }
            center[p] = char('0' + bestSym);
        }
        return center;
    };

    while ((int)out.size() < Mmax && !uncovered.empty() && budgetLeft > 0) {
        // 1) rank wings by current fraction; focus on the BOTTOM few, not just the
        // single worst -- a candidate seed drawn from the union of the worst
        // wings' own pools is far more likely to land near a motif they SHARE
        // (the whole point of the checker's minimum objective) than a seed drawn
        // from only one wing at a time.
        vector<int> order(K);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) { return curFrac(a) < curFrac(b); });
        double worstFrac = curFrac(order[0]);
        if (worstFrac >= 0.999999) break; // every wing fully served

        int focusCount = min(K, 3);
        vector<char> isFocus(K, 0);
        for (int i = 0; i < focusCount; i++) isFocus[order[i]] = 1;

        // 2) candidate seeds: the union of the bottom wings' own uncovered
        // demands, so a template near a motif shared by several poor wings is
        // reachable in one draw; fall back to the global uncovered pool if that
        // union is empty (their remaining shortfall may already be unreachable).
        vector<int> pool;
        for (int idx : uncovered) if (isFocus[owner[idx]]) pool.push_back(idx);
        if (pool.empty()) pool = uncovered;

        int m = (int)pool.size();
        int trials = min(CAND, m);

        // per-wing scarcity weight: a wing far from full coverage counts far more
        // than one already nearly served -- this is the "apportion budget by
        // intrinsic covering difficulty" rule, applied smoothly across ALL wings
        // (not just the single current worst), so a candidate that helps several
        // still-poor wings at once is valued correctly even if none of them is
        // individually the bottom wing right now.
        vector<double> weight(K);
        for (int k = 0; k < K; k++) {
            double f = curFrac(k);
            double base = 1.0 / (f + 0.05);
            weight[k] = base * base * base * base;
        }

        string bestTmpl; int bestR = -1;
        double bestScore = -1.0;
        ll bestCost = -1;

        for (int t = 0; t < trials; t++) {
            int pick = pool[rng() % m];
            string center = majorityVote(flat[pick]);   // denoised candidate center
            // For a denoised center the newly-covered count as a function of r
            // rises steeply up to the true corruption radius of that motif and
            // then flattens -- so scoring by (weighted gain)/cost, not by capped
            // "improvement over the current bottleneck", naturally lands on the
            // radius that fully closes the motif instead of nibbling at it with
            // the cheapest radius that merely ties the second-worst wing.
            for (int r = 1; r <= Rmax; r++) {
                ll cost = (ll)(r + 1) * (r + 1);
                if (cost > budgetLeft) continue;
                vector<ll> gainPerWing(K, 0);
                ll totalGain = 0;
                for (int idx : uncovered)
                    if (hammingDist(flat[idx], center) <= r) { gainPerWing[owner[idx]]++; totalGain++; }
                if (totalGain <= 0) continue;
                double value = 0.0;
                for (int k = 0; k < K; k++) value += weight[k] * (double)gainPerWing[k] / (double)n[k];
                double score = value / (double)cost;
                if (score > bestScore) {
                    bestScore = score;
                    bestTmpl = center;
                    bestR = r;
                    bestCost = cost;
                }
            }
        }
        if (bestR < 0) break; // nothing affordable/useful found this round

        out.push_back({bestTmpl, bestR});
        budgetLeft -= bestCost;
        vector<int> remain;
        remain.reserve(uncovered.size());
        for (int idx : uncovered) {
            if (hammingDist(flat[idx], bestTmpl) <= bestR) coveredCnt[owner[idx]]++;
            else remain.push_back(idx);
        }
        uncovered.swap(remain);
    }

    printf("%d\n", (int)out.size());
    for (auto &p : out) printf("%s %d\n", p.first.c_str(), p.second);
    return 0;
}
