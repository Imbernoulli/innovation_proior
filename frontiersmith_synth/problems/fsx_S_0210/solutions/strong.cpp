// TIER: strong
// Density-greedy init + local search (reassign / evict / drop) with seeded
// randomized multi-restart. Fully deterministic (fixed seed, iteration-bounded).
#include <bits/stdc++.h>
using namespace std;

int P, B;
vector<vector<int>> w, v;
vector<long long> cap;

// evaluate + repair-free greedy from a given priority order of (block,pump) pairs
long long buildGreedy(const vector<int>& order, const vector<pair<int,int>>& pj,
                      vector<int>& a) {
    vector<long long> rem = cap;
    a.assign(B + 1, 0);
    long long tot = 0;
    for (int idx : order) {
        int p = pj[idx].first, j = pj[idx].second;
        if (a[j] != 0) continue;
        if (rem[p] >= w[p][j]) { rem[p] -= w[p][j]; a[j] = p; tot += v[p][j]; }
    }
    return tot;
}

int main() {
    if (scanf("%d %d", &P, &B) != 2) return 0;
    cap.assign(P + 1, 0);
    for (int p = 1; p <= P; p++) scanf("%lld", &cap[p]);
    w.assign(P + 1, vector<int>(B + 1));
    v.assign(P + 1, vector<int>(B + 1));
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) scanf("%d %d", &w[p][j], &v[p][j]);

    // candidate (block,pump) pairs and their density
    vector<pair<int,int>> pj;               // (pump, block)
    vector<double> dens;
    pj.reserve((size_t)P * B);
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) {
            pj.push_back({p, j});
            dens.push_back((double)v[p][j] / (double)w[p][j]);
        }
    int M = (int)pj.size();

    vector<int> baseOrder(M);
    iota(baseOrder.begin(), baseOrder.end(), 0);
    sort(baseOrder.begin(), baseOrder.end(),
         [&](int x, int y) { return dens[x] > dens[y]; });

    mt19937 rng(987654321u);

    auto localSearch = [&](vector<int>& a) -> long long {
        vector<long long> load(P + 1, 0);
        long long tot = 0;
        for (int j = 1; j <= B; j++) if (a[j]) { load[a[j]] += w[a[j]][j]; tot += v[a[j]][j]; }
        bool improved = true;
        int guard = 0;
        while (improved && guard++ < 40) {
            improved = false;
            for (int j = 1; j <= B; j++) {
                int cur = a[j];
                long long curVal = cur ? v[cur][j] : 0;
                long long freeCur = cur ? load[cur] - w[cur][j] : 0; // load if we remove j
                // try every pump (including keeping / dropping) for best net yield
                int bestP = cur; long long bestVal = curVal;
                for (int p = 1; p <= P; p++) {
                    if (p == cur) continue;
                    long long lp = (cur == p) ? load[p] : load[p]; // p != cur here
                    if (lp + w[p][j] <= cap[p] && v[p][j] > bestVal) {
                        bestVal = v[p][j]; bestP = p;
                    }
                }
                if (bestP != cur) {
                    if (cur) load[cur] -= w[cur][j];
                    load[bestP] += w[bestP][j];
                    tot += (bestVal - curVal);
                    a[j] = bestP;
                    improved = true;
                }
                (void)freeCur;
            }
            // pairwise swap between two assigned blocks on different pumps
            for (int j1 = 1; j1 <= B; j1++) {
                int p1 = a[j1]; if (!p1) continue;
                for (int t = 0; t < 6; t++) {
                    int j2 = (int)(rng() % B) + 1;
                    int p2 = a[j2];
                    if (!p2 || p1 == p2 || j1 == j2) continue;
                    // swap pumps of j1 and j2
                    long long l1 = load[p1] - w[p1][j1] + w[p1][j2];
                    long long l2 = load[p2] - w[p2][j2] + w[p2][j1];
                    if (l1 <= cap[p1] && l2 <= cap[p2]) {
                        long long delta = (v[p1][j2] + v[p2][j1]) - (v[p1][j1] + v[p2][j2]);
                        if (delta > 0) {
                            load[p1] = l1; load[p2] = l2;
                            a[j1] = p2; a[j2] = p1;
                            tot += delta;
                            improved = true;
                            break; // a[j1] changed; stale p1 would corrupt later tries
                        }
                    }
                }
            }
        }
        return tot;
    };

    // baseline greedy + local search
    vector<int> best;
    long long bestTot = buildGreedy(baseOrder, pj, best);
    bestTot = localSearch(best);

    int restarts = 60;
    long long budget = 3000000; // rough op budget to stay well under time limit
    long long used = (long long)M * (restarts);
    if (used > budget) restarts = (int)(budget / max(1, M));
    if (restarts < 8) restarts = 8;

    for (int r = 0; r < restarts; r++) {
        vector<int> order = baseOrder;
        // perturb: randomly swap nearby entries to break density ties
        int swaps = M / 6 + 1;
        for (int s = 0; s < swaps; s++) {
            int i = (int)(rng() % M);
            int span = 5 + (int)(rng() % 15);
            int k = min(M - 1, i + (int)(rng() % span));
            swap(order[i], order[k]);
        }
        vector<int> a;
        buildGreedy(order, pj, a);
        long long tot = localSearch(a);
        if (tot > bestTot) { bestTot = tot; best = a; }
    }

    for (int j = 1; j <= B; j++) printf("%d%c", best[j], j == B ? '\n' : ' ');
    return 0;
}
