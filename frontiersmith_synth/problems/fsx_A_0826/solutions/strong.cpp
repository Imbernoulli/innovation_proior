// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// The insight: two-phase COST-AWARE agglomeration.
// Phase 1 forms villages exactly like the greedy recipe (a small geometric threshold ->
// this part of the decision is easy and both tiers get it right).
// Phase 2 is the actual insight: treat each phase-1 village as a node, compute the true
// single-linkage (nearest member-to-member) distance between every pair of villages, and
// process pairs in ascending distance order -- but only ACCEPT a merge into one shared
// regional digester when it strictly lowers the instance's own A + Bc*cap^0.6 + pipeline
// cost formula. This directly measures each group's position relative to the concave/
// linear crossover instead of applying one fixed clustering resolution everywhere, so it
// merges the tight groups (genuinely cheaper) and leaves the loose groups split (genuinely
// cheaper), whichever the instance's own geometry and cost parameters dictate.

struct DSU {
    int n;
    vector<int> par;
    vector<double> cap, transp;
    vector<int> rep;
    vector<vector<int>> members;
    void init(int n_, vector<long long>& w) {
        n = n_;
        par.resize(n + 1); iota(par.begin(), par.end(), 0);
        cap.assign(n + 1, 0.0); transp.assign(n + 1, 0.0); rep.assign(n + 1, 0);
        members.assign(n + 1, vector<int>());
        for (int i = 1; i <= n; i++) { cap[i] = (double)w[i]; rep[i] = i; members[i] = {i}; }
    }
    int find(int x) { return par[x] == x ? x : par[x] = find(par[x]); }
};

static vector<long long> X, Y, W;

static double dist(int i, int j) {
    double dx = (double)(X[i] - X[j]), dy = (double)(Y[i] - Y[j]);
    return sqrt(dx * dx + dy * dy);
}

static bool tryMerge(DSU& d, int u, int v, long long Afix, long long Bc, double L, bool costAware) {
    int ru = d.find(u), rv = d.find(v);
    if (ru == rv) return false;
    if (d.members[ru].size() > d.members[rv].size()) swap(ru, rv);
    double capA = d.cap[ru], capB = d.cap[rv];
    double before = 2.0 * Afix + Bc * pow(capA, 0.6) + Bc * pow(capB, 0.6) + d.transp[ru] + d.transp[rv];
    int cand[2] = { d.rep[ru], d.rep[rv] };
    double bestT = 1e300; int bestR = -1;
    for (int c = 0; c < 2; c++) {
        double t = 0.0; int rr = cand[c];
        for (int m : d.members[ru]) t += (double)W[m] * dist(m, rr);
        for (int m : d.members[rv]) t += (double)W[m] * dist(m, rr);
        t *= L;   // pipeline loss is L * w * distance -- MUST match the checker's objective
        if (t < bestT) { bestT = t; bestR = rr; }
    }
    double capMerged = capA + capB;
    double after = Afix + Bc * pow(capMerged, 0.6) + bestT;
    if (costAware && after >= before - 1e-6) return false;
    d.par[ru] = rv;
    for (int m : d.members[ru]) d.members[rv].push_back(m);
    d.members[ru].clear();
    d.cap[rv] = capMerged;
    d.transp[rv] = bestT;
    d.rep[rv] = bestR;
    return true;
}

int main() {
    int M; long long A, Bc, Lm;
    if (!(cin >> M >> A >> Bc >> Lm)) return 0;
    double L = Lm / 1000.0;
    X.assign(M + 1, 0); Y.assign(M + 1, 0); W.assign(M + 1, 0);
    for (int i = 1; i <= M; i++) cin >> X[i] >> Y[i] >> W[i];

    DSU d; d.init(M, W);

    // median nearest-neighbour distance (O(M^2), M <= 2000)
    vector<double> nn(M + 1, 1e18);
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= M; j++) {
            if (i == j) continue;
            double dd = dist(i, j);
            if (dd < nn[i]) nn[i] = dd;
        }
    }
    vector<double> sorted_nn(nn.begin() + 1, nn.end());
    sort(sorted_nn.begin(), sorted_nn.end());
    double median = sorted_nn[sorted_nn.size() / 2];
    double R1 = max(1.0, median * 4.0);

    // ---- phase 1: village formation (same recipe as greedy) ----
    for (int i = 1; i <= M; i++)
        for (int j = i + 1; j <= M; j++)
            if (dist(i, j) <= R1) tryMerge(d, i, j, A, Bc, L, false);

    // ---- extract phase-1 cells ----
    vector<int> cellRoot;
    {
        vector<char> seen(M + 1, 0);
        for (int i = 1; i <= M; i++) {
            int r = d.find(i);
            if (!seen[r]) { seen[r] = 1; cellRoot.push_back(r); }
        }
    }
    int C = (int)cellRoot.size();

    // ---- phase 2: true single-linkage inter-cell distance, cost-aware merge ----
    // C is small (<= a few hundred) so O(C^2 * avg member^2) is cheap; cap total work with
    // a simple size guard just in case a pathological instance produces many tiny cells.
    vector<tuple<double,int,int>> pairs;
    long long workBudget = 8000000;
    for (int a = 0; a < C && workBudget > 0; a++) {
        int ra = cellRoot[a];
        for (int b = a + 1; b < C && workBudget > 0; b++) {
            int rb = cellRoot[b];
            double best = 1e300;
            long long cost = (long long)d.members[ra].size() * (long long)d.members[rb].size();
            workBudget -= cost;
            for (int mi : d.members[ra])
                for (int mj : d.members[rb]) {
                    double dd = dist(mi, mj);
                    if (dd < best) best = dd;
                }
            pairs.push_back({best, ra, rb});
        }
    }
    sort(pairs.begin(), pairs.end(), [](const tuple<double,int,int>& p, const tuple<double,int,int>& q) {
        return get<0>(p) < get<0>(q);
    });
    for (auto& pr : pairs) {
        int ra = get<1>(pr), rb = get<2>(pr);
        tryMerge(d, ra, rb, A, Bc, L, true);
    }

    // ---- collect final clusters and print (digester coordinates = chosen farm's location) ----
    vector<int> rootOf(M + 1);
    map<int, int> rootToId;
    vector<int> siteOfId;
    for (int i = 1; i <= M; i++) {
        int r = d.find(i);
        rootOf[i] = r;
        if (!rootToId.count(r)) {
            rootToId[r] = (int)siteOfId.size() + 1;
            siteOfId.push_back(d.rep[r]);
        }
    }
    int K = (int)siteOfId.size();
    cout << K << "\n";
    for (int j = 0; j < K; j++) cout << X[siteOfId[j]] << " " << Y[siteOfId[j]] << "\n";
    for (int i = 1; i <= M; i++) cout << rootToId[rootOf[i]] << (i < M ? ' ' : '\n');
    return 0;
}
