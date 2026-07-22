// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious first approach: single-pass geometric clustering with ONE global distance
// threshold (a small multiple of the median nearest-neighbour distance). This reliably
// forms one digester per tight local cluster of farms ("villages") -- but it NEVER
// re-evaluates whether two nearby clusters should share one regional digester: the
// threshold is calibrated to intra-village spacing, so it can never bridge to a
// coarser scale even when the instance's own economics would make that pay off.

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

// Attempts to merge the clusters containing u and v. If costAware is false, always merges.
// If costAware is true, only merges when the combined single-digester cost beats the sum
// of the two separate digester costs (using the two clusters' current best representative
// sites as the only merged-site candidates).
static bool tryMerge(DSU& d, int u, int v, long long Afix, long long Bc, bool costAware) {
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
    X.assign(M + 1, 0); Y.assign(M + 1, 0); W.assign(M + 1, 0);
    for (int i = 1; i <= M; i++) cin >> X[i] >> Y[i] >> W[i];

    DSU d; d.init(M, W);

    // median nearest-neighbour distance across all farms (O(M^2), M <= 2000)
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

    // phase 1 only: connect any pair within R1, unconditionally (village formation)
    for (int i = 1; i <= M; i++)
        for (int j = i + 1; j <= M; j++)
            if (dist(i, j) <= R1) tryMerge(d, i, j, A, Bc, false);

    // collect final clusters and print (digester coordinates = the cluster's chosen farm's location)
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
