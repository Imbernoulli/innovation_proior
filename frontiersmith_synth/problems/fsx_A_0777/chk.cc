#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker for "Weaving a Fast Gossip Network Across a Long Ridge of Villages".
// Feasibility: degree cap d per village, total wirelength (sum of |x_a-x_b|
// over chosen edges) <= W, no self-loops / duplicate edges.
// Objective F: algebraic connectivity (2nd-smallest eigenvalue) of the
// normalized Laplacian I - D^{-1/2} A D^{-1/2} of the participant's network,
// scaled to an integer. If the network is disconnected (or empty) F=0.
// Baseline B: the same quantity for the plain path graph (village i -- i+1
// for all i), built internally from the same coordinates.

static const ll SCALE = 1000000;
static const int MAXN = 260;
int par[MAXN];
int find(int x){ return par[x]==x ? x : par[x]=find(par[x]); }
void uni(int a,int b){ a=find(a); b=find(b); if(a!=b) par[a]=b; }

// Cyclic Jacobi eigenvalue algorithm for a real symmetric matrix (dense,
// deterministic, fixed sweep cap). Returns all eigenvalues (unsorted order
// not guaranteed related to input order).
vector<double> jacobiEigen(vector<vector<double>> A, int n) {
    for (int sweep = 0; sweep < 80; sweep++) {
        double off = 0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                off += A[p][q] * A[p][q];
        if (off < 1e-20) break;
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                if (fabs(A[p][q]) < 1e-16) continue;
                double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                A[p][q] = A[q][p] = 0.0;
                for (int i = 0; i < n; i++) {
                    if (i == p || i == q) continue;
                    double aip = A[i][p], aiq = A[i][q];
                    A[i][p] = A[p][i] = c * aip - s * aiq;
                    A[i][q] = A[q][i] = s * aip + c * aiq;
                }
            }
        }
    }
    vector<double> eig(n);
    for (int i = 0; i < n; i++) eig[i] = A[i][i];
    return eig;
}

// Second-smallest eigenvalue of the normalized Laplacian of the graph
// defined by `edges` on n vertices (0-indexed), given per-vertex degree.
double algebraicConnectivity(int n, const vector<pair<int,int>>& edges, const vector<int>& deg) {
    vector<double> invsqrt(n, 0.0);
    for (int i = 0; i < n; i++) invsqrt[i] = 1.0 / sqrt((double)deg[i]);
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) L[i][i] = 1.0;
    for (auto &e : edges) {
        int a = e.first, b = e.second;
        double val = -invsqrt[a] * invsqrt[b];
        L[a][b] += val; L[b][a] += val;
    }
    vector<double> eig = jacobiEigen(L, n);
    sort(eig.begin(), eig.end());
    double lambda2 = eig[1];
    if (!isfinite(lambda2) || lambda2 < 0) lambda2 = 0;
    if (lambda2 > 2.0) lambda2 = 2.0;
    return lambda2;
}

// Same index-only hierarchical-bridge topology as gen.cpp / strong.cpp, used
// HERE only to compute a calibration reference value (never to judge
// feasibility of the participant's own output).
void addCrossEdgesRef(int lo, int hi, vector<pair<int,int>>& edges) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    if (hi - lo == 2) {
        edges.push_back({lo, mid});
    } else {
        edges.push_back({lo, hi - 1});
        edges.push_back({mid - 1, mid});
    }
    addCrossEdgesRef(lo, mid, edges);
    addCrossEdgesRef(mid, hi, edges);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int k = inf.readInt();
    int d = inf.readInt();
    ll W = inf.readLong();
    vector<ll> x(n);
    for (int i = 0; i < n; i++) x[i] = inf.readLong();
    (void)k;

    int maxEdges = min(2000000, n * d);
    int m = ouf.readInt(0, maxEdges, "m");

    vector<pair<int,int>> edges;
    edges.reserve(m);
    vector<int> deg(n, 0);
    set<pair<int,int>> seen;
    ll totalCost = 0;

    for (int e = 0; e < m; e++) {
        int a = ouf.readInt(1, n, "a");
        int b = ouf.readInt(1, n, "b");
        if (a == b) quitf(_wa, "self-loop at village %d", a);
        int ua = a - 1, ub = b - 1;
        if (ua > ub) swap(ua, ub);
        pair<int,int> key = {ua, ub};
        if (seen.count(key)) quitf(_wa, "duplicate edge (%d,%d)", a, b);
        seen.insert(key);
        deg[ua]++; deg[ub]++;
        if (deg[ua] > d) quitf(_wa, "village %d degree exceeds cap d=%d", ua + 1, d);
        if (deg[ub] > d) quitf(_wa, "village %d degree exceeds cap d=%d", ub + 1, d);
        ll cost = llabs(x[ub] - x[ua]);
        totalCost += cost;
        if (totalCost > W) quitf(_wa, "wirelength budget exceeded: %lld > %lld", totalCost, W);
        edges.push_back({ua, ub});
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d edges", m);

    // connectivity check
    for (int i = 0; i < n; i++) par[i] = i;
    for (auto &e : edges) uni(e.first, e.second);
    bool connected = true;
    int root = find(0);
    for (int i = 1; i < n; i++) if (find(i) != root) { connected = false; break; }

    ll F;
    if (!connected || m == 0) {
        F = 0;
    } else {
        double lambda2 = algebraicConnectivity(n, edges, deg);
        F = (ll)llround(lambda2 * (double)SCALE);
        if (F < 0) F = 0;
    }

    // internal baseline B: plain path graph on the same coordinates
    vector<pair<int,int>> pedges;
    vector<int> pdeg(n, 0);
    for (int i = 0; i + 1 < n; i++) {
        pedges.push_back({i, i + 1});
        pdeg[i]++; pdeg[i + 1]++;
    }
    double lambda2p = algebraicConnectivity(n, pedges, pdeg);
    ll B = (ll)llround(lambda2p * (double)SCALE);
    if (B <= 0) B = 1;

    // Reference "good" construction: the hierarchical fold-bridge topology
    // (same recursion used by gen.cpp / solutions/strong.cpp), evaluated on
    // these coordinates purely for score CALIBRATION -- not a feasibility
    // check on the participant's output. Raw algebraic-connectivity values
    // for a genuine expander stay roughly constant order while the path
    // baseline B decays like Theta(1/n^2), so a plain multiplicative F/B
    // ratio would blow through any fixed cap for large n; instead the score
    // is a BOUNDED linear interpolation between B (trivial, ->0.1) and a
    // target set just beyond the reference construction's own value (->0.85),
    // leaving genuine headroom (up to 1.0) above the reference.
    vector<pair<int,int>> refEdges;
    addCrossEdgesRef(0, n, refEdges);
    vector<int> refDeg(n, 0);
    for (auto &e : refEdges) { refDeg[e.first]++; refDeg[e.second]++; }
    double lambda2ref = algebraicConnectivity(n, refEdges, refDeg);
    ll Bref = (ll)llround(lambda2ref * (double)SCALE);
    if (Bref <= B) Bref = B + 1;

    const double R0 = 0.85;       // ratio achieved by the reference construction
    const double RTRIV = 0.1;     // ratio achieved by the trivial baseline (F=B)
    double denom = (double)(Bref - B) * 0.9 / (R0 - RTRIV); // Btarget - B
    if (denom < 1.0) denom = 1.0;
    double ratio = RTRIV + 0.9 * ((double)(F - B)) / denom;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    quitp(ratio, "OK F=%lld B=%lld Bref=%lld Ratio: %.6f", F, B, Bref, ratio);
    return 0;
}
