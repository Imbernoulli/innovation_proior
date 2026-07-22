// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: the objective is a global operator norm in the Laplacian eigenbasis,
// not a sum of local edge affinities. A seating is good iff its tempo vector
// avoids the graph's SLOW eigenvector subspace (the Fiedler-like modes that
// carry a bottleneck cut's current). Phase 1 approximates the Fiedler vector
// by shifted power iteration (deflating the constant mode every step), orders
// seats along it, then INTERLEAVES the sorted frequency multiset from both
// extremes (min, max, 2nd-min, 2nd-max, ...) along that order. Because the
// multiset is symmetric (every +v is paired with a -v), each consecutive pair
// in the interleave sums to exactly 0, so every prefix of the Fiedler order
// (which approximates a sweep cut / the bottleneck cut) carries a near-zero
// net charge -- deliberately seating dissimilar tempos next to each other on
// purpose, instead of smoothing locally. Phase 2 is a bounded exact local
// refinement: repeatedly resolve the REAL balance equations, find the single
// worst-loaded arm, and try swapping one of its two seats' metronomes with
// whichever other seat's metronome is closest to erasing that arm's exact
// imbalance -- verified against the true objective before accepting.
int n, m;
vector<vector<int>> adj;
vector<pair<int,int>> edges;

vector<double> solvePotentials(const vector<ll> &omega) {
    int sz = n - 1;
    vector<vector<double>> A(sz, vector<double>(sz, 0.0));
    vector<double> rhs(sz, 0.0);
    for (int i = 2; i <= n; i++) {
        int ri = i - 2;
        rhs[ri] = (double)omega[i];
        A[ri][ri] += (double)adj[i].size();
        for (int nb : adj[i]) {
            if (nb == 1) continue;
            A[ri][nb - 2] -= 1.0;
        }
    }
    for (int col = 0; col < sz; col++) {
        int piv = col;
        double best = fabs(A[col][col]);
        for (int r = col + 1; r < sz; r++)
            if (fabs(A[r][col]) > best) { best = fabs(A[r][col]); piv = r; }
        if (piv != col) { swap(A[piv], A[col]); swap(rhs[piv], rhs[col]); }
        double d = A[col][col];
        if (fabs(d) < 1e-12) d = (d >= 0 ? 1e-12 : -1e-12);
        for (int r = col + 1; r < sz; r++) {
            double factor = A[r][col] / d;
            if (factor == 0.0) continue;
            for (int c = col; c < sz; c++) A[r][c] -= factor * A[col][c];
            rhs[r] -= factor * rhs[col];
        }
    }
    vector<double> sol(sz, 0.0);
    for (int r = sz - 1; r >= 0; r--) {
        double s = rhs[r];
        for (int c = r + 1; c < sz; c++) s -= A[r][c] * sol[c];
        double d = A[r][r];
        if (fabs(d) < 1e-12) d = (d >= 0 ? 1e-12 : -1e-12);
        sol[r] = s / d;
    }
    vector<double> x(n + 1, 0.0);
    for (int i = 2; i <= n; i++) x[i] = sol[i - 2];
    return x;
}

double worstEdge(const vector<double> &x, int &wu, int &wv) {
    double F = -1;
    for (auto &e : edges) {
        double f = fabs(x[e.first] - x[e.second]);
        if (f > F) { F = f; wu = e.first; wv = e.second; }
    }
    return F;
}

int main() {
    cin >> n >> m;
    adj.assign(n + 1, {});
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        edges[i] = {u, v};
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<ll> f(n + 1);
    for (int i = 1; i <= n; i++) cin >> f[i];

    // ---- phase 1: approximate Fiedler vector via shifted power iteration ----
    int maxDeg = 1;
    for (int i = 1; i <= n; i++) maxDeg = max(maxDeg, (int)adj[i].size());
    double c = 2.0 * maxDeg + 1.0;

    vector<double> v(n + 1);
    for (int i = 1; i <= n; i++) {
        unsigned h = (unsigned)i * 2654435761u;
        v[i] = (double)(h % 100000) / 100000.0 - 0.5;
    }
    // A long path (or any graph with a tiny spectral gap between its two
    // slowest modes) converges very slowly under plain power iteration --
    // 300 steps is nowhere near enough once n gets into the hundreds, so
    // scale the iteration budget with n (still trivially cheap: O(n+m) per
    // step).
    int piIters = min(60000, 500 * n);
    vector<double> w(n + 1);
    for (int iter = 0; iter < piIters; iter++) {
        for (int i = 1; i <= n; i++) {
            double lv = (double)adj[i].size() * v[i];
            for (int nb : adj[i]) lv -= v[nb];
            w[i] = c * v[i] - lv;
        }
        double mean = 0;
        for (int i = 1; i <= n; i++) mean += w[i];
        mean /= n;
        double norm = 0;
        for (int i = 1; i <= n; i++) { w[i] -= mean; norm += w[i] * w[i]; }
        norm = sqrt(max(norm, 1e-18));
        for (int i = 1; i <= n; i++) v[i] = w[i] / norm;
    }

    vector<int> nodeOrder(n);
    for (int i = 0; i < n; i++) nodeOrder[i] = i + 1;
    sort(nodeOrder.begin(), nodeOrder.end(), [&](int a, int b) { return v[a] < v[b]; });

    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b) { return f[a] < f[b]; });

    vector<int> zig(n);
    int lo = 0, hi = n - 1;
    for (int k = 0; k < n; k++) {
        if (k % 2 == 0) zig[k] = idx[lo++];
        else zig[k] = idx[hi--];
    }

    vector<int> p(n + 1);
    for (int k = 0; k < n; k++) p[nodeOrder[k]] = zig[k];

    vector<ll> omega(n + 1);
    for (int i = 1; i <= n; i++) omega[i] = f[p[i]];

    // ---- phase 2: bounded exact local refinement on the true objective ----
    vector<double> x = solvePotentials(omega);
    int wu, wv;
    double curF = worstEdge(x, wu, wv);
    int maxIters = min(6 * n, 700);
    for (int it = 0; it < maxIters; it++) {
        double target = (double)(omega[wu] + omega[wv]) / 2.0;
        vector<int> cand;
        for (int w2 = 1; w2 <= n; w2++) if (w2 != wu && w2 != wv) cand.push_back(w2);
        sort(cand.begin(), cand.end(), [&](int a, int b) {
            return fabs((double)omega[a] - target) < fabs((double)omega[b] - target);
        });
        int tryCount = min((int)cand.size(), 3);
        double bestF = curF; int bestA = -1, bestW = -1; vector<double> bestX;
        for (int ci = 0; ci < tryCount; ci++) {
            int w2 = cand[ci];
            for (int a : {wu, wv}) {
                swap(omega[a], omega[w2]);
                vector<double> x2 = solvePotentials(omega);
                int du, dv;
                double F2 = worstEdge(x2, du, dv);
                if (F2 < bestF - 1e-9) { bestF = F2; bestA = a; bestW = w2; bestX = x2; }
                swap(omega[a], omega[w2]); // revert trial
            }
        }
        if (bestA < 0) break; // no improving swap found -> local optimum for this pass
        swap(omega[bestA], omega[bestW]);
        swap(p[bestA], p[bestW]);
        curF = bestF;
        x = bestX;
        worstEdge(x, wu, wv);
    }

    for (int i = 1; i <= n; i++) cout << p[i] << (i < n ? ' ' : '\n');
    return 0;
}
