#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Resonant Beacon Cover -- generator.
//
// Emits: q n r m M Kmax
//        m lines of H (n ints in [0,q-1])
//        M lines of "n site symbols" + "tolerance bit"
//
// Sites are built from two pools:
//   PLANTED clusters: pick a random null-space vector g of H (H*g=0 mod q, so
//     any beacon placed exactly at g is resonant and reaches radius r+1+t).
//     Perturb g by a random Hamming distance in [0, r+1] to build each site in
//     the cluster, so the single beacon g covers the *entire* cluster at the
//     boosted radius. A candidate drawn from the cluster itself (not the true
//     center g) generally only reaches a much smaller sub-ball at the
//     UNBOOSTED radius r -- this is the planted trap: greedy search that only
//     tries site points as centers cannot find g and needs many more beacons
//     per cluster than the algebraic construction does.
//   NOISE sites: uniformly random strings, unrelated to any kernel vector,
//     forcing every strategy to pay close to one beacon per noise site.
// Later test ids grow q^n, M, and the planted/noise mix (including one
// NEEDLE test where a single kernel vector explains most of the instance).
// ---------------------------------------------------------------------------

static int Q, N, R, M2;
static long long QN;

static long long qpow(int q, int n) {
    long long v = 1;
    for (int i = 0; i < n; i++) v *= q;
    return v;
}

// Enumerate every vector v in Z_q^n with H*v == 0 (mod q).
static vector<vector<int>> nullSpace(const vector<vector<int>>& H, int m, int n, int q, long long total) {
    vector<vector<int>> ker;
    vector<int> v(n, 0);
    for (long long idx = 0; idx < total; idx++) {
        long long t = idx;
        for (int i = 0; i < n; i++) { v[i] = (int)(t % q); t /= q; }
        bool ok = true;
        for (int row = 0; row < m && ok; row++) {
            long long s = 0;
            for (int j = 0; j < n; j++) s += (long long)H[row][j] * v[j];
            if (s % q != 0) ok = false;
        }
        if (ok) ker.push_back(v);
    }
    return ker;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int id = atoi(argv[1]);
    if (id < 1 || id > 10) id = 1;

    // per-test parameters: q, n, r, m, M, numClusters, fracPlanted(x100), dominant(x100, 0=none)
    struct P { int q, n, r, m, M, clusters, fracPlanted, dominant; };
    static const P tbl[11] = {
        {0,0,0,0,0,0,0,0},
        {2, 6, 1, 2, 12,   2, 85,  0},   // 1: tiny, example scale
        {2, 8, 1, 3, 40,   3, 75,  0},   // 2: small
        {3, 6, 1, 2, 50,   3, 55,  0},   // 3: alphabet variety q=3, more noise
        {5, 6, 1, 2, 80,   3, 70,  0},   // 4: alphabet variety q=5
        {2, 10,1, 4, 150,  5, 65,  0},   // 5
        {2, 11,1, 5, 300,  6, 75,  0},   // 6: bigger, more clusters
        {3, 8, 1, 3, 300,  5, 50,  0},   // 7: TRAP, mostly noise, q=3
        {2, 12,1, 5, 600,  7, 80,  0},   // 8: TRAP planted, large
        {2, 12,1, 6, 900,  7, 80, 55},   // 9: NEEDLE, one dominant cluster
        {2, 14,1, 6, 2000, 9, 75,  0},   // 10: largest, fills the envelope
    };
    P p = tbl[id];
    Q = p.q; N = p.n; R = p.r; M2 = p.M;
    QN = qpow(Q, N);

    int m = p.m;
    vector<vector<int>> H(m, vector<int>(N));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < N; j++)
            H[i][j] = rnd.next(0, Q - 1);

    vector<vector<int>> ker = nullSpace(H, m, N, Q, QN);
    if (ker.empty()) ker.push_back(vector<int>(N, 0)); // all-zero always qualifies in theory

    int Kmax = M2;
    long long total = M2;
    int numClusters = max(1, p.clusters);

    // cluster sizes
    vector<long long> csize(numClusters, 0);
    long long plantedTotal = (long long)llround(total * (p.fracPlanted / 100.0));
    if (p.dominant > 0 && numClusters >= 1) {
        long long dom = (long long)llround(total * (p.dominant / 100.0));
        csize[0] = dom;
        long long rest = plantedTotal - dom;
        if (rest < 0) rest = 0;
        for (int i = 1; i < numClusters; i++) csize[i] = rest / (numClusters - 1);
    } else {
        for (int i = 0; i < numClusters; i++) csize[i] = plantedTotal / numClusters;
    }
    long long plantedSum = 0;
    for (auto c : csize) plantedSum += c;
    long long noiseCount = total - plantedSum;
    if (noiseCount < 0) noiseCount = 0;

    // build target list
    vector<vector<int>> sites;
    vector<int> tol;
    sites.reserve(total);
    tol.reserve(total);

    for (int ci = 0; ci < numClusters; ci++) {
        if (csize[ci] <= 0) continue;
        const vector<int>& g = ker[rnd.next(0, (int)ker.size() - 1)];
        for (long long s = 0; s < csize[ci]; s++) {
            vector<int> x = g;
            // ~28% of a cluster's members are placed just OUTSIDE even the
            // boosted radius (still near the cluster, but neither a naive
            // nor an algebraic beacon reaches them for free) -- this caps how
            // much of a cluster a single resonant beacon can swallow, so the
            // score cannot saturate just by finding one center per cluster.
            bool outlier = rnd.next(0, 99) < 28;
            int dmax = outlier ? min(N, R + 4) : (R + 1);
            int dmin = outlier ? (R + 2) : 0;
            if (dmin > dmax) dmin = dmax;
            int d = rnd.next(dmin, dmax);
            // choose d distinct positions to perturb
            vector<int> pos(N);
            for (int i = 0; i < N; i++) pos[i] = i;
            for (int i = 0; i < d; i++) {
                int j = rnd.next(i, N - 1);
                swap(pos[i], pos[j]);
                int p0 = pos[i];
                int old = x[p0];
                int nv = rnd.next(0, Q - 1);
                if (Q > 1 && nv == old) nv = (nv + 1) % Q;
                x[p0] = nv;
            }
            sites.push_back(x);
            tol.push_back(rnd.next(0, 1));
        }
    }
    for (long long s = 0; s < noiseCount; s++) {
        vector<int> x(N);
        for (int i = 0; i < N; i++) x[i] = rnd.next(0, Q - 1);
        sites.push_back(x);
        tol.push_back(rnd.next(0, 1));
    }

    // deterministic shuffle so cluster membership order gives no positional hint
    for (int i = (int)sites.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(sites[i], sites[j]);
        swap(tol[i], tol[j]);
    }

    printf("%d %d %d %d %d %d\n", Q, N, R, m, (int)sites.size(), Kmax);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N; j++) printf("%d%c", H[i][j], j + 1 < N ? ' ' : '\n');
    }
    string buf;
    for (size_t i = 0; i < sites.size(); i++) {
        for (int j = 0; j < N; j++) { buf += to_string(sites[i][j]); buf += ' '; }
        buf += to_string(tol[i]);
        buf += '\n';
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
