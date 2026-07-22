// TIER: strong
// Insight: don't restrict candidate beacons to the sites you must cover --
// exploit the algebraic side-structure instead. Because q^n is capped small
// enough to brute-force (a guarantee the problem makes explicit), enumerate
// every vector in the null space of H (every one of them is automatically
// resonant, radius r+1+t_i). Run set-cover greedy over THIS candidate set. A
// null-space vector is both a valid beacon AND -- because sites were planted
// as small perturbations of null-space vectors -- likely to sit at the true
// center of an entire cluster, so one such beacon can cover what raw-site
// greedy needs many off-center picks to approximate. Any site the null-space
// candidates cannot reach (isolated noise) falls back to being its own beacon,
// guaranteeing feasibility.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int q, n, r, m, M, Kmax;
    if (scanf("%d %d %d %d %d %d", &q, &n, &r, &m, &M, &Kmax) != 6) return 0;
    vector<vector<int>> H(m, vector<int>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) scanf("%d", &H[i][j]);

    vector<vector<int>> site(M, vector<int>(n));
    vector<int> tol(M);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) scanf("%d", &site[i][j]);
        scanf("%d", &tol[i]);
    }

    long long qn = 1;
    for (int i = 0; i < n; i++) { qn *= q; if (qn > 200000) { qn = 200000; break; } }

    vector<vector<int>> ker;
    {
        vector<int> v(n, 0);
        for (long long idx = 0; idx < qn; idx++) {
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
    }
    if (ker.empty()) ker.push_back(vector<int>(n, 0));

    auto dist = [&](const vector<int>& a, const vector<int>& b) {
        int d = 0;
        for (int j = 0; j < n; j++) if (a[j] != b[j]) d++;
        return d;
    };

    const int MAXM = 2048;
    int K = (int)ker.size();
    vector<bitset<MAXM>> cover(K);
    for (int c = 0; c < K; c++) {
        int rc = r + 1; // every kernel vector is resonant by construction
        for (int i = 0; i < M; i++)
            if (dist(ker[c], site[i]) <= rc + tol[i]) cover[c].set(i);
    }

    bitset<MAXM> uncovered;
    for (int i = 0; i < M; i++) uncovered.set(i);
    vector<vector<int>> chosen;
    while (uncovered.any()) {
        int best = -1, bestCnt = -1;
        for (int c = 0; c < K; c++) {
            int cnt = (int)(cover[c] & uncovered).count();
            if (cnt > bestCnt) { bestCnt = cnt; best = c; }
        }
        if (best < 0 || bestCnt <= 0) break;
        chosen.push_back(ker[best]);
        uncovered &= ~cover[best];
    }
    // fallback: leftover (isolated/noise) sites become their own beacon
    for (int i = 0; i < M; i++) if (uncovered.test(i)) chosen.push_back(site[i]);

    printf("%d\n", (int)chosen.size());
    string buf;
    for (auto& c : chosen) {
        for (int j = 0; j < n; j++) { buf += to_string(c[j]); buf += (j + 1 < n) ? ' ' : '\n'; }
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
