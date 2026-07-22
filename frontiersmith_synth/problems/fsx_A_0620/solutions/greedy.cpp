// TIER: greedy
// The obvious approach: weighted set-cover, restricted to candidates drawn
// from the site list itself (you cannot enumerate all q^n possible beacons in
// general, so the natural fallback is "use one of the points I actually need
// to cover"). Repeatedly add the site whose ball (radius r+t_i; almost never
// resonant since H is essentially never satisfied by an arbitrary site) covers
// the most still-uncovered sites. This never looks at the null space of H, so
// it cannot find the well-centered, radius-boosted beacons that unlock whole
// planted clusters in one shot.
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

    auto resonant = [&](const vector<int>& c) {
        for (int row = 0; row < m; row++) {
            long long acc = 0;
            for (int j = 0; j < n; j++) acc += (long long)H[row][j] * c[j];
            if (acc % q != 0) return false;
        }
        return true;
    };
    auto dist = [&](const vector<int>& a, const vector<int>& b) {
        int d = 0;
        for (int j = 0; j < n; j++) if (a[j] != b[j]) d++;
        return d;
    };

    const int MAXM = 2048;
    vector<bitset<MAXM>> cover(M);
    for (int c = 0; c < M; c++) {
        int rc = r + (resonant(site[c]) ? 1 : 0);
        for (int i = 0; i < M; i++)
            if (dist(site[c], site[i]) <= rc + tol[i]) cover[c].set(i);
    }

    bitset<MAXM> uncovered;
    for (int i = 0; i < M; i++) uncovered.set(i);
    vector<int> chosen;
    while (uncovered.any()) {
        int best = -1, bestCnt = -1;
        for (int c = 0; c < M; c++) {
            int cnt = (int)(cover[c] & uncovered).count();
            if (cnt > bestCnt) { bestCnt = cnt; best = c; }
        }
        if (best < 0 || bestCnt <= 0) break;
        chosen.push_back(best);
        uncovered &= ~cover[best];
    }
    // fallback: any leftover site becomes its own beacon (should be rare/empty)
    for (int i = 0; i < M; i++) if (uncovered.test(i)) chosen.push_back(i);

    printf("%d\n", (int)chosen.size());
    string buf;
    for (int c : chosen) {
        for (int j = 0; j < n; j++) { buf += to_string(site[c][j]); buf += (j + 1 < n) ? ' ' : '\n'; }
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
