// TIER: strong
// Multi-restart single-flip local search to convergence. From all-WEST plus many
// deterministic random restarts; full sweeps until no improving flip; keep best.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> W;
vector<int> R;
vector<int> cvar, csign, coff;   // CSR clause -> literals
vector<int> ivar_c, ivar_s, ioff; // CSR var -> (clause, sign)

vector<int> cnt, curr;
ll curW;

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    W.resize(m); R.resize(m);
    vector<vector<pair<int,int>>> tmp(m);
    long long totalLits = 0;
    for (int j = 0; j < m; j++) {
        int k;
        scanf("%lld %d %d", &W[j], &R[j], &k);
        tmp[j].reserve(k);
        for (int t = 0; t < k; t++) {
            int lit; scanf("%d", &lit);
            int v = abs(lit), s = lit > 0 ? 1 : -1;
            tmp[j].push_back({v, s});
        }
        totalLits += k;
    }
    coff.assign(m + 1, 0);
    for (int j = 0; j < m; j++) coff[j + 1] = coff[j] + (int)tmp[j].size();
    cvar.resize(totalLits); csign.resize(totalLits);
    vector<int> deg(n + 1, 0);
    for (int j = 0; j < m; j++) for (auto& p : tmp[j]) deg[p.first]++;
    ioff.assign(n + 2, 0);
    for (int v = 1; v <= n; v++) ioff[v + 1] = ioff[v] + deg[v];
    ivar_c.resize(totalLits); ivar_s.resize(totalLits);
    vector<int> pos(n + 1);
    for (int v = 1; v <= n; v++) pos[v] = ioff[v];
    for (int j = 0; j < m; j++) {
        int base = coff[j], idx = 0;
        for (auto& p : tmp[j]) {
            cvar[base + idx] = p.first;
            csign[base + idx] = p.second;
            int pp = pos[p.first]++;
            ivar_c[pp] = j; ivar_s[pp] = p.second;
            idx++;
        }
    }

    vector<int> x(n + 1, 0);
    cnt.assign(m, 0); curr.assign(m, 0);

    auto recompute = [&]() {
        curW = 0;
        for (int j = 0; j < m; j++) {
            int c = 0;
            for (int t = coff[j]; t < coff[j + 1]; t++) {
                int v = cvar[t], s = csign[t];
                bool sat = (s > 0) ? (x[v] == 1) : (x[v] == 0);
                if (sat) c++;
            }
            cnt[j] = c;
            curr[j] = (c % 3 == R[j]) ? 1 : 0;
            if (curr[j]) curW += W[j];
        }
    };
    auto gainFlip = [&](int v) -> ll {
        ll g = 0;
        for (int p = ioff[v]; p < ioff[v + 1]; p++) {
            int j = ivar_c[p], s = ivar_s[p];
            bool met = (s > 0) ? (x[v] == 1) : (x[v] == 0);
            int nc = cnt[j] + (met ? -1 : +1);
            int newPay = (nc % 3 == R[j]) ? 1 : 0;
            g += (ll)(newPay - curr[j]) * W[j];
        }
        return g;
    };
    auto doFlip = [&](int v) {
        for (int p = ioff[v]; p < ioff[v + 1]; p++) {
            int j = ivar_c[p], s = ivar_s[p];
            bool met = (s > 0) ? (x[v] == 1) : (x[v] == 0);
            int old = curr[j];
            cnt[j] += (met ? -1 : +1);
            int nw = (cnt[j] % 3 == R[j]) ? 1 : 0;
            curr[j] = nw;
            curW += (ll)(nw - old) * W[j];
        }
        x[v] ^= 1;
    };

    // local search to convergence from current x; returns final weight
    const int SWEEP_CAP = 60;
    auto localSearch = [&]() -> ll {
        recompute();
        for (int sweep = 0; sweep < SWEEP_CAP; sweep++) {
            bool improved = false;
            for (int v = 1; v <= n; v++) {
                if (gainFlip(v) > 0) { doFlip(v); improved = true; }
            }
            if (!improved) break;
        }
        return curW;
    };

    vector<int> best(n + 1, 0);
    ll bestW = -1;

    // budget of restarts based on problem size (deterministic count)
    long long unit = max<long long>(1, totalLits);
    int restarts = (int)max<long long>(4, min<long long>(80, 120000000LL / (SWEEP_CAP * unit)));

    mt19937 rng(0xC0FFEEu); // fixed seed -> deterministic

    for (int it = 0; it < restarts; it++) {
        if (it == 0) {
            for (int v = 1; v <= n; v++) x[v] = 0;          // all-WEST start
        } else if (it == 1) {
            for (int v = 1; v <= n; v++) x[v] = 1;          // all-EAST start
        } else {
            for (int v = 1; v <= n; v++) x[v] = (int)(rng() & 1u);
        }
        ll wv = localSearch();
        if (wv > bestW) { bestW = wv; for (int v = 1; v <= n; v++) best[v] = x[v]; }
    }

    for (int v = 1; v <= n; v++) printf("%d%c", best[v], v == n ? '\n' : ' ');
    return 0;
}
