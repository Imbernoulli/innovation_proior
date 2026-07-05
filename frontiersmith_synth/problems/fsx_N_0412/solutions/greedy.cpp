// TIER: greedy
// Single left-to-right pass of best-response single-bit flips, starting all-WEST.
// One sweep only, no re-sweep, no restarts -> beats trivial but leaves gains on table.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> W;
vector<int> R;
// per clause: list of (var, sign)
vector<int> cvar, csign, coff; // CSR
vector<int> cnt;               // current met-signal count per clause
vector<int> curr;              // 1 if clause currently pays

// incidence: for each var, list of (clause, sign)
vector<int> ivar_c, ivar_s, ioff;

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
    // build CSR for clauses
    coff.assign(m + 1, 0);
    for (int j = 0; j < m; j++) coff[j + 1] = coff[j] + (int)tmp[j].size();
    cvar.resize(totalLits); csign.resize(totalLits);
    // build incidence counts
    vector<int> deg(n + 1, 0);
    for (int j = 0; j < m; j++) for (auto& p : tmp[j]) deg[p.first]++;
    ioff.assign(n + 2, 0);
    for (int v = 1; v <= n; v++) ioff[v + 1] = ioff[v] + deg[v];
    ivar_c.resize(totalLits); ivar_s.resize(totalLits);
    vector<int> pos(n + 1);
    for (int v = 1; v <= n; v++) pos[v] = ioff[v];
    for (int j = 0; j < m; j++) {
        int base = coff[j];
        int idx = 0;
        for (auto& p : tmp[j]) {
            cvar[base + idx] = p.first;
            csign[base + idx] = p.second;
            int pp = pos[p.first]++;
            ivar_c[pp] = j; ivar_s[pp] = p.second;
            idx++;
        }
    }

    vector<int> x(n + 1, 0); // all WEST
    cnt.assign(m, 0);
    curr.assign(m, 0);
    for (int j = 0; j < m; j++) {
        int c = 0;
        for (int t = coff[j]; t < coff[j + 1]; t++) {
            int v = cvar[t], s = csign[t];
            bool sat = (s > 0) ? (x[v] == 1) : (x[v] == 0);
            if (sat) c++;
        }
        cnt[j] = c;
        curr[j] = (c % 3 == R[j]) ? 1 : 0;
    }

    auto gainFlip = [&](int v) -> ll {
        ll g = 0;
        for (int p = ioff[v]; p < ioff[v + 1]; p++) {
            int j = ivar_c[p], s = ivar_s[p];
            // literal currently met?
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
            cnt[j] += (met ? -1 : +1);
            curr[j] = (cnt[j] % 3 == R[j]) ? 1 : 0;
        }
        x[v] ^= 1;
    };

    // one sweep, first-improvement
    for (int v = 1; v <= n; v++) {
        if (gainFlip(v) > 0) doFlip(v);
    }

    for (int v = 1; v <= n; v++) printf("%d%c", x[v], v == n ? '\n' : ' ');
    return 0;
}
