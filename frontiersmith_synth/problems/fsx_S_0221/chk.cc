#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int F = inf.readInt();

    vector<vector<ll>> S(F, vector<ll>(F));
    for (int a = 0; a < F; a++)
        for (int b = 0; b < F; b++) S[a][b] = inf.readInt();

    vector<vector<int>> seq(n, vector<int>(m)), fam(n, vector<int>(m));
    vector<vector<ll>> p(n, vector<ll>(m));
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) seq[j][k] = inf.readInt();
        for (int k = 0; k < m; k++) p[j][k] = inf.readInt();
        for (int k = 0; k < m; k++) fam[j][k] = inf.readInt();
    }

    // ---- read participant starts (integer -> nan/inf auto-rejected) ----
    vector<vector<ll>> st(n, vector<ll>(m));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++)
            st[j][k] = ouf.readLong(0LL, (ll)4e18, "start");
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // ---- precedence ----
    for (int j = 0; j < n; j++) {
        for (int k = 1; k < m; k++) {
            if (st[j][k] < st[j][k - 1] + p[j][k - 1])
                quitf(_wa, "precedence violated: order %d position %d", j, k);
        }
    }

    // ---- station no-overlap + sequence-dependent setup ----
    vector<vector<pair<int,int>>> onm(m); // (job, pos) per station
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++)
            onm[seq[j][k]].push_back({j, k});

    for (int mm = 0; mm < m; mm++) {
        auto& v = onm[mm];
        sort(v.begin(), v.end(), [&](const pair<int,int>& x, const pair<int,int>& y) {
            ll sx = st[x.first][x.second], sy = st[y.first][y.second];
            if (sx != sy) return sx < sy;
            if (x.first != y.first) return x.first < y.first;
            return x.second < y.second;
        });
        for (size_t i = 1; i < v.size(); i++) {
            int jp = v[i-1].first, kp = v[i-1].second;
            int jc = v[i].first,   kc = v[i].second;
            ll enda = st[jp][kp] + p[jp][kp];
            ll need = enda + S[fam[jp][kp]][fam[jc][kc]];
            if (st[jc][kc] < need)
                quitf(_wa, "station %d overlap/setup: op(%d,%d) starts %lld < required %lld",
                      mm, jc, kc, st[jc][kc], need);
        }
    }

    // ---- participant makespan ----
    ll M = 0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++)
            M = max(M, st[j][k] + p[j][k]);

    // ---- baseline B: fully-sequential reference schedule ----
    vector<ll> jrdy(n, 0), mfree(m, 0);
    vector<int> mfamv(m, -1);
    ll t = 0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            int mm = seq[j][k], fa = fam[j][k];
            ll pp = p[j][k];
            ll est = jrdy[j];
            ll mst = (mfamv[mm] >= 0) ? (mfree[mm] + S[mfamv[mm]][fa]) : 0LL;
            ll start = max(t, max(est, mst));
            ll end = start + pp;
            t = end;
            jrdy[j] = end; mfree[mm] = end; mfamv[mm] = fa;
        }
    }
    ll B = t;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, M));
    quitp(sc / 1000.0, "OK M=%lld B=%lld Ratio: %.6f", M, B, sc / 1000.0);
    return 0;
}
