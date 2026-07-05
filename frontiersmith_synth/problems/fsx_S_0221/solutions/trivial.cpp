// TIER: trivial
// Fully-sequential reference schedule (exactly the checker's baseline B) -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, F;
    if (scanf("%d %d %d", &n, &m, &F) != 3) return 0;
    vector<vector<ll>> S(F, vector<ll>(F));
    for (int a = 0; a < F; a++)
        for (int b = 0; b < F; b++) scanf("%lld", &S[a][b]);

    vector<vector<int>> seq(n, vector<int>(m)), fam(n, vector<int>(m));
    vector<vector<ll>> p(n, vector<ll>(m));
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) scanf("%d", &seq[j][k]);
        for (int k = 0; k < m; k++) scanf("%lld", &p[j][k]);
        for (int k = 0; k < m; k++) scanf("%d", &fam[j][k]);
    }

    vector<vector<ll>> st(n, vector<ll>(m));
    vector<ll> jrdy(n, 0), mfree(m, 0);
    vector<int> mfamv(m, -1);
    ll t = 0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            int mm = seq[j][k], fa = fam[j][k];
            ll est = jrdy[j];
            ll mst = (mfamv[mm] >= 0) ? (mfree[mm] + S[mfamv[mm]][fa]) : 0LL;
            ll start = max(t, max(est, mst));
            ll end = start + p[j][k];
            st[j][k] = start;
            t = end; jrdy[j] = end; mfree[mm] = end; mfamv[mm] = fa;
        }
    }

    string out;
    out.reserve((size_t)n * m * 7);
    char buf[24];
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            int len = sprintf(buf, "%lld", st[j][k]);
            out.append(buf, len);
            out.push_back(k + 1 < m ? ' ' : '\n');
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
