// TIER: greedy
// Giffler-Thompson active-schedule generation with a Shortest-Processing-Time (SPT)
// conflict tie-break. One pass; a simple dispatch rule that beats the serial baseline.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, F;
vector<vector<ll>> S;
vector<vector<int>> seq, fam;
vector<vector<ll>> p;

int main() {
    if (scanf("%d %d %d", &n, &m, &F) != 3) return 0;
    S.assign(F, vector<ll>(F));
    for (int a = 0; a < F; a++)
        for (int b = 0; b < F; b++) scanf("%lld", &S[a][b]);
    seq.assign(n, vector<int>(m));
    fam.assign(n, vector<int>(m));
    p.assign(n, vector<ll>(m));
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) scanf("%d", &seq[j][k]);
        for (int k = 0; k < m; k++) scanf("%lld", &p[j][k]);
        for (int k = 0; k < m; k++) scanf("%d", &fam[j][k]);
    }

    vector<vector<ll>> st(n, vector<ll>(m));
    vector<int> pos(n, 0);
    vector<ll> jrdy(n, 0), mfree(m, 0), rem(n, 0);
    vector<int> mfamv(m, -1);
    for (int j = 0; j < n; j++) {
        ll s = 0; for (int k = 0; k < m; k++) s += p[j][k];
        rem[j] = s;
    }
    vector<int> active(n);
    for (int j = 0; j < n; j++) active[j] = j;
    int A = n;

    auto startOf = [&](int j) -> ll {
        int k = pos[j], mm = seq[j][k], fa = fam[j][k];
        ll mst = mfree[mm] + (mfamv[mm] >= 0 ? S[mfamv[mm]][fa] : 0LL);
        return max(jrdy[j], mst);
    };

    long long total = (long long)n * m;
    for (long long done = 0; done < total; done++) {
        // pass 1: earliest completion
        ll bestC = LLONG_MAX; int bestM = -1;
        for (int a = 0; a < A; a++) {
            int j = active[a];
            ll s = startOf(j);
            ll c = s + p[j][pos[j]];
            if (c < bestC) { bestC = c; bestM = seq[j][pos[j]]; }
        }
        // pass 2: conflict set on bestM, pick SPT (min p, tie min job)
        int chosen = -1; ll chP = 0; int chJ = 0;
        for (int a = 0; a < A; a++) {
            int j = active[a];
            if (seq[j][pos[j]] != bestM) continue;
            ll s = startOf(j);
            if (s < bestC) {
                ll pk = p[j][pos[j]];
                if (chosen == -1 || pk < chP || (pk == chP && j < chJ)) {
                    chosen = a; chP = pk; chJ = j;
                }
            }
        }
        // schedule chosen
        int j = active[chosen], k = pos[j], mm = seq[j][k], fa = fam[j][k];
        ll s = startOf(j), e = s + p[j][k];
        st[j][k] = s;
        jrdy[j] = e; mfree[mm] = e; mfamv[mm] = fa; rem[j] -= p[j][k];
        pos[j]++;
        if (pos[j] == m) { active[chosen] = active[A - 1]; A--; }
    }

    string out; out.reserve((size_t)n * m * 7);
    char buf[24];
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) {
            int len = sprintf(buf, "%lld", st[j][k]);
            out.append(buf, len);
            out.push_back(k + 1 < m ? ' ' : '\n');
        }
    fputs(out.c_str(), stdout);
    return 0;
}
