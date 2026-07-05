// TIER: greedy
// Myopic per-step commitment: keep only what warm-up forces on, then add the
// lowest no-load generators until capacity meets this step's residual demand.
// Ignores fuel rate, start-up churn and look-ahead -> beats "all on" but leaves
// plenty on the table.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T, G;
    if (scanf("%d %d", &T, &G) != 2) return 0;
    vector<ll> P(G), b(G), rate(G), K(G), U(G);
    for (int g = 0; g < G; g++) scanf("%lld %lld %lld %lld %lld",&P[g],&b[g],&rate[g],&K[g],&U[g]);
    vector<ll> D(T), W(T);
    for (int t = 0; t < T; t++) scanf("%lld %lld",&D[t],&W[t]);

    vector<vector<char>> x(T, vector<char>(G, 0));
    vector<int> mustOnUntil(G, -1);           // inclusive last forced-on step

    // candidate order: cheapest no-load first (tie: bigger capacity first)
    vector<int> cand(G);
    for (int g = 0; g < G; g++) cand[g] = g;
    sort(cand.begin(), cand.end(), [&](int a, int c){
        if (b[a] != b[c]) return b[a] < b[c];
        return P[a] > P[c];
    });

    for (int t = 0; t < T; t++) {
        ll R = D[t] - W[t]; if (R < 0) R = 0;
        ll cap = 0;
        for (int g = 0; g < G; g++) if (mustOnUntil[g] >= t) { x[t][g] = 1; cap += P[g]; }
        for (int idx = 0; idx < G && cap < R; idx++) {
            int g = cand[idx];
            if (x[t][g]) continue;
            x[t][g] = 1; cap += P[g];
            mustOnUntil[g] = min((ll)T - 1, (ll)t + U[g] - 1);
        }
    }

    for (int t = 0; t < T; t++) {
        for (int g = 0; g < G; g++) { if (g) fputc(' ', stdout); fputc(x[t][g] ? '1':'0', stdout); }
        fputc('\n', stdout);
    }
    return 0;
}
