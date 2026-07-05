// TIER: greedy
// Per-round myopic set cover: repeatedly switch on the sensor covering the most
// still-deficient targets (tie: higher charge, then lower index). Cost-blind, no
// pruning -> feasible but wasteful.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int S, G, T, Cfull, Cmin, Drain, Recharge;
    if (scanf("%d %d %d", &S, &G, &T) != 3) return 0;
    scanf("%d %d %d %d", &Cfull, &Cmin, &Drain, &Recharge);
    vector<ll> D(G + 1);
    for (int g = 1; g <= G; g++) scanf("%lld", &D[g]);
    vector<vector<int>> cov(S + 1);
    vector<ll> c(S + 1);
    vector<vector<int>> tsens(G + 1);      // target -> sensors reaching it
    for (int s = 1; s <= S; s++) {
        int k;
        scanf("%lld %d", &c[s], &k);
        cov[s].resize(k);
        for (int j = 0; j < k; j++) { scanf("%d", &cov[s][j]); tsens[cov[s][j]].push_back(s); }
    }

    vector<ll> h(S + 1, Cfull);
    vector<ll> cover(G + 1, 0);
    vector<char> sat(G + 1, 0);
    vector<char> on(S + 1, 0);
    vector<int> defc(S + 1, 0);            // # still-deficient targets a sensor covers

    string out;
    out.reserve(1 << 16);

    for (int r = 1; r <= T; r++) {
        for (int g = 1; g <= G; g++) { cover[g] = 0; sat[g] = 0; }
        int unmet = 0;
        for (int g = 1; g <= G; g++) if (D[g] > 0) unmet++; else sat[g] = 1;
        for (int s = 1; s <= S; s++) { on[s] = 0; defc[s] = (int)cov[s].size(); }

        vector<int> chosen;
        while (unmet > 0) {
            int best = -1; ll bestKey1 = -1, bestH = -1;
            for (int s = 1; s <= S; s++) {
                if (on[s]) continue;
                if (defc[s] <= 0) continue;
                if (defc[s] > bestKey1 ||
                    (defc[s] == bestKey1 && h[s] > bestH)) {
                    bestKey1 = defc[s]; bestH = h[s]; best = s;
                }
            }
            if (best < 0) break;           // cannot happen: all-on is feasible
            on[best] = 1;
            chosen.push_back(best);
            for (int t : cov[best]) {
                if (sat[t]) continue;
                cover[t] += h[best];
                if (cover[t] >= D[t]) {
                    sat[t] = 1; unmet--;
                    for (int s2 : tsens[t]) if (!on[s2]) defc[s2]--;
                }
            }
        }

        out += to_string((int)chosen.size());
        for (int s : chosen) { out += ' '; out += to_string(s); }
        out += '\n';

        for (int s = 1; s <= S; s++) {
            if (on[s]) h[s] = max((ll)Cmin, h[s] - Drain);
            else       h[s] = min((ll)Cfull, h[s] + Recharge);
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
