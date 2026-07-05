// TIER: strong
// Per-round cost-aware set cover (pick argmax of (charge * deficit-covered)/cost),
// then a redundancy-pruning pass that drops the most expensive still-redundant
// sensors. Cost-aware selection + pruning -> markedly cheaper than the cost-blind
// greedy, with different per-test structure.
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
    vector<vector<int>> tsens(G + 1);
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
    vector<int> defc(S + 1, 0);

    string out;
    out.reserve(1 << 16);

    for (int r = 1; r <= T; r++) {
        for (int g = 1; g <= G; g++) { cover[g] = 0; sat[g] = 0; }
        int unmet = 0;
        for (int g = 1; g <= G; g++) if (D[g] > 0) unmet++; else sat[g] = 1;
        for (int s = 1; s <= S; s++) { on[s] = 0; defc[s] = (int)cov[s].size(); }

        // ---- greedy build by value-per-cost ----
        while (unmet > 0) {
            int best = -1; double bestKey = -1.0;
            for (int s = 1; s <= S; s++) {
                if (on[s] || defc[s] <= 0) continue;
                double key = (double)defc[s] * (double)h[s] / (double)c[s];
                if (key > bestKey) { bestKey = key; best = s; }
            }
            if (best < 0) break;
            on[best] = 1;
            for (int t : cov[best]) {
                if (sat[t]) continue;
                cover[t] += h[best];
                if (cover[t] >= D[t]) {
                    sat[t] = 1; unmet--;
                    for (int s2 : tsens[t]) if (!on[s2]) defc[s2]--;
                }
            }
        }

        // ---- redundancy pruning: drop most-expensive still-redundant sensors ----
        vector<int> act;
        for (int s = 1; s <= S; s++) if (on[s]) act.push_back(s);
        sort(act.begin(), act.end(), [&](int a, int b) {
            if (c[a] != c[b]) return c[a] > c[b];       // expensive first
            return h[a] < h[b];
        });
        for (int s : act) {
            bool removable = true;
            for (int t : cov[s]) {
                if (cover[t] - h[s] < D[t]) { removable = false; break; }
            }
            if (removable) {
                on[s] = 0;
                for (int t : cov[s]) cover[t] -= h[s];
            }
        }

        vector<int> chosen;
        for (int s = 1; s <= S; s++) if (on[s]) chosen.push_back(s);
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
