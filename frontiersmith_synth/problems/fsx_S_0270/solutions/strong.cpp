// TIER: strong
// Value-density greedy seed + reassignment local search with randomized restarts.
// Moves: (re)assign a beacon to a different station, drop it, or swap two beacons'
// stations -- accept any move that raises total value while respecting all budgets.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m, n;
vector<vector<int>> v, w;
vector<int> cap;

ll totalValue(const vector<int>& a) {
    ll F = 0;
    for (int j = 1; j <= n; j++) if (a[j]) F += v[a[j]][j];
    return F;
}

int main() {
    if (scanf("%d %d", &m, &n) != 2) return 0;
    cap.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) scanf("%d", &cap[i]);
    v.assign(m + 1, vector<int>(n + 1));
    w.assign(m + 1, vector<int>(n + 1));
    for (int j = 1; j <= n; j++)
        for (int i = 1; i <= m; i++) scanf("%d %d", &v[i][j], &w[i][j]);

    // ---- greedy seed (value density) ----
    struct Opt { double d; int val; int j; int i; };
    vector<Opt> opts;
    for (int j = 1; j <= n; j++)
        for (int i = 1; i <= m; i++)
            opts.push_back({(double)v[i][j] / (double)w[i][j], v[i][j], j, i});
    sort(opts.begin(), opts.end(), [](const Opt& A, const Opt& B) {
        if (A.d != B.d) return A.d > B.d;
        return A.val > B.val;
    });
    vector<int> baseA(n + 1, 0), rem(cap);
    vector<char> done(n + 1, 0);
    for (auto& o : opts) {
        if (done[o.j]) continue;
        if (w[o.i][o.j] <= rem[o.i]) { baseA[o.j] = o.i; rem[o.i] -= w[o.i][o.j]; done[o.j] = 1; }
    }

    auto localSearch = [&](vector<int> a, mt19937& rng) {
        vector<ll> load(m + 1, 0);
        for (int j = 1; j <= n; j++) if (a[j]) load[a[j]] += w[a[j]][j];
        bool improved = true;
        int guard = 0;
        while (improved && guard++ < 4000) {
            improved = false;
            // single-beacon best reassignment
            for (int j = 1; j <= n; j++) {
                int cur = a[j];
                int curVal = cur ? v[cur][j] : 0;
                int bestI = cur, bestGain = 0;
                for (int i = 0; i <= m; i++) {
                    if (i == cur) continue;
                    int newVal = i ? v[i][j] : 0;
                    if (i != 0) {
                        ll nl = load[i] + w[i][j];
                        if (nl > cap[i]) continue;
                    }
                    int gain = newVal - curVal;
                    if (gain > bestGain) { bestGain = gain; bestI = i; }
                }
                if (bestI != cur) {
                    if (cur) load[cur] -= w[cur][j];
                    if (bestI) load[bestI] += w[bestI][j];
                    a[j] = bestI; improved = true;
                }
            }
            // swap two beacons' stations if it helps and stays feasible
            int tries = 3 * n;
            while (tries-- > 0) {
                int x = uniform_int_distribution<int>(1, n)(rng);
                int y = uniform_int_distribution<int>(1, n)(rng);
                if (x == y) continue;
                int ix = a[x], iy = a[y];
                if (ix == iy) continue;
                if (ix == 0 || iy == 0) continue;      // only swap two assigned beacons
                int oldVal = v[ix][x] + v[iy][y];
                int newVal = v[iy][x] + v[ix][y];      // x->iy, y->ix
                if (newVal <= oldVal) continue;
                // station ix: lose x, gain y ; station iy: lose y, gain x
                ll newLix = load[ix] - w[ix][x] + w[ix][y];
                ll newLiy = load[iy] - w[iy][y] + w[iy][x];
                if (newLix > cap[ix] || newLiy > cap[iy]) continue;
                load[ix] = newLix; load[iy] = newLiy;
                a[x] = iy; a[y] = ix; improved = true;
            }
        }
        return a;
    };

    mt19937 rng(987654321u);
    vector<int> best = localSearch(baseA, rng);
    ll bestF = totalValue(best);

    // randomized restarts: perturb the seed and re-optimize
    for (int r = 0; r < 12; r++) {
        vector<int> a(n + 1, 0);
        vector<ll> load(m + 1, 0);
        // randomized greedy: shuffle beacon processing order, assign to a random
        // among the top-value feasible stations
        vector<int> order(n);
        for (int j = 0; j < n; j++) order[j] = j + 1;
        shuffle(order.begin(), order.end(), rng);
        for (int idx = 0; idx < n; idx++) {
            int j = order[idx];
            // candidate stations that fit, ranked by value
            vector<pair<int,int>> cand; // (value, station)
            for (int i = 1; i <= m; i++)
                if (load[i] + w[i][j] <= cap[i]) cand.push_back({v[i][j], i});
            if (cand.empty()) continue;
            sort(cand.rbegin(), cand.rend());
            int topk = min((int)cand.size(), 2);
            int pick = cand[uniform_int_distribution<int>(0, topk - 1)(rng)].second;
            a[j] = pick; load[pick] += w[pick][j];
        }
        vector<int> improved = localSearch(a, rng);
        ll f = totalValue(improved);
        if (f > bestF) { bestF = f; best = improved; }
    }

    for (int j = 1; j <= n; j++) printf("%d%c", best[j], j == n ? '\n' : ' ');
    return 0;
}
