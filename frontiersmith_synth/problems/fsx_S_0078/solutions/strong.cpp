// TIER: strong
// Multi-start constructive heuristic + local search.  Builds candidate assignments from
// several rack orderings (by value, by value-density v/min-load, by footprint), keeps the
// best-value one, then improves it with (a) a fill pass placing any still-uncooled rack and
// (b) an eviction-swap pass that kicks a lower-value rack off a unit to admit a higher-value
// rack whenever that strictly raises total served value.  Deterministic.
#include <bits/stdc++.h>
using namespace std;

static int N, M;
static vector<long long> v, C;
static vector<vector<long long>> d;

// construct by a given rack order; returns assignment and its total value.
static pair<vector<int>, long long> construct(const vector<int>& order) {
    vector<long long> rem = C;
    vector<int> a(N, 0);
    long long val = 0;
    for (int i : order) {
        int best = -1; long long bl = LLONG_MAX;
        for (int j = 0; j < M; j++)
            if (rem[j] >= d[i][j] && d[i][j] < bl) { bl = d[i][j]; best = j; }
        if (best >= 0) { rem[best] -= d[i][best]; a[i] = best + 1; val += v[i]; }
    }
    return {a, val};
}

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 0;
    v.resize(N); for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    C.resize(M); for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    d.assign(N, vector<long long>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) scanf("%lld", &d[i][j]);

    vector<long long> minLoad(N);
    for (int i = 0; i < N; i++) { long long m = LLONG_MAX; for (int j = 0; j < M; j++) m = min(m, d[i][j]); minLoad[i] = max(1LL, m); }

    vector<int> base(N); iota(base.begin(), base.end(), 0);

    vector<vector<int>> orders;
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){ return v[x] != v[y] ? v[x] > v[y] : x < y; }); orders.push_back(o); }
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){ double a1=(double)v[x]/minLoad[x], a2=(double)v[y]/minLoad[y]; return a1 != a2 ? a1 > a2 : x < y; }); orders.push_back(o); }
    { auto o = base; sort(o.begin(), o.end(), [&](int x,int y){ return minLoad[x] != minLoad[y] ? minLoad[x] < minLoad[y] : x < y; }); orders.push_back(o); }

    vector<int> bestA; long long bestVal = -1;
    for (auto& o : orders) {
        auto r = construct(o);
        if (r.second > bestVal) { bestVal = r.second; bestA = r.first; }
    }

    // rebuild remaining capacities + per-unit member lists for local search
    vector<long long> rem = C;
    vector<vector<int>> members(M);
    for (int i = 0; i < N; i++) if (bestA[i] != 0) { int j = bestA[i]-1; rem[j] -= d[i][j]; members[j].push_back(i); }

    for (int pass = 0; pass < 6; pass++) {
        bool improved = false;
        // (a) fill: place any uncooled rack that now fits
        for (int i = 0; i < N; i++) if (bestA[i] == 0) {
            int best = -1; long long bl = LLONG_MAX;
            for (int j = 0; j < M; j++) if (rem[j] >= d[i][j] && d[i][j] < bl) { bl = d[i][j]; best = j; }
            if (best >= 0) { rem[best] -= d[i][best]; bestA[i] = best+1; members[best].push_back(i); improved = true; }
        }
        // (b) eviction swap: admit uncooled rack i by evicting one lower-value rack
        for (int i = 0; i < N; i++) if (bestA[i] == 0) {
            int chosenUnit = -1, evict = -1; long long lossBest = LLONG_MAX;
            for (int j = 0; j < M; j++) {
                if (d[i][j] > C[j]) continue;
                long long need = d[i][j] - rem[j];
                if (need <= 0) { chosenUnit = j; evict = -1; lossBest = 0; break; } // fits directly
                // find a single lower-value member whose removal frees enough
                for (int k : members[j]) {
                    if (v[k] < v[i] && rem[j] + d[k][j] >= d[i][j] && v[k] < lossBest) {
                        lossBest = v[k]; evict = k; chosenUnit = j;
                    }
                }
            }
            if (chosenUnit >= 0 && (evict >= 0 || lossBest == 0)) {
                int j = chosenUnit;
                if (evict >= 0) {
                    // remove evict
                    rem[j] += d[evict][j]; bestA[evict] = 0;
                    auto& mv = members[j];
                    mv.erase(find(mv.begin(), mv.end(), evict));
                }
                rem[j] -= d[i][j]; bestA[i] = j+1; members[j].push_back(i);
                improved = true;
            }
        }
        if (!improved) break;
    }

    for (int i = 0; i < N; i++) printf("%d%c", bestA[i], i + 1 < N ? ' ' : '\n');
    return 0;
}
