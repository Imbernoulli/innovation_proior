// TIER: greedy
// Min-duration assignment + list scheduling with gap insertion.
// Each phase is assigned its fastest eligible crew; then we repeatedly dispatch the
// ready phase with the earliest release (SPT tie-break), placing it in the earliest
// feasible gap on its crew.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<int> o(n);
    // elig[j][k] = (crew,dur); assigned crew/dur chosen = min duration
    vector<vector<vector<pair<int, int>>>> elig(n);
    vector<vector<int>> asC(n), asD(n);
    for (int j = 0; j < n; j++) {
        scanf("%d", &o[j]);
        elig[j].resize(o[j]);
        asC[j].resize(o[j]);
        asD[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            int e;
            scanf("%d", &e);
            int bc = -1, bd = INT_MAX;
            for (int idx = 0; idx < e; idx++) {
                int c, d;
                scanf("%d %d", &c, &d);
                elig[j][k].push_back({c, d});
                if (d < bd) { bd = d; bc = c; }
            }
            asC[j][k] = bc;
            asD[j][k] = bd;
        }
    }

    vector<vector<pair<long long, long long>>> busy(m);
    vector<int> ptr(n, 0);
    vector<long long> lastEnd(n, 0);
    vector<vector<long long>> start(n);
    for (int j = 0; j < n; j++) start[j].assign(o[j], 0);

    int remaining = 0;
    for (int j = 0; j < n; j++) remaining += o[j];

    auto place = [&](int c, long long release, long long d) -> long long {
        auto& v = busy[c];
        sort(v.begin(), v.end());
        long long tpos = release;
        for (auto& iv : v) {
            if (tpos + d <= iv.first) break;      // fits before this interval
            if (iv.second > tpos) tpos = iv.second;
        }
        v.push_back({tpos, tpos + d});
        return tpos;
    };

    while (remaining > 0) {
        int bj = -1;
        long long bestRel = LLONG_MAX, bestDur = LLONG_MAX;
        for (int j = 0; j < n; j++) {
            if (ptr[j] >= o[j]) continue;
            long long rel = lastEnd[j];
            long long dur = asD[j][ptr[j]];
            if (rel < bestRel || (rel == bestRel && dur < bestDur)) {
                bestRel = rel; bestDur = dur; bj = j;
            }
        }
        int k = ptr[bj];
        int c = asC[bj][k];
        long long d = asD[bj][k];
        long long s = place(c, lastEnd[bj], d);
        start[bj][k] = s;
        lastEnd[bj] = s + d;
        ptr[bj]++;
        remaining--;
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%d %lld%c", asC[j][k], start[j][k], k + 1 == o[j] ? '\n' : ' ');
    return 0;
}
