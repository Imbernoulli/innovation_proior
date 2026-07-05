#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    // Per (order, pass): list of (dam -> duration) eligible options.
    vector<vector<vector<pair<int,int>>>> opt(n);
    long long B = 0;  // sum of fastest eligible durations = fully-serial baseline
    for (int j = 0; j < n; j++) {
        int o = inf.readInt();
        opt[j].resize(o);
        for (int k = 0; k < o; k++) {
            int c = inf.readInt();
            int best = INT_MAX;
            for (int e = 0; e < c; e++) {
                int dam = inf.readInt();
                int dur = inf.readInt();
                opt[j][k].push_back({dam, dur});
                best = min(best, dur);
            }
            B += best;
        }
    }

    // Read participant schedule; validate range, eligibility, precedence.
    vector<vector<pair<long long,long long>>> iv(m);  // per-dam busy intervals [s,e)
    long long F = 0;
    const long long SMAX = 2000000000LL;
    for (int j = 0; j < n; j++) {
        int o = (int)opt[j].size();
        long long prevEnd = 0;
        for (int k = 0; k < o; k++) {
            int dam = ouf.readInt(0, m - 1, format("dam[%d][%d]", j, k));
            long long s = ouf.readLong(0LL, SMAX, format("start[%d][%d]", j, k));
            // eligibility + resolve duration
            int dur = -1;
            for (auto& pr : opt[j][k])
                if (pr.first == dam) { dur = pr.second; break; }
            if (dur < 0)
                quitf(_wa, "order %d pass %d routed to ineligible dam %d", j, k, dam);
            if (s < prevEnd)
                quitf(_wa, "precedence violated: order %d pass %d starts at %lld but previous pass ends at %lld",
                      j, k, s, prevEnd);
            long long e = s + dur;
            prevEnd = e;
            F = max(F, e);
            iv[dam].push_back({s, e});
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // No-overlap per dam.
    for (int d = 0; d < m; d++) {
        auto& v = iv[d];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "dam %d runs overlapping passes: [%lld,%lld) and [%lld,%lld)",
                      d, v[i - 1].first, v[i - 1].second, v[i].first, v[i].second);
    }

    if (B <= 0) B = 1;  // guard (never happens: durations >= 1)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
