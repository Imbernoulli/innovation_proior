#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<vector<int>> mach(n), dur(n);
    long long B = 0;  // total work = fully-serial makespan (the calibration baseline)
    for (int j = 0; j < n; j++) {
        int o = inf.readInt();
        mach[j].resize(o);
        dur[j].resize(o);
        for (int k = 0; k < o; k++) {
            mach[j][k] = inf.readInt();
            dur[j][k]  = inf.readInt();
            B += dur[j][k];
        }
    }

    // Read participant schedule; validate precedence and range.
    vector<vector<pair<long long, long long>>> iv(m);  // per-team intervals [s, e)
    long long F = 0;
    const long long SMAX = 2000000000LL;
    for (int j = 0; j < n; j++) {
        int o = (int)mach[j].size();
        long long prevEnd = 0;
        for (int k = 0; k < o; k++) {
            long long s = ouf.readLong(0LL, SMAX, format("s[%d][%d]", j, k));
            if (s < prevEnd)
                quitf(_wa, "precedence violated: mission %d leg %d starts at %lld but previous leg ends at %lld",
                      j, k, s, prevEnd);
            long long e = s + dur[j][k];
            prevEnd = e;
            F = max(F, e);
            iv[mach[j][k]].push_back({s, e});
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // No-overlap per team.
    for (int t = 0; t < m; t++) {
        auto& v = iv[t];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "team %d handles overlapping legs: [%lld,%lld) and [%lld,%lld)",
                      t, v[i - 1].first, v[i - 1].second, v[i].first, v[i].second);
    }

    if (B <= 0) B = 1;  // guard (never happens: durations >= 1)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
