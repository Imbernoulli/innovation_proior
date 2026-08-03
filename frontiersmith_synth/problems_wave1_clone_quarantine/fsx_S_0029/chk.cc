#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Wildlife Corridor Construction: total weighted completion time JSSP scorer.
//   F = sum_j w_j * C_j  (participant's weighted total completion, minimize)
//   B = weighted total completion of the fully-serial baseline (corridor by corridor
//       in input order on one global timeline) -- always feasible, the calibration point.
//   ratio = min(1.0, 0.1 * B / F).
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<vector<int>> mach(n), dur(n);
    vector<long long> w(n);

    for (int j = 0; j < n; j++) {
        int o = inf.readInt();
        w[j] = inf.readInt();
        mach[j].resize(o);
        dur[j].resize(o);
        for (int k = 0; k < o; k++) {
            mach[j][k] = inf.readInt();
            dur[j][k]  = inf.readInt();
        }
    }

    // Internal baseline B: serial global timeline, corridor 0 fully, then 1, ...
    long long B = 0, clock = 0;
    for (int j = 0; j < n; j++) {
        for (size_t k = 0; k < dur[j].size(); k++) clock += dur[j][k];
        B += w[j] * clock;   // clock is now this corridor's completion day
    }
    if (B <= 0) B = 1;  // guard (never happens: durations,weights >= 1)

    // Read + validate participant schedule.
    vector<vector<pair<long long,long long>>> iv(m);  // per-crew [s,e) intervals
    long long F = 0;
    const long long SMAX = 1000000000LL;
    for (int j = 0; j < n; j++) {
        int o = (int)mach[j].size();
        long long prevEnd = 0, cj = 0;
        for (int k = 0; k < o; k++) {
            long long s = ouf.readLong(0LL, SMAX, format("s[%d][%d]", j, k));
            if (s < prevEnd)
                quitf(_wa, "precedence violated: corridor %d segment %d starts %lld but previous ends %lld",
                      j, k, s, prevEnd);
            long long e = s + dur[j][k];
            prevEnd = e;
            cj = e;
            iv[mach[j][k]].push_back({s, e});
        }
        F += w[j] * cj;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // No-overlap per crew.
    for (int c = 0; c < m; c++) {
        auto& v = iv[c];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i-1].second)
                quitf(_wa, "crew %d builds overlapping segments: [%lld,%lld) and [%lld,%lld)",
                      c, v[i-1].first, v[i-1].second, v[i].first, v[i].second);
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
