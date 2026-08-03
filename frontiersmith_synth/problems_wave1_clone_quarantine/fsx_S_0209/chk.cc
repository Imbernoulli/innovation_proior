#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<int> w(n);
    vector<vector<int>> mach(n), dur(n);
    for (int j = 0; j < n; j++) {
        w[j] = inf.readInt();
        int o = inf.readInt();
        mach[j].resize(o);
        dur[j].resize(o);
        for (int k = 0; k < o; k++) {
            mach[j][k] = inf.readInt();
            dur[j][k]  = inf.readInt();
        }
    }

    // Internal baseline B: fully-serial schedule in ride index order, all tasks
    // back-to-back on one global timeline. Weighted opening time of that schedule.
    long long B = 0, cursor = 0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < (int)dur[j].size(); k++) cursor += dur[j][k];
        B += (long long)w[j] * cursor;   // C_j = cursor after ride j's last task
    }
    if (B <= 0) B = 1;  // guard (never happens: weights, durations >= 1)

    // Read participant schedule; validate range + precedence; collect per-crew intervals.
    vector<vector<pair<long long, long long>>> iv(m);  // per-crew [s, e)
    long long F = 0;
    const long long SMAX = 2000000000LL;
    for (int j = 0; j < n; j++) {
        int o = (int)mach[j].size();
        long long prevEnd = 0, lastEnd = 0;
        for (int k = 0; k < o; k++) {
            long long s = ouf.readLong(0LL, SMAX, format("s[%d][%d]", j, k));
            if (s < prevEnd)
                quitf(_wa, "precedence violated: ride %d task %d starts at %lld but previous task ends at %lld",
                      j, k, s, prevEnd);
            long long e = s + dur[j][k];
            prevEnd = e;
            lastEnd = e;
            iv[mach[j][k]].push_back({s, e});
        }
        F += (long long)w[j] * lastEnd;   // weighted opening time of ride j
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // No-overlap per crew.
    for (int c = 0; c < m; c++) {
        auto& v = iv[c];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "crew %d handles overlapping tasks: [%lld,%lld) and [%lld,%lld)",
                      c, v[i - 1].first, v[i - 1].second, v[i].first, v[i].second);
    }

    if (F <= 0) F = 1;  // guard (never happens: weights, durations >= 1)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
