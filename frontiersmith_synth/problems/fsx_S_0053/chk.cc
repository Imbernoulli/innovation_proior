#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    // elig[j][k] = list of (crew, duration) eligible for phase k of stage j.
    vector<vector<vector<pair<int, int>>>> elig(n);
    long long B = 0;  // serial-default makespan: sum of first-eligible durations.
    for (int j = 0; j < n; j++) {
        int o = inf.readInt();
        elig[j].resize(o);
        for (int k = 0; k < o; k++) {
            int e = inf.readInt();
            for (int idx = 0; idx < e; idx++) {
                int c = inf.readInt();
                int d = inf.readInt();
                elig[j][k].push_back({c, d});
            }
            B += elig[j][k][0].second;  // first eligible crew's duration
        }
    }

    // Read participant schedule; validate eligibility, range, precedence.
    vector<vector<pair<long long, long long>>> iv(m);  // per-crew busy intervals [s,e)
    long long F = 0;
    const long long SMAX = 2000000000LL;
    for (int j = 0; j < n; j++) {
        int o = (int)elig[j].size();
        long long prevEnd = 0;
        for (int k = 0; k < o; k++) {
            int c = ouf.readInt(0, m - 1, format("c[%d][%d]", j, k));
            // resolve duration of chosen crew; must be eligible
            long long d = -1;
            for (auto& pr : elig[j][k])
                if (pr.first == c) { d = pr.second; break; }
            if (d < 0)
                quitf(_wa, "stage %d phase %d assigned to crew %d which is not eligible",
                      j, k, c);
            long long s = ouf.readLong(0LL, SMAX, format("s[%d][%d]", j, k));
            if (s < prevEnd)
                quitf(_wa, "precedence violated: stage %d phase %d starts at %lld but previous phase ends at %lld",
                      j, k, s, prevEnd);
            long long en = s + d;
            prevEnd = en;
            F = max(F, en);
            iv[c].push_back({s, en});
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // No-overlap per crew.
    for (int c = 0; c < m; c++) {
        auto& v = iv[c];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "crew %d works overlapping phases: [%lld,%lld) and [%lld,%lld)",
                      c, v[i - 1].first, v[i - 1].second, v[i].first, v[i].second);
    }

    if (B <= 0) B = 1;  // guard (durations >= 1 so never triggers)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
