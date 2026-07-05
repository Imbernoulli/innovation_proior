#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int J = inf.readInt();
    int M = inf.readInt();

    vector<vector<int>> mach(J), dur(J);
    long long B = 0; // total work == serial makespan == baseline
    for (int i = 0; i < J; i++) {
        int L = inf.readInt();
        mach[i].resize(L);
        dur[i].resize(L);
        for (int k = 0; k < L; k++) {
            mach[i][k] = inf.readInt();
            dur[i][k] = inf.readInt();
            B += dur[i][k];
        }
    }

    const long long SMAX = (long long)1e12;
    // per-bay list of intervals [start,end)
    vector<vector<pair<long long,long long>>> bay(M + 1);

    long long F = 0;
    for (int i = 0; i < J; i++) {
        int L = (int)mach[i].size();
        long long prevEnd = 0;
        for (int k = 0; k < L; k++) {
            long long s = ouf.readLong(0, SMAX, format("s[%d][%d]", i + 1, k + 1));
            long long e = s + dur[i][k];
            if (s < prevEnd) {
                quitf(_wa, "inverter %d stage %d starts at %lld before previous stage finished at %lld",
                      i + 1, k + 1, s, prevEnd);
            }
            int m = mach[i][k];
            bay[m].push_back({s, e});
            prevEnd = e;
            if (e > F) F = e;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // no-overlap on every bay
    for (int m = 1; m <= M; m++) {
        auto &v = bay[m];
        sort(v.begin(), v.end());
        for (size_t j = 1; j < v.size(); j++) {
            if (v[j].first < v[j - 1].second) {
                quitf(_wa, "bay %d double-booked: interval [%lld,%lld) overlaps [%lld,%lld)",
                      m, v[j].first, v[j].second, v[j - 1].first, v[j - 1].second);
            }
        }
    }

    if (B <= 0) quitf(_fail, "internal baseline non-positive");
    if (F <= 0) quitf(_wa, "makespan must be positive");

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
