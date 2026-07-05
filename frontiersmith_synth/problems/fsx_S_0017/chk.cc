#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int J = inf.readInt();
    int M = inf.readInt();

    vector<vector<pair<int,int>>> jobs(J); // (machine, dur)
    ll B = 0;           // internal baseline = sum of all durations (sequential schedule)
    int total = 0;
    for (int j = 0; j < J; j++) {
        int k = inf.readInt();
        for (int o = 0; o < k; o++) {
            int m = inf.readInt();
            int d = inf.readInt();
            jobs[j].push_back({m, d});
            B += d;
            total++;
        }
    }

    // ---- read participant's start times (one per operation, input order) ----
    vector<ll> st(total);
    {
        int idx = 0;
        for (int j = 0; j < J; j++)
            for (size_t o = 0; o < jobs[j].size(); o++)
                st[idx++] = ouf.readLong(0LL, 2000000000LL, "start");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- validate precedence + compute per-machine intervals + makespan ----
    vector<vector<pair<ll,ll>>> byM(M + 1); // machine -> list of [start,end)
    ll F = 0;
    int idx = 0;
    for (int j = 0; j < J; j++) {
        ll prevEnd = -1;
        for (size_t o = 0; o < jobs[j].size(); o++) {
            ll s = st[idx];
            ll e = s + jobs[j][o].second;
            if (o > 0 && s < prevEnd)
                quitf(_wa, "precedence violated: module %d op %d starts at %lld < %lld (end of previous op)",
                      j + 1, (int)o + 1, s, prevEnd);
            byM[jobs[j][o].first].push_back({s, e});
            if (e > F) F = e;
            prevEnd = e;
            idx++;
        }
    }

    // ---- one-operation-per-workstation (no overlap) ----
    for (int m = 1; m <= M; m++) {
        auto& v = byM[m];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++) {
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "workstation %d overlap: [%lld,%lld) and [%lld,%lld)",
                      m, v[i - 1].first, v[i - 1].second, v[i].first, v[i].second);
        }
    }

    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);
    if (F <= 0) quitf(_wa, "nonpositive makespan");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
