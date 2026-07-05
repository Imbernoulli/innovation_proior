#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int M = inf.readInt(2, 1000000, "M");
    int J = inf.readInt(1, 1000000, "J");

    vector<vector<int>> mach(J), grn(J);
    ll B = 0;                              // sum of all green windows = fully-serial makespan
    ll totalOps = 0;
    for (int j = 0; j < J; j++) {
        int L = inf.readInt(1, 1000000, "L");
        mach[j].resize(L);
        grn[j].resize(L);
        for (int i = 0; i < L; i++) {
            mach[j][i] = inf.readInt(1, M, "m");
            grn[j][i] = inf.readInt(1, 1000000, "p");
            B += grn[j][i];
            totalOps++;
        }
    }
    if (B <= 0 || totalOps <= 0) quitf(_fail, "bad instance: B=%lld ops=%lld", B, totalOps);

    // ---- read participant's start times (in convoy/route order) ----
    const ll START_HI = (ll)4e15;
    vector<vector<ll>> st(J);
    // per-machine list of (start, end) intervals for the no-overlap test
    vector<vector<pair<ll,ll>>> ivs(M + 1);
    ll F = 0;
    for (int j = 0; j < J; j++) {
        int L = (int)mach[j].size();
        st[j].resize(L);
        for (int i = 0; i < L; i++) {
            ll s = ouf.readLong(0LL, START_HI, "start");
            st[j][i] = s;
            if (i >= 1) {
                ll needed = st[j][i - 1] + grn[j][i - 1];
                if (s < needed)
                    quitf(_wa, "precedence: convoy %d stop %d starts at %lld < %lld (previous stop ends)",
                          j + 1, i + 1, s, needed);
            }
            ll e = s + grn[j][i];
            ivs[mach[j][i]].push_back({s, e});
            F = max(F, e);
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- no-overlap per intersection ----
    for (int m = 1; m <= M; m++) {
        auto& v = ivs[m];
        sort(v.begin(), v.end());
        for (size_t a = 1; a < v.size(); a++) {
            if (v[a].first < v[a - 1].second)
                quitf(_wa, "no-overlap violated at intersection %d: [%lld,%lld) overlaps [%lld,%lld)",
                      m, v[a - 1].first, v[a - 1].second, v[a].first, v[a].second);
        }
    }

    if (F <= 0) quitf(_wa, "empty/degenerate schedule");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
