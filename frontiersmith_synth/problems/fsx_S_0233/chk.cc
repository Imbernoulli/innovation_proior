#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<long long> r(n);
    vector<int> o(n);
    vector<vector<int>> mac(n), dur(n);
    for (int j = 0; j < n; j++) {
        r[j] = inf.readInt();
        o[j] = inf.readInt();
        mac[j].resize(o[j]);
        dur[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            mac[j][k] = inf.readInt();
            dur[j][k] = inf.readInt();
        }
    }

    const long long SMAX = 2000000000LL;
    vector<vector<long long>> s(n);
    for (int j = 0; j < n; j++) {
        s[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            s[j][k] = ouf.readLong(0, SMAX, format("s[%d][%d]", j, k));
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // ready time + precedence
    for (int j = 0; j < n; j++) {
        if (s[j][0] < r[j])
            quitf(_wa, "telescope %d starts at %lld before ready time %lld", j, s[j][0], r[j]);
        for (int k = 1; k < o[j]; k++)
            if (s[j][k] < s[j][k - 1] + (long long)dur[j][k - 1])
                quitf(_wa, "telescope %d precedence violated at step %d", j, k);
    }

    // no-overlap per resource
    vector<vector<pair<long long, long long>>> iv(m);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            iv[mac[j][k]].push_back({s[j][k], s[j][k] + (long long)dur[j][k]});
    for (int c = 0; c < m; c++) {
        sort(iv[c].begin(), iv[c].end());
        for (size_t i = 1; i < iv[c].size(); i++)
            if (iv[c][i].first < iv[c][i - 1].second)
                quitf(_wa, "resource %d serves two steps at once", c);
    }

    // participant makespan F
    long long F = 0;
    for (int j = 0; j < n; j++)
        F = max(F, s[j][o[j] - 1] + (long long)dur[j][o[j] - 1]);

    // internal serial baseline B (always feasible)
    long long cur = 0;
    for (int j = 0; j < n; j++) {
        long long tt = max(cur, r[j]);
        for (int k = 0; k < o[j]; k++) tt += dur[j][k];
        cur = tt;
    }
    long long B = cur;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
