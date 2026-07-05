#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int J = inf.readInt(1, 2000000, "J");
    int M = inf.readInt(1, 6, "M");

    // read instance: per operation, its bay and duration
    vector<vector<int>> bay(J, vector<int>(M));
    vector<vector<ll>> dur(J, vector<ll>(M));
    ll T = 0; // fully-serialized makespan = baseline B
    for (int j = 0; j < J; j++) {
        for (int o = 0; o < M; o++) {
            bay[j][o] = inf.readInt(1, M, "bay");
            dur[j][o] = inf.readInt(1, 1000, "dur");
            T += dur[j][o];
        }
    }
    if (T <= 0) quitf(_fail, "bad instance: T=%lld", T);

    ll HORIZON = 10LL * T; // generous but finite start-time bound

    // read participant schedule: J*M start times, row-major
    vector<vector<ll>> st(J, vector<ll>(M));
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++)
            st[j][o] = ouf.readLong(0LL, HORIZON, "start");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // feasibility 1: route precedence
    for (int j = 0; j < J; j++)
        for (int o = 1; o < M; o++)
            if (st[j][o] < st[j][o - 1] + dur[j][o - 1])
                quitf(_wa, "order %d step %d starts at %lld but previous step ends at %lld (precedence)",
                      j, o, st[j][o], st[j][o - 1] + dur[j][o - 1]);

    // feasibility 2: bay exclusivity (no overlap per machine)
    // gather intervals per bay, sort by start, check adjacent
    vector<vector<pair<ll,ll>>> byBay(M + 1); // (start, end)
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++)
            byBay[bay[j][o]].push_back({st[j][o], st[j][o] + dur[j][o]});
    for (int m = 1; m <= M; m++) {
        auto& v = byBay[m];
        sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); i++)
            if (v[i].first < v[i - 1].second)
                quitf(_wa, "bay %d has overlapping operations: [.., %lld) vs [%lld, ..)",
                      m, v[i - 1].second, v[i].first);
    }

    // objective: makespan
    ll F = 0;
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++)
            F = max(F, st[j][o] + dur[j][o]);
    if (F <= 0) quitf(_wa, "degenerate makespan");

    ll B = T;
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
