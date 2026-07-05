#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();

    vector<vector<int>> mach(N, vector<int>(M)), dur(N, vector<int>(M));
    long long B = 0;   // serial baseline makespan = total work
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mach[i][j] = inf.readInt();
            dur[i][j] = inf.readInt();
            B += dur[i][j];
        }
    }

    // Read participant schedule: N lines, each M integer start times (read flat).
    const long long HI = (long long)1e15;
    vector<vector<long long>> st(N, vector<long long>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            st[i][j] = ouf.readLong(0LL, HI, "start");
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d*%d start times", N, M);

    // Precedence within each loop.
    for (int i = 0; i < N; i++) {
        for (int j = 1; j < M; j++) {
            long long prevEnd = st[i][j - 1] + dur[i][j - 1];
            if (st[i][j] < prevEnd) {
                quitf(_wa, "precedence violated loop %d stage %d: start %lld < %lld",
                      i + 1, j + 1, st[i][j], prevEnd);
            }
        }
    }

    // No-overlap per cooling unit.
    // Collect (start,end) intervals per machine, sort, check pairwise disjoint.
    vector<vector<pair<long long,long long>>> perU(M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int u = mach[i][j];
            perU[u].push_back({st[i][j], st[i][j] + dur[i][j]});
        }
    }
    for (int u = 0; u < M; u++) {
        auto& v = perU[u];
        sort(v.begin(), v.end());
        for (size_t p = 1; p < v.size(); p++) {
            if (v[p].first < v[p - 1].second) {
                quitf(_wa, "unit %d overlap: interval [%lld,%lld) starts before previous ends %lld",
                      u, v[p].first, v[p].second, v[p - 1].second);
            }
        }
    }

    // Objective: makespan.
    long long F = 0;
    for (int i = 0; i < N; i++) {
        long long e = st[i][M - 1] + dur[i][M - 1];
        if (e > F) F = e;
    }

    if (B <= 0) quitf(_fail, "internal baseline non-positive");
    if (F <= 0) quitf(_wa, "zero makespan cannot be feasible");

    double sc_score = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc_score / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc_score / 1000.0);
    return 0;
}
