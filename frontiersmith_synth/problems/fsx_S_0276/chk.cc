#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Volcano Watch: Solar/Diesel Compute Replay for Sensor Backlogs".
// Minimization.  The participant prints exactly T lines "job_t mode_t".
// Feasibility (any violation -> score 0):
//   * job_t in [0,N], mode_t in [0,1];
//   * if job_t = j != 0 then r_j <= t <= d_j (task window);
//   * if an active step uses solar (mode 0) then a_t = 1 (solar-capable).
// Objective F = sum of chosen power cost over active steps
//             + sum of v_j over tasks with cnt_j < p_j (uncleared).
// Baseline B = do-nothing (pause every step) = sum_j v_j  (positive since v_j >= 1).
//   ratio = min(1, (B / max(1,F)) / 10).

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();
    int N = inf.readInt();
    vector<int> a(T + 1);
    vector<long long> cs(T + 1), cd(T + 1);
    for (int t = 1; t <= T; t++) {
        a[t]  = inf.readInt();
        cs[t] = inf.readInt();
        cd[t] = inf.readInt();
    }
    vector<int> r(N + 1), d(N + 1), p(N + 1);
    vector<long long> v(N + 1);
    for (int j = 1; j <= N; j++) {
        r[j] = inf.readInt();
        d[j] = inf.readInt();
        p[j] = inf.readInt();
        v[j] = inf.readInt();
    }

    // ---- read participant schedule: exactly T lines ----
    vector<long long> cnt(N + 1, 0);
    long long power = 0;
    for (int t = 1; t <= T; t++) {
        int job  = ouf.readInt(0, N, "job");
        int mode = ouf.readInt(0, 1, "mode");
        if (job != 0) {
            if (t < r[job] || t > d[job])
                quitf(_wa, "step %d assigns task %d outside its window [%d,%d]",
                      t, job, r[job], d[job]);
            if (mode == 0) {           // solar
                if (a[t] == 0)
                    quitf(_wa, "step %d uses solar but is not solar-capable (a_%d=0)", t, t);
                power += cs[t];
            } else {                   // diesel
                power += cd[t];
            }
            cnt[job]++;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d schedule lines", T);

    // ---- objective F ----
    long long penalty = 0;
    for (int j = 1; j <= N; j++)
        if (cnt[j] < (long long)p[j]) penalty += v[j];
    long long F = power + penalty;

    // ---- baseline B: pause everything ----
    long long B = 0;
    for (int j = 1; j <= N; j++) B += v[j];
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
