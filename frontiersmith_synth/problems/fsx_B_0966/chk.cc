#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Furnace Retained-Heat Batch Chains".
//
// Input:
//   n V LAMBDA
//   DECAY R0
//   C_HEAT C_COOL T_HEAT T_COOL BASE_TIME PROC_PER_UNIT
//   n lines: T_i s_i d_i w_i   (1-indexed job i)
//
// Output: m, then m lines "c  idx_1 .. idx_c" (batch k's job indices), batches
// listed in PROCESSING ORDER.
//
// For batch k (1-indexed), Temp_k = max T_i over its jobs. Let PrevTemp = R0 if
// k==1 else Temp_{k-1}. Between batches the furnace loses a FIXED amount of
// heat (idle radiative loss), independent of how hot it was:
//   R_k = max(0, PrevTemp - DECAY)
// If Temp_k >= R_k (heating / level -- top up what leaked away):
//   cost_k = C_HEAT*(Temp_k - R_k),  heat_time_k = T_HEAT*(Temp_k - R_k)
// else (Temp_k < R_k: still too hot from last time, must be actively FORCED
// down to the lower target -- slower and pricier per degree):
//   cost_k = C_COOL*(R_k - Temp_k),  heat_time_k = T_COOL*(R_k - Temp_k)
// with C_COOL > C_HEAT and T_COOL > T_HEAT. Because the loss is a FIXED amount
// rather than a fraction, an ascending or flat run of batches pays only
// O(DECAY) per step regardless of how high the temperatures are, while a
// single big descent pays for the whole drop at the costlier cooling rate --
// this is what makes the schedule's cost DIRECTIONAL.
// Batch duration dur_k = BASE_TIME + PROC_PER_UNIT*size_k + heat_time_k, where
// size_k = sum of s_i in batch k. Completion time C_k = C_{k-1} + dur_k (C_0=0).
// Every job i in batch k finishes at C_k; tardiness_i = max(0, C_k - d_i).
//
// Objective (MIN): F = sum_k cost_k + LAMBDA * sum_i w_i * tardiness_i.
//
// Baseline B (checker-computed): the "do nothing clever" reference -- naive
// sequential first-fit packing in raw job-index order (ignores both
// temperature and due dates), batches processed in that same formation order.
// This is exactly what solutions/trivial.cpp reproduces (-> ratio 0.1). B is
// usually strictly positive, but a degenerate legal instance (e.g. a single
// job whose temperature exactly equals the starting retained heat and whose
// due date never binds) can compute B=0; max(1,B) floors that case so the
// ratio formula stays well-defined (this is a REAL branch, not a no-op).
// Score (min): sc = min(1000, 100*max(1,B)/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

struct Job { ll T, s, d, w; };

static ll heatCost(ll prevTemp, ll temp, ll decay, ll cheat, ll ccool, ll *outTime, ll theat, ll tcool) {
    ll R = max((ll)0, prevTemp - decay);
    if (temp >= R) { *outTime = theat * (temp - R); return cheat * (temp - R); }
    else           { *outTime = tcool * (R - temp); return ccool * (R - temp); }
}

// Given a sequence of batches (each a vector of job indices, 1-indexed into `jobs`),
// compute the objective F exactly per the spec above.
static ll evalSchedule(const vector<vector<int>> &batches, const vector<Job> &jobs,
                        ll decay, ll R0, ll cheat, ll ccool, ll theat, ll tcool,
                        ll baseTime, ll procPerUnit, ll lambda) {
    ll F = 0, prevTemp = R0, Cprev = 0;
    for (auto &b : batches) {
        ll temp = 0, size = 0;
        for (int idx : b) { temp = max(temp, jobs[idx].T); size += jobs[idx].s; }
        ll htime = 0;
        ll cost = heatCost(prevTemp, temp, decay, cheat, ccool, &htime, theat, tcool);
        F += cost;
        ll dur = baseTime + procPerUnit * size + htime;
        ll Ck = Cprev + dur;
        for (int idx : b) {
            ll tardi = Ck - jobs[idx].d;
            if (tardi > 0) F += lambda * jobs[idx].w * tardi;
        }
        Cprev = Ck;
        prevTemp = temp;
    }
    return F;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    ll V = inf.readLong();
    ll LAMBDA = inf.readLong();
    ll DECAY = inf.readLong(), R0 = inf.readLong();
    ll CHEAT = inf.readLong(), CCOOL = inf.readLong();
    ll THEAT = inf.readLong(), TCOOL = inf.readLong();
    ll BASE_TIME = inf.readLong(), PROC_PER_UNIT = inf.readLong();

    vector<Job> jobs(n + 1);   // 1-indexed
    for (int i = 1; i <= n; i++) {
        jobs[i].T = inf.readLong();
        jobs[i].s = inf.readLong();
        jobs[i].d = inf.readLong();
        jobs[i].w = inf.readLong();
    }

    // ---- internal baseline B: naive sequential first-fit in raw index order,
    // ignoring both temperature and due dates (a line worker just loads the
    // furnace as jobs arrive, up to capacity, and fires it as soon as full) ----
    vector<vector<int>> trivBatches;
    {
        vector<int> cur; ll curSize = 0;
        for (int i = 1; i <= n; i++) {
            if (curSize + jobs[i].s > V && !cur.empty()) { trivBatches.push_back(cur); cur.clear(); curSize = 0; }
            cur.push_back(i); curSize += jobs[i].s;
        }
        if (!cur.empty()) trivBatches.push_back(cur);
    }
    ll B = evalSchedule(trivBatches, jobs, DECAY, R0, CHEAT, CCOOL, THEAT, TCOOL,
                         BASE_TIME, PROC_PER_UNIT, LAMBDA);
    if (B <= 0) B = 1;   // B can legitimately be 0 in a degenerate instance; floor it

    // ---- read + validate participant's batches ----
    int m = ouf.readInt(1, n, "num_batches");
    vector<vector<int>> batches(m);
    vector<char> seen(n + 1, 0);
    int covered = 0;
    for (int k = 0; k < m; k++) {
        int c = ouf.readInt(1, n, "batch_size");
        ll size = 0;
        for (int t = 0; t < c; t++) {
            int idx = ouf.readInt(1, n, "job_index");
            if (seen[idx]) quitf(_wa, "job %d assigned to more than one batch", idx);
            seen[idx] = 1;
            covered++;
            size += jobs[idx].s;
            batches[k].push_back(idx);
        }
        if (size > V) quitf(_wa, "batch %d total size %lld exceeds capacity V=%lld", k + 1, size, V);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the batch list");
    if (covered != n) quitf(_wa, "only %d of %d jobs covered by the batch partition", covered, n);

    ll F = evalSchedule(batches, jobs, DECAY, R0, CHEAT, CCOOL, THEAT, TCOOL,
                         BASE_TIME, PROC_PER_UNIT, LAMBDA);
    if (F < 0) quitf(_wa, "objective computed negative (should never happen)");

    double sc = min(1000.0, 100.0 * (double)max((ll)1, B) / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
