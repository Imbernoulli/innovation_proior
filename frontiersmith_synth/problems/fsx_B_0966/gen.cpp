#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Furnace Retained-Heat Batch Chains"   family: retained-heat-batch-chains
//
// n jobs, each (T soak temperature, s size, d due date, w tardiness weight).
// Furnace capacity V. Physics constants DECAY/R0 (fixed per-batch idle heat
// loss + start temp) and C_HEAT/C_COOL/T_HEAT/T_COOL/BASE_TIME/PROC_PER_UNIT
// are FIXED across all tests (same physics, different job mixes) except V and
// LAMBDA which vary per test to reshape the composition/capacity and
// energy/tardiness trade-off.
//
// Retained heat entering a batch is R = max(0, PrevTemp - DECAY): a FIXED
// amount leaks away regardless of temperature level, so reaching any target
// AT OR ABOVE R costs only C_HEAT*(gap-to-R) -- an ascending or flat run of
// batches pays a small roughly-constant amount per step. Forcing the furnace
// DOWN below R costs C_COOL*(R-target), pricier per degree. With
// R1 = max(0, R0-DECAY), the minimum possible completion time for a job of
// temperature T processed ALONE FIRST is BASE_TIME + PROC_PER_UNIT*s +
// (T-R1)*T_HEAT for T>=R1 -- i.e. roughly T-86 with this file's constants
// (R1=105, BASE_TIME=15, T_HEAT=1, PROC_PER_UNIT=1, s=4).
//
// Design: most jobs get a huge due date (never binding); a chosen subset of
// "urgent" jobs get a due date calibrated to be reachable ONLY if pulled out
// and processed near the front (using the formula above with slack), while a
// pure ascending-temperature schedule (the "cluster by similar temperature,
// then sort ascending" textbook rule) defers the hottest jobs -- exactly the
// urgent ones -- to the very end, blowing every one of those deadlines. A
// due-date-aware chain-braiding schedule pulls the urgent batch(es) out of
// order, eating one deliberate reset, to hit them.
// -----------------------------------------------------------------------------

struct Job { ll T, s, d, w; };

static void addJob(vector<Job> &jobs, ll T, ll s, ll d, ll w) {
    jobs.push_back({T, s, d, w});
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- fixed furnace physics (identical across all 10 tests) ----
    const ll DECAY = 45;                         // fixed idle heat loss between batches
    const ll R0 = 150;                           // starting retained heat (R1 = 105)
    const ll C_HEAT = 3, C_COOL = 9;              // cooling costs 3x more energy/degree
    const ll T_HEAT = 1, T_COOL = 3;              // and 3x more time/degree
    const ll BASE_TIME = 15, PROC_PER_UNIT = 1;   // fixed load/unload + size-proportional time
    // rescue(T) := minimum completion time if a size-4 job of temperature T
    // (T>=105) is processed ALONE as the very first batch = T - 86.

    vector<Job> jobs;
    ll V = 24, LAMBDA = 1;
    const ll LOOSE_D = 200000LL;   // matches the statement's d_i upper bound; still never binding

    auto bulkCluster = [&](int cnt, ll temp, ll tempJitter, ll szLo, ll szHi) {
        for (int i = 0; i < cnt; i++) {
            ll T = temp + rnd.next((int)(-tempJitter), (int)tempJitter);
            if (T < 10) T = 10;
            ll s = rnd.next((int)szLo, (int)szHi);
            addJob(jobs, T, s, LOOSE_D, 1 + rnd.next(0, 4));
        }
    };
    auto urgentJob = [&](ll temp, ll s, ll due, ll w) {
        addJob(jobs, temp, s, due, w);
    };
    // rescue-calibrated due date: reachable if pulled to the front (with slack),
    // unreachable once a real bulk of other batches precedes it.
    auto rescueDue = [&](ll temp, ll slack) { return temp - 86 + slack; };

    if (testId == 1) {
        // tiny sanity / example case: 3 clean ascending clusters, no deadline pressure.
        V = 20; LAMBDA = 1;
        bulkCluster(2, 130, 5, 3, 5);
        bulkCluster(2, 260, 5, 3, 5);
        bulkCluster(2, 420, 5, 3, 5);
    } else if (testId == 2) {
        // small, mild trap: one high-temp urgent job amid a low-temp fleet.
        V = 22; LAMBDA = 1;
        bulkCluster(10, 140, 20, 2, 6);
        urgentJob(220, 4, rescueDue(220, 20), 16);
    } else if (testId == 3) {
        // planted: clean ascending cluster ladder, no conflicts -> rewards ascending
        // order but strong composition still edges out greedy via tighter packing.
        V = 24; LAMBDA = 1;
        bulkCluster(4, 130, 8, 2, 8);
        bulkCluster(4, 220, 8, 2, 8);
        bulkCluster(4, 350, 8, 2, 8);
        bulkCluster(4, 480, 8, 2, 8);
    } else if (testId == 4) {
        // trap: several HIGH-temp urgent jobs stuck deep in a LOW-temp fleet.
        V = 22; LAMBDA = 1;
        bulkCluster(18, 130, 25, 2, 6);
        urgentJob(420, 3, rescueDue(420, 55), 14);
        urgentJob(440, 3, rescueDue(440, 70), 15);
        urgentJob(400, 4, rescueDue(400, 85), 13);
    } else if (testId == 5) {
        // trap, double-rescue: TWO separate high-temp urgent jobs (staggered due
        // dates) both buried at the ascending-hot end of a low-temp fleet -- a
        // pure ascending sweep defers BOTH to the very end and blows both
        // deadlines; the insight is to braid a short ascending rescue chain
        // (430 -> 480, itself cheap since ascending) up front, then resume the
        // low bulk (one deliberate reset back down).
        V = 22; LAMBDA = 1;
        bulkCluster(20, 150, 25, 2, 6);
        urgentJob(430, 4, rescueDue(430, 55), 14);
        urgentJob(480, 4, rescueDue(480, 90), 15);
    } else if (testId == 6) {
        // needle + mild trap: a nearly-free perfect ascending chain hidden
        // among random-temp noise (capacity is tight so which jobs share a
        // batch decides whether the chain survives intact), plus one urgent
        // hot job so a pure temperature sort (which ties with due-date order
        // whenever every due date is loose) is forced to diverge from a
        // deadline-aware schedule.
        V = 18; LAMBDA = 1;
        ll t = 130;
        for (int i = 0; i < 14; i++) {
            addJob(jobs, t, 3 + rnd.next(0, 3), LOOSE_D, 1 + rnd.next(0, 3));
            t += 12 + rnd.next(0, 6);             // small ascending steps: nearly free
        }
        for (int i = 0; i < 20; i++) {
            ll T = 90 + rnd.next(0, 600);          // scattered noise temperatures
            ll s = rnd.next(2, 6);
            addJob(jobs, T, s, LOOSE_D, 1 + rnd.next(0, 4));
        }
        urgentJob(340, 4, rescueDue(340, 40), 14);
    } else if (testId == 7) {
        // bigger trap combo: two urgent pockets, both hot, staggered urgency.
        V = 24; LAMBDA = 1;
        bulkCluster(30, 190, 30, 2, 7);
        bulkCluster(14, 460, 30, 2, 7);
        for (int i = 0; i < 3; i++) urgentJob(640 - i * 10, 3, rescueDue(640 - i * 10, 60 + i * 45), 5 + i);
    } else if (testId == 8) {
        // large mixed regime: several clusters + noise + a moderate urgent pocket.
        V = 26; LAMBDA = 1;
        bulkCluster(22, 160, 20, 2, 8);
        bulkCluster(22, 300, 20, 2, 8);
        bulkCluster(18, 450, 20, 2, 8);
        for (int i = 0; i < 22; i++) {
            ll T = 110 + rnd.next(0, 560);
            addJob(jobs, T, rnd.next(2, 8), LOOSE_D, 1 + rnd.next(0, 5));
        }
        for (int i = 0; i < 4; i++) urgentJob(560 + i * 8, 3, rescueDue(560 + i * 8, 60 + i * 40), 5);
    } else if (testId == 9) {
        // adversarial: nearly all-distinct temperatures (no clean clusters), tight
        // capacity forces genuine composition decisions; moderate urgency mix.
        V = 20; LAMBDA = 1;
        for (int i = 0; i < 90; i++) {
            ll T = 110 + rnd.next(0, 700);
            addJob(jobs, T, rnd.next(2, 7), LOOSE_D, 1 + rnd.next(0, 5));
        }
        for (int i = 0; i < 3; i++) {
            ll T = 650 + rnd.next(0, 30);
            urgentJob(T, 3, rescueDue(T, 60 + i * 40), 5 + rnd.next(0, 3));
        }
    } else {
        // testId == 10: max stress, fill the envelope, combine every trap flavor.
        V = 22; LAMBDA = 1;
        bulkCluster(38, 170, 25, 2, 7);
        bulkCluster(26, 340, 25, 2, 7);
        bulkCluster(20, 500, 25, 2, 7);
        for (int i = 0; i < 20; i++) {
            ll T = 110 + rnd.next(0, 650);
            addJob(jobs, T, rnd.next(2, 7), LOOSE_D, 1 + rnd.next(0, 5));
        }
        for (int i = 0; i < 4; i++) {
            ll T = 640 + rnd.next(0, 30);
            urgentJob(T, 3 + rnd.next(0, 2), rescueDue(T, 55 + i * 35), 5 + rnd.next(0, 4));
        }
    }

    // shuffle emitted order so job index carries no structural signal
    for (int i = (int)jobs.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(jobs[i], jobs[j]);
    }

    int n = (int)jobs.size();
    printf("%d %lld %lld\n", n, V, LAMBDA);
    printf("%lld %lld\n", DECAY, R0);
    printf("%lld %lld %lld %lld %lld %lld\n", C_HEAT, C_COOL, T_HEAT, T_COOL, BASE_TIME, PROC_PER_UNIT);
    for (auto &j : jobs) printf("%lld %lld %lld %lld\n", j.T, j.s, j.d, j.w);
    return 0;
}
