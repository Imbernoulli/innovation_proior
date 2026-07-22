#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker/scorer for "Move the Toll Booths, Shrink the Engines".
//
// Input:  N K T ; L_1..L_K ; N station menus (Kv, then Kv (delay,area) pairs) ;
//         M=N-K belts (w c), grouped by pipeline in belt order.
// Output (participant): line1 = N ints x_v (chosen variant per station);
//         line2 = M ints w'_e (new booth count per belt, same order as input).
//
// Feasibility: x_v in [0,Kv-1]; per pipeline, new booth counts are >=0 and sum to
// the SAME total as that pipeline's original counts; walking each pipeline in
// order, a station right after the start or right after a belt with new count>=1
// starts a fresh combinational run, else its arrival = prev arrival + own delay;
// every station that is last-of-pipeline or precedes a belt with new count>=1
// must have arrival <= T.
//
// Objective (MIN): F = sum(area of chosen variant) + sum(new booth count * cost).
// Internal baseline B (checker-built): keep original booth counts, run every
// station at its priciest/fastest variant -- always feasible by construction.
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

struct Variant { ll d, a; };

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int K = inf.readInt();
    ll  T = inf.readLong();

    vector<int> lens(K);
    for (int p = 0; p < K; p++) lens[p] = inf.readInt();

    vector<int> Kv(N + 1);
    vector<array<Variant,3>> menu(N + 1);
    for (int v = 1; v <= N; v++) {
        int kv = inf.readInt();
        Kv[v] = kv;
        for (int i = 0; i < kv; i++) {
            ll d = inf.readLong(), a = inf.readLong();
            menu[v][i] = {d, a};
        }
    }

    vector<int> off(K);
    { int cur = 0; for (int p = 0; p < K; p++) { off[p] = cur; cur += lens[p]; } }
    int M = N - K;

    vector<vector<ll>> w0(K), cost(K);
    vector<int> beltPipeOf, beltLocalOf;
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        w0[p].assign(L - 1, 0);
        cost[p].assign(L - 1, 0);
        for (int i = 0; i < L - 1; i++) {
            w0[p][i] = inf.readLong();
            cost[p][i] = inf.readLong();
        }
    }

    // ---- internal baseline B: original booth counts, fastest variant everywhere ----
    ll B = 0;
    for (int v = 1; v <= N; v++) B += menu[v][Kv[v] - 1].a;
    for (int p = 0; p < K; p++) for (int i = 0; i < (int)w0[p].size(); i++) B += w0[p][i] * cost[p][i];
    if (B <= 0) B = 1;

    // ---- read participant output ----
    vector<int> x(N + 1);
    for (int v = 1; v <= N; v++) x[v] = ouf.readInt(0, Kv[v] - 1, "variant");

    vector<vector<ll>> wp(K);
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        wp[p].assign(L - 1, 0);
        for (int i = 0; i < L - 1; i++) wp[p][i] = ouf.readLong(0LL, (ll)2e9, "booth_count");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- conservation per pipeline ----
    for (int p = 0; p < K; p++) {
        ll s0 = 0, s1 = 0;
        for (int i = 0; i < (int)w0[p].size(); i++) { s0 += w0[p][i]; s1 += wp[p][i]; }
        if (s0 != s1)
            quitf(_wa, "pipeline %d: booth total changed (%lld -> %lld)", p + 1, s0, s1);
    }

    // ---- timing feasibility + objective F ----
    ll F = 0;
    for (int v = 1; v <= N; v++) F += menu[v][x[v]].a;
    for (int p = 0; p < K; p++) for (int i = 0; i < (int)wp[p].size(); i++) F += wp[p][i] * cost[p][i];

    for (int p = 0; p < K; p++) {
        int L = lens[p];
        ll run = 0;
        for (int i = 0; i < L; i++) {
            int v = off[p] + i + 1;
            ll own = menu[v][x[v]].d;
            bool freshRun = (i == 0) || (wp[p][i - 1] >= 1);
            run = freshRun ? own : (run + own);
            bool checkpoint = (i == L - 1) || (i < L - 1 && wp[p][i] >= 1);
            if (checkpoint && run > T)
                quitf(_wa, "pipeline %d station-local %d: arrival %lld > T=%lld", p + 1, i + 1, run, T);
        }
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld M=%d Ratio: %.6f", F, B, M, sc / 1000.0);
    return 0;
}
