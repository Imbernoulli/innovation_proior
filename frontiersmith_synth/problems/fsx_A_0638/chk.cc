#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------------
// Checker / scorer for "Staggered Leaky Windbreaks Against the Steppe Gale".
//
// Grid: W columns (0..W-1, the wind's travel direction) x H lanes (0..H-1, lateral).
// Windbreak zone: columns 0..Wb-1, plantable lane band [Ymin,Ymax]. Crop cells (x,y)
// with x in [Wb,W-1] are listed explicitly. inflow[y] enters at column 0's "before"
// state. Every column: (1) add the pending relief pool from the previous column,
// (2) a rational crosswind drift rotates the whole lane vector by exactly one lane
// (cyclic, wrap) whenever the accumulator crosses 1, (3) if this column is inside the
// windbreak zone, each planted cell of porosity level p removes drag[p]/1000 of that
// lane's CURRENT (pre-column) speed; of the removed amount, jet[p]/1000 is thrown
// immediately onto the two nearest lanes (mirrored back at the band edges -> tip
// jets), the rest spreads over the 5 lanes centered here (weights 1:2:4:2:1) into the
// relief pool for the NEXT column. All removals/redistributions inside one column are
// computed from a single simultaneous snapshot (no ordering artifacts) and conserve
// the total exactly (mass can move between lanes/columns, never vanish or appear).
//
// Objective (MIN): F = sum over listed crop cells (x,y) of (lane speed at x,y)^2,
//   subject to: only cells in [0,Wb)x[Ymin,Ymax] may be nonzero, values in [0,Pmax],
//   and sum of cost[p] over planted cells <= K.
//
// Baseline B (checker-computed) = F of the all-zero (no trees planted) grid, i.e. the
//   "do nothing" construction -- exactly what solutions/trivial.cpp submits.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------------

static ll W, H, Wb, Pmax;
static ll Ymin, Ymax, K;
static vector<ll> inflow;
static ll driftNum, driftDen;
static vector<ll> cost, drag, jetfrac; // 1-indexed to Pmax
static int M;
static vector<vector<int>> cropsAtX; // cropsAtX[x] = list of lane y (may repeat)

static inline ll reflect(ll k, ll Hn){
    if (Hn <= 1) return 0;
    ll period = 2 * (Hn - 1);
    ll r = k % period; if (r < 0) r += period;
    if (r >= Hn) r = period - r;
    return r;
}

// distribute `amount` over targets[] with integer weights[] (sum=W), adding the
// (exactly-conserved) shares into dst[target]. Remainder units go to the targets
// with the largest weight, ties broken by earliest position, one unit at a time.
static void distribute(ll amount, const vector<ll>& targets, const vector<ll>& weights, vector<ll>& dst){
    ll wsum = 0; for (ll w : weights) wsum += w;
    if (wsum <= 0 || amount == 0) return;
    vector<ll> share(targets.size());
    ll used = 0;
    for (size_t i = 0; i < targets.size(); i++){
        share[i] = (amount * weights[i]) / wsum;
        used += share[i];
    }
    ll rem = amount - used;
    // order indices by weight desc, then position asc, hand out remainder units
    vector<int> order(targets.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (weights[a] != weights[b]) return weights[a] > weights[b];
        return a < b;
    });
    int oi = 0;
    while (rem > 0 && !order.empty()){
        share[order[oi % order.size()]] += 1;
        rem--; oi++;
    }
    for (size_t i = 0; i < targets.size(); i++) dst[targets[i]] += share[i];
}

// grid may be empty (nullptr) for the baseline "do nothing" run.
static ll simulate(const vector<vector<int>>* grid){
    vector<ll> s(H), pool(H, 0);
    for (ll y = 0; y < H; y++) s[y] = inflow[y];
    ll shiftAcc = 0;
    ll F = 0;
    for (ll x = 0; x < W; x++){
        for (ll y = 0; y < H; y++){ s[y] += pool[y]; pool[y] = 0; }
        shiftAcc += driftNum;
        if (shiftAcc >= driftDen){
            shiftAcc -= driftDen;
            vector<ll> ns(H);
            for (ll y = 0; y < H; y++) ns[y] = s[(y - 1 + H) % H];
            s.swap(ns);
        }
        if (x < Wb && grid != nullptr){
            vector<ll> s0 = s;
            vector<ll> delta(H, 0), poolNext(H, 0);
            for (ll y = Ymin; y <= Ymax; y++){
                int p = (*grid)[x][y];
                if (p <= 0) continue;
                ll removed = (s0[y] * drag[p]) / 1000;
                delta[y] -= removed;
                ll jetAmt = (removed * jetfrac[p]) / 1000;
                ll diffAmt = removed - jetAmt;
                vector<ll> jt = {reflect(y - 1, H), reflect(y + 1, H)};
                vector<ll> jw = {1, 1};
                distribute(jetAmt, jt, jw, delta);
                vector<ll> dt = {reflect(y - 2, H), reflect(y - 1, H), reflect(y, H), reflect(y + 1, H), reflect(y + 2, H)};
                vector<ll> dw = {1, 2, 4, 2, 1};
                distribute(diffAmt, dt, dw, poolNext);
            }
            for (ll y = 0; y < H; y++) s[y] = s0[y] + delta[y];
            pool = poolNext;
        }
        if (x >= Wb){
            for (int y : cropsAtX[x]) F += s[y] * s[y];
        }
    }
    return F;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    W = inf.readLong(); H = inf.readLong(); Wb = inf.readLong(); Pmax = inf.readLong();
    Ymin = inf.readLong(); Ymax = inf.readLong();
    K = inf.readLong();
    inflow.assign(H, 0);
    for (ll y = 0; y < H; y++) inflow[y] = inf.readLong();
    driftNum = inf.readLong(); driftDen = inf.readLong();
    cost.assign(Pmax + 1, 0); drag.assign(Pmax + 1, 0); jetfrac.assign(Pmax + 1, 0);
    for (ll p = 1; p <= Pmax; p++) cost[p] = inf.readLong();
    for (ll p = 1; p <= Pmax; p++) drag[p] = inf.readLong();
    for (ll p = 1; p <= Pmax; p++) jetfrac[p] = inf.readLong();
    M = inf.readInt();
    cropsAtX.assign(W, {});
    for (int i = 0; i < M; i++){
        ll x = inf.readLong(); ll y = inf.readLong();
        cropsAtX[x].push_back((int)y);
    }

    // ---- baseline B: do nothing ----
    ll B = simulate(nullptr);
    if (B <= 0) B = 1;

    // ---- read + validate participant grid ----
    vector<vector<int>> grid(Wb, vector<int>(H, 0));
    ll totalCost = 0;
    for (ll x = 0; x < Wb; x++){
        for (ll y = 0; y < H; y++){
            int p = ouf.readInt(0, (int)Pmax, "porosity_level");
            if (p > 0 && (y < Ymin || y > Ymax))
                quitf(_wa, "tree planted at (x=%lld,y=%lld) outside plantable band [%lld,%lld]", x, y, Ymin, Ymax);
            grid[x][y] = p;
            if (p > 0) totalCost += cost[p];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the grid");
    if (totalCost > K) quitf(_wa, "budget exceeded: used %lld > K=%lld", totalCost, K);

    ll F = simulate(&grid);
    if (F < 0) quitf(_wa, "internal objective negative (should be impossible)");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
