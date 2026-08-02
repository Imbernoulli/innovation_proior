#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Pairing Guard on a Contested Board".
//
// Input : n m K ; zone[0..n-1] ; m lines "s w c_1..c_s" ; a length-n permutation
//         `order` -- the Aggressor's fixed, precomputed claim-attempt order.
// Output: p ; then p lines "a b" -- disjoint cell couples the Guard commits to
//         (0<=p<=n/2, each cell in at most one pair, sum(1+|zone_a-zone_b|)<=K).
//
// Simulation (deterministic, replayed identically for the baseline and for the
// participant's pairing): n turns, alternating. Aggressor's turn: scan `order`
// from where it left off, skip already-claimed cells, claim the first free one.
// Guard's turn: if the cell the Aggressor just took has a live partner under the
// committed pairing, claim that partner; otherwise claim the smallest-indexed
// still-free cell (uninformed fallback). A line is SAFE iff the Guard owns at
// least one of its cells when every cell is claimed.
//
// Objective (MAX): F = total weight of SAFE lines.
// Baseline B (checker-computed): F under the EMPTY pairing (purely reactive,
//   uninformed defense) -- exactly what the trivial reference submits.
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int n, m;
static vector<int> zone;
static vector<vector<int>> lineCells;
static vector<ll> lineW;
static vector<int> order;

// Replays the whole game for a given (possibly empty) pairing and returns F.
static ll simulate(const vector<int> &partner) {
    vector<int> owner(n, -1);   // -1 unclaimed, 0 Aggressor, 1 Guard
    int claimed = 0, ptr = 0, low = 0;
    while (claimed < n) {
        // Aggressor's turn
        while (ptr < n && owner[order[ptr]] != -1) ptr++;
        int x = order[ptr];
        owner[x] = 0;
        claimed++;
        if (claimed == n) break;
        // Guard's turn
        int y = -1;
        if (partner[x] != -1 && owner[partner[x]] == -1) y = partner[x];
        else {
            while (low < n && owner[low] != -1) low++;
            y = low;
        }
        owner[y] = 1;
        claimed++;
    }
    ll F = 0;
    for (int i = 0; i < m; i++) {
        bool safe = false;
        for (int c : lineCells[i]) if (owner[c] == 1) { safe = true; break; }
        if (safe) F += lineW[i];
    }
    return F;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    ll K = inf.readLong();
    zone.resize(n);
    for (int i = 0; i < n; i++) zone[i] = inf.readInt(0, 9, "zone");
    lineCells.assign(m, {});
    lineW.assign(m, 0);
    for (int i = 0; i < m; i++) {
        int s = inf.readInt(2, 6, "line_size");
        ll w = inf.readLong();
        lineCells[i].resize(s);
        for (int j = 0; j < s; j++) lineCells[i][j] = inf.readInt(0, n - 1, "line_cell");
        lineW[i] = w;
    }
    order.resize(n);
    vector<char> seen(n, 0);
    for (int i = 0; i < n; i++) {
        order[i] = inf.readInt(0, n - 1, "order_cell");
        if (seen[order[i]]) quitf(_fail, "malformed input: order is not a permutation");
        seen[order[i]] = 1;
    }

    // ---- internal baseline B: empty pairing ----
    vector<int> emptyPartner(n, -1);
    ll B = simulate(emptyPartner);
    if (B <= 0) B = 1;

    // ---- read & validate the participant's pairing ----
    int p = ouf.readInt(0, n / 2, "num_pairs");
    vector<int> partner(n, -1);
    vector<char> used(n, 0);
    ll cost = 0;
    for (int i = 0; i < p; i++) {
        int a = ouf.readInt(0, n - 1, "cell_a");
        int b = ouf.readInt(0, n - 1, "cell_b");
        if (a == b) quitf(_wa, "pair %d: a == b (%d)", i, a);
        if (used[a]) quitf(_wa, "cell %d used in more than one pair", a);
        if (used[b]) quitf(_wa, "cell %d used in more than one pair", b);
        used[a] = used[b] = 1;
        partner[a] = b;
        partner[b] = a;
        cost += 1 + (ll)abs(zone[a] - zone[b]);
    }
    if (cost > K) quitf(_wa, "pairing budget exceeded: spent %lld > K=%lld", cost, K);
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the pairing");

    // ---- score the participant's pairing under the fixed replay ----
    ll F = simulate(partner);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld K=%lld spent=%lld Ratio: %.6f",
          F, B, K, cost, sc / 1000.0);
    return 0;
}
