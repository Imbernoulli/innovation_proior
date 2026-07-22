#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Seating the Feuding Clans".
//
// Input:
//   N K M
//   phase_1 .. phase_N            (clan of each guest, 1..K)
//   M lines: u v c                (an exchange between guests u,v, count c)
//
// Output: a permutation of {1..N} -- the guests in seating order along the table.
//
// Objective (MIN), given seat positions pos[g] (1-based index in the printed order):
//   TRANS  = sum over the M exchange records of  c * bucket(|pos[u]-pos[v]|)
//     bucket(1)=0, bucket(2..3)=1, bucket(4..7)=3, bucket(>=8)=6
//   TAX    = CHANGE_TAX for every adjacent seat pair whose two guests are in
//            different clans
//   F = TRANS + TAX
//
// Baseline B (checker-computed): F evaluated on the IDENTITY order 1,2,...,N
//   -- exactly what solutions/trivial.cpp prints, so ratio = 0.1 there.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int CHANGE_TAX = 60;

static int bucket(int d) {
    if (d <= 1) return 0;
    if (d <= 3) return 1;
    if (d <= 7) return 3;
    return 6;
}

int N, K, M;
vector<int> phase;          // 1..N
vector<array<int,3>> exch;  // {u, v, c}

ll evalOrder(const vector<int>& order) {
    vector<int> pos(N + 1, 0);
    for (int i = 0; i < N; i++) pos[order[i]] = i + 1;
    ll F = 0;
    for (auto& e : exch) {
        int u = e[0], v = e[1], c = e[2];
        int d = abs(pos[u] - pos[v]);
        F += (ll)c * bucket(d);
    }
    for (int i = 0; i + 1 < N; i++) {
        if (phase[order[i]] != phase[order[i + 1]]) F += CHANGE_TAX;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    K = inf.readInt();
    M = inf.readInt();

    phase.assign(N + 1, 0);
    for (int g = 1; g <= N; g++) phase[g] = inf.readInt(1, K, "phase");

    exch.assign(M, {0, 0, 0});
    for (int i = 0; i < M; i++) {
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        int c = inf.readInt(1, 1000000, "c");
        exch[i] = {u, v, c};
    }

    // ---- read participant output: permutation of {1..N} ----
    vector<char> seen(N + 1, 0);
    vector<int> order(N);
    for (int i = 0; i < N; i++) {
        int x = ouf.readInt(1, N, "guest");
        if (seen[x]) quitf(_wa, "guest %d printed more than once", x);
        seen[x] = 1;
        order[i] = x;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the seating order");

    ll F = evalOrder(order);

    // ---- internal baseline: identity order 1,2,...,N ----
    vector<int> idOrder(N);
    for (int i = 0; i < N; i++) idOrder[i] = i + 1;
    ll B = evalOrder(idOrder);
    // Defensive floor per contract (B must be positive); unreachable in practice
    // here since gen.cpp Fisher-Yates-shuffles the clan assignment across guest
    // ids, so the identity order essentially never has zero changeovers/bucket cost.
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
