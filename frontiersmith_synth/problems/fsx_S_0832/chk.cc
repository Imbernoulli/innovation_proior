#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Wire Power to Comm So Both Survive".
// Maximization.  Validates the participant's coupling is a permutation of 1..N.
// Runs the interdependent cascade (percolation-to-giant-component in each layer,
// alternating with dependency propagation across the coupling) to a fixed point,
// for each of the K attack scenarios given in the input, using the SAME coupling.
// F = total surviving mutually-connected nodes summed over scenarios (maximize).
// B = the same total for the identity coupling (P-node i <-> C-node i), the
// always-feasible "do nothing clever" baseline.
//   ratio = min(1, (F / max(1,B)) / 10)

static int N;
static vector<vector<int>> adjP, adjC;
static vector<vector<int>> attacks;

// Run the cascade to a fixed point for one attack scenario and one coupling
// (matchPtoC[i] = the C-node id, 0-indexed, coupled to P-node i).  Returns the
// number of P-nodes alive (== number of mutually-alive coupled pairs) at the
// fixed point.
static long long runCascade(const vector<int>& attacked, const vector<int>& matchPtoC) {
    vector<char> aliveP(N, 1), aliveC(N, 1);
    for (int v : attacked) aliveP[v] = 0;
    vector<int> matchCtoP(N);
    for (int i = 0; i < N; i++) matchCtoP[matchPtoC[i]] = i;

    vector<int> comp(N);
    int cap = N + 5;   // alive-set is monotonically non-increasing -> <=N changed rounds needed
    for (int round = 0; round < cap; round++) {
        bool changed = false;

        // ---- P giant component among currently-alive P nodes ----
        fill(comp.begin(), comp.end(), -1);
        vector<int> sizeOf;
        for (int s = 0; s < N; s++) {
            if (!aliveP[s] || comp[s] != -1) continue;
            int cid = (int)sizeOf.size();
            int sz = 0;
            vector<int> stack{s};
            comp[s] = cid;
            while (!stack.empty()) {
                int u = stack.back(); stack.pop_back(); sz++;
                for (int w : adjP[u]) if (aliveP[w] && comp[w] == -1) { comp[w] = cid; stack.push_back(w); }
            }
            sizeOf.push_back(sz);
        }
        int bestC = -1, bestSz = -1;
        for (int i = 0; i < (int)sizeOf.size(); i++) if (sizeOf[i] > bestSz) { bestSz = sizeOf[i]; bestC = i; }
        for (int v = 0; v < N; v++)
            if (aliveP[v] && comp[v] != bestC) { aliveP[v] = 0; changed = true; }

        // ---- propagate P deaths onto matched C partners ----
        for (int i = 0; i < N; i++)
            if (!aliveP[i] && aliveC[matchPtoC[i]]) { aliveC[matchPtoC[i]] = 0; changed = true; }

        // ---- C giant component among currently-alive C nodes ----
        fill(comp.begin(), comp.end(), -1);
        sizeOf.clear();
        for (int s = 0; s < N; s++) {
            if (!aliveC[s] || comp[s] != -1) continue;
            int cid = (int)sizeOf.size();
            int sz = 0;
            vector<int> stack{s};
            comp[s] = cid;
            while (!stack.empty()) {
                int u = stack.back(); stack.pop_back(); sz++;
                for (int w : adjC[u]) if (aliveC[w] && comp[w] == -1) { comp[w] = cid; stack.push_back(w); }
            }
            sizeOf.push_back(sz);
        }
        bestC = -1; bestSz = -1;
        for (int i = 0; i < (int)sizeOf.size(); i++) if (sizeOf[i] > bestSz) { bestSz = sizeOf[i]; bestC = i; }
        for (int v = 0; v < N; v++)
            if (aliveC[v] && comp[v] != bestC) { aliveC[v] = 0; changed = true; }

        // ---- propagate C deaths onto matched P partners ----
        for (int j = 0; j < N; j++)
            if (!aliveC[j] && aliveP[matchCtoP[j]]) { aliveP[matchCtoP[j]] = 0; changed = true; }

        if (!changed) break;
    }

    long long surv = 0;
    for (int i = 0; i < N; i++) if (aliveP[i]) surv++;
    return surv;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    int Mp = inf.readInt();
    int Mc = inf.readInt();
    adjP.assign(N, {});
    adjC.assign(N, {});
    for (int e = 0; e < Mp; e++) {
        int a = inf.readInt() - 1, b = inf.readInt() - 1;
        adjP[a].push_back(b); adjP[b].push_back(a);
    }
    for (int e = 0; e < Mc; e++) {
        int a = inf.readInt() - 1, b = inf.readInt() - 1;
        adjC[a].push_back(b); adjC[b].push_back(a);
    }
    int K = inf.readInt();
    attacks.assign(K, {});
    for (int t = 0; t < K; t++) {
        int A = inf.readInt();
        attacks[t].resize(A);
        for (int i = 0; i < A; i++) attacks[t][i] = inf.readInt() - 1;
    }

    // ---- read + validate participant coupling: must be a permutation of 1..N ----
    vector<int> matchPtoC(N);
    vector<char> used(N, 0);
    for (int i = 0; i < N; i++) {
        int c = ouf.readInt(1, N, "coupling target") - 1;
        if (used[c]) quitf(_wa, "C-node %d is coupled to more than one P-node", c + 1);
        used[c] = 1;
        matchPtoC[i] = c;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the N coupling targets");

    // ---- participant objective F: total survivors across all K scenarios ----
    long long F = 0;
    for (int t = 0; t < K; t++) F += runCascade(attacks[t], matchPtoC);

    // ---- baseline B: identity coupling (P-node i <-> C-node i) ----
    vector<int> identity(N);
    for (int i = 0; i < N; i++) identity[i] = i;
    long long B = 0;
    for (int t = 0; t < K; t++) B += runCascade(attacks[t], identity);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
