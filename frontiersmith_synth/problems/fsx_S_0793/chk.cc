#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Grow the Glyph"  (family: reverse-engineered-ca-seeder)
//
// Input:  N T B M   then one line: the length-N target string ('0'/'1').
//   N cells on a ring; radius-2 linear (XOR) rule with 5-bit mask M (bit (d+2)
//   set means offset d in {-2,-1,0,1,2} feeds the update); T synchronous steps.
// Output: k  (0<=k<=B)  then k DISTINCT cell indices in [0,N-1] to seed as live
//   (all other cells start at 0).
//
// Objective (MAX): F = number of cells matching the target after evolving the
//   seed forward T steps.
// Baseline B_base (checker-computed, "do nothing"): the all-zero seed is a fixed
//   point of any XOR rule (XOR of zeros is zero), so it stays all-zero forever;
//   its match count is simply the number of '0' cells in the target. This is the
//   trivial reference (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B_base)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt(1, 100000, "N");
    int T = inf.readInt(0, 100000, "T");
    int Bbudget = inf.readInt(0, N, "B");
    int M = inf.readInt(1, 31, "M");
    string target = inf.readToken();
    if ((int)target.size() != N) quitf(_fail, "generator bug: |target| != N");
    for (char c : target) if (c != '0' && c != '1') quitf(_fail, "generator bug: bad target char");

    // ---- internal baseline: all-zero seed stays all-zero (fixed point of XOR) ----
    int Bbase = 0;
    for (int i = 0; i < N; i++) if (target[i] == '0') Bbase++;
    if (Bbase <= 0) Bbase = 1;   // generator guarantees Bbase>0; safety only

    // ---- read participant seed ----
    int k = ouf.readInt(0, Bbudget, "k");
    vector<char> seen(N, 0);
    vector<char> state(N, 0);
    for (int i = 0; i < k; i++){
        int p = ouf.readInt(0, N - 1, "pos");
        if (seen[p]) quitf(_wa, "position %d listed more than once", p);
        seen[p] = 1;
        state[p] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- evolve forward T steps under the radius-2 XOR rule ----
    vector<char> cur = state, nxt(N);
    for (int step = 0; step < T; step++){
        for (int i = 0; i < N; i++){
            int v = 0;
            for (int d = -2; d <= 2; d++){
                if (M & (1 << (d + 2))){
                    int j = ((i + d) % N + N) % N;
                    v ^= cur[j];
                }
            }
            nxt[i] = (char)v;
        }
        swap(cur, nxt);
    }

    int F = 0;
    for (int i = 0; i < N; i++) if (cur[i] == (target[i] == '1')) F++;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1, Bbase));
    quitp(sc / 1000.0, "OK F=%d Bbase=%d k=%d/%d Ratio: %.6f", F, Bbase, k, Bbudget, sc / 1000.0);
    return 0;
}
