#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Halo Seed: Growing a Picture Backward Through a 2D
// Automaton". family: cellular-automaton-preimage
//
// Input:  N T alpha beta ; 32-char rule table (rule[v] for v=16*center+8*up+
//         4*down+2*left+1*right, rule[0]='0'); N rows of the target grid.
// Output: N rows of the participant's seed grid (initial state), same shape.
//
// Objective (MIN): evolve the seed T steps under the rule (toroidal wrap) to
//   get S; H = Hamming mismatch(S,target); L = live cells in the seed.
//   F = alpha*H + beta*L.
// Baseline B (checker-computed): the all-dead seed's cost, i.e.
//   B = alpha * popcount(target)  (since rule[0]=0 keeps it a fixed point,
//   H_baseline = popcount(target), L_baseline = 0). This is exactly what
//   the trivial reference reproduces (-> ratio 0.1).
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    ll alpha = inf.readLong();
    ll beta = inf.readLong();

    string ruleTok = inf.readToken();
    if ((int)ruleTok.size() != 32) quitf(_fail, "internal: bad rule length");
    char rule[32];
    for (int v = 0; v < 32; v++){
        if (ruleTok[v] != '0' && ruleTok[v] != '1') quitf(_fail, "internal: bad rule char");
        rule[v] = ruleTok[v];
    }

    vector<string> target(N);
    for (int i = 0; i < N; i++){
        string tok = inf.readToken();
        if ((int)tok.size() != N) quitf(_fail, "internal: bad target row length");
        target[i] = tok;
    }

    // ---- read + validate participant's seed grid ----
    vector<string> init(N);
    for (int i = 0; i < N; i++){
        string tok = ouf.readToken();
        if ((int)tok.size() != N)
            quitf(_wa, "row %d: expected %d characters, got %d", i, N, (int)tok.size());
        for (char c : tok)
            if (c != '0' && c != '1')
                quitf(_wa, "row %d: invalid character '%c' (must be 0/1)", i, (int)c);
        init[i] = tok;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the seed grid");

    // ---- forward-simulate T steps from the participant's seed ----
    vector<string> cur = init, nxt(N, string(N, '0'));
    for (int s = 0; s < T; s++){
        for (int i = 0; i < N; i++){
            int up = (i - 1 + N) % N, down = (i + 1) % N;
            for (int j = 0; j < N; j++){
                int left = (j - 1 + N) % N, right = (j + 1) % N;
                int c = cur[i][j] - '0';
                int u = cur[up][j] - '0';
                int d = cur[down][j] - '0';
                int l = cur[i][left] - '0';
                int r = cur[i][right] - '0';
                int v = (c << 4) | (u << 3) | (d << 2) | (l << 1) | r;
                nxt[i][j] = rule[v];
            }
        }
        swap(cur, nxt);
    }

    ll H = 0, L = 0, popTarget = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            if (cur[i][j] != target[i][j]) H++;
            if (init[i][j] == '1') L++;
            if (target[i][j] == '1') popTarget++;
        }

    ll F = alpha * H + beta * L;
    ll B = alpha * popTarget;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld H=%lld L=%lld Ratio: %.6f",
          F, B, H, L, sc / 1000.0);
    return 0;
}
