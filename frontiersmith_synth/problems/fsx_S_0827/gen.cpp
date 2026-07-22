#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Halo Seed" generator. family: cellular-automaton-preimage
//
// Emits: N T alpha beta / 32-char rule table / N rows of target (N-char strings).
//
// Rule encoding: v = 16*center + 8*up + 4*down + 2*left + 1*right (0..31);
// rule[0] is always forced to '0' (all-dead neighborhood stays dead), so the
// all-dead grid is always a fixed point and gives a well-defined baseline.
//
// kind (per test, the ladder below):
//   0 identity : rule[v] = center bit only -> exact, easy sanity case.
//   1 mild     : identity + a few random single-entry flips.
//   2 doncare  : for each of the 16 neighbor patterns, ~50% chance the RULE
//                is independent of the center bit (both center values give the
//                same output) -- exploitable preimage freedom (reversibility
//                hook); remaining patterns are forced center-SENSITIVE.
//   3 chaotic  : fully random 32-entry table (apart from rule[0]) -> a random
//                boolean function of 5 inputs is generically high-sensitivity,
//                so small seed errors blow up over T steps (the trap).
//   4 needle   : chaotic rule, but the TARGET is forward-simulated from a
//                SPARSE planted seed (low live-cell density) -- a very cheap
//                exact solution exists, hidden behind a chaotic map.
//   5 mixed    : doncare-style table with an extra layer of random overwrites
//                -- partially structured, partially chaotic.
//
// PLANTED TRAP (>=3 of 10 cases): kinds 3/4 (testIds 5,6,7,8,9) use a fully
// random, high-avalanche rule table under T>=3 steps. A naive cell-by-cell
// backward pass (using not-yet-resolved neighbor guesses) compounds error
// every step and lands far from the target; only a search that keeps whole
// neighborhoods consistent (and, for kind 4, that can find the hidden sparse
// seed at all) gets close.
// -----------------------------------------------------------------------------

static int gN, gT;
static char rule[32];

void buildRule(int kind){
    for (int v = 0; v < 32; v++) rule[v] = '0';
    if (kind == 0){
        for (int v = 0; v < 32; v++) rule[v] = ((v >> 4) & 1) ? '1' : '0';
    } else if (kind == 1){
        for (int v = 0; v < 32; v++) rule[v] = ((v >> 4) & 1) ? '1' : '0';
        int flips = 2 + rnd.next(0, 2);
        for (int k = 0; k < flips; k++){
            int idx = 1 + rnd.next(0, 30);
            rule[idx] = (rule[idx] == '0') ? '1' : '0';
        }
    } else if (kind == 2 || kind == 5){
        for (int nb = 0; nb < 16; nb++){
            bool dontcare = rnd.next(0, 99) < 50;
            if (dontcare){
                char r = rnd.next(0, 1) ? '1' : '0';
                rule[nb] = r; rule[16 + nb] = r;
            } else {
                int r0 = rnd.next(0, 1);
                int r1 = 1 - r0;
                rule[nb] = r0 ? '1' : '0';
                rule[16 + nb] = r1 ? '1' : '0';
            }
        }
        if (kind == 5){
            for (int v = 0; v < 32; v++)
                if (rnd.next(0, 99) < 35) rule[v] = rnd.next(0, 1) ? '1' : '0';
        }
    } else { // kind 3 or 4: chaotic
        for (int v = 0; v < 32; v++) rule[v] = rnd.next(0, 1) ? '1' : '0';
    }
    rule[0] = '0';
}

void step(const vector<string> &cur, vector<string> &nxt, int N){
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
}

int densityFor(int kind){
    if (kind == 0 || kind == 1) return 30;
    if (kind == 2 || kind == 5) return 35;
    if (kind == 3) return 40;
    return 6; // kind 4: needle, sparse planted seed
}

int main(int argc, char *argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static int Nl[11] = {0, 6, 10, 16, 24, 32, 45, 60, 80, 110, 150};
    static int Tl[11] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    static int Kl[11] = {0, 0, 1, 2, 5, 3, 3, 4, 3, 4, 5};
    static int Al[11] = {0, 5, 5, 6, 6, 8, 8, 10, 10, 12, 12};
    static int Bl[11] = {0, 1, 1, 3, 3, 2, 2, 3, 3, 4, 4};

    int N = Nl[testId], T = Tl[testId], kind = Kl[testId];
    int alpha = Al[testId], beta = Bl[testId];
    gN = N; gT = T;

    buildRule(kind);
    int dens = densityFor(kind);

    vector<string> cur(N, string(N, '0'));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (rnd.next(0, 99) < dens) cur[i][j] = '1';

    vector<string> nxt(N, string(N, '0'));
    for (int s = 0; s < T; s++){
        step(cur, nxt, N);
        swap(cur, nxt);
    }

    bool any = false;
    for (int i = 0; i < N && !any; i++)
        for (int j = 0; j < N; j++)
            if (cur[i][j] == '1'){ any = true; break; }
    if (!any) cur[0][0] = '1';

    printf("%d %d %d %d\n", N, T, alpha, beta);
    for (int v = 0; v < 32; v++) putchar(rule[v]);
    putchar('\n');
    for (int i = 0; i < N; i++) printf("%s\n", cur[i].c_str());
    return 0;
}
