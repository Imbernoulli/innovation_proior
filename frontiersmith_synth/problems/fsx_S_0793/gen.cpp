#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Grow the Glyph"  (generator)  family: reverse-engineered-ca-seeder
//
// A ring of N cells evolves under a RADIUS-2 LINEAR (XOR) cellular automaton: a
// 5-bit mask M selects which offsets d in {-2,-1,0,1,2} feed the update
//     next[i] = XOR_{d : bit(M,d) set} state[(i+d) mod N]
// applied synchronously for T steps. Because XOR is linear over GF(2), the whole
// T-step map is a fixed NxN linear operator A (mod 2) determined only by (N,M,T).
//
// PLANTED STRUCTURE: every test's target is produced by picking a SPARSE precursor
// s0 (weight w <= B) and running it forward T steps -> target = A * s0. So an exact,
// budget-feasible precursor is guaranteed to exist; a solver must recover it (or an
// equally good one) by reasoning about A, not by guessing.
//
// TRAP (verified at generation time, retried until it holds -- not hoped-for): the
// natural first-instinct "greedy" heuristic is simulated HERE with the exact same
// algorithm solutions/greedy.cpp uses (a naive per-step Jacobi backward substitution
// that peels one layer at a time using a single distinguished tap and silently
// substitutes the LAYER ABOVE for the other taps instead of solving the ring-wide
// simultaneous system). We reject instances until BOTH hold:
//   (a) the do-nothing baseline B_base = zero-count(target) is bounded away from N
//       (so an EXACT reconstruction, which always matches all N cells, stays well
//       under the 10x-baseline score cap -- leaves headroom above 'strong'), and
//   (b) the naive-Jacobi greedy clearly beats do-nothing (ladder sanity) yet lands
//       well short of a genuine precursor (the trap: it never reaches N matches).
// Output: N T B M  then the length-N target string.
// -----------------------------------------------------------------------------

static int maskOf(int off){ return off + 2; } // offsets -2..2 -> bits 0..4

void evolve(vector<char>& s, int N, int M, int T){
    vector<char> cur = s, nxt(N);
    for (int step = 0; step < T; step++){
        for (int i = 0; i < N; i++){
            int v = 0;
            for (int d = -2; d <= 2; d++){
                if (M & (1 << maskOf(d))){
                    int j = ((i + d) % N + N) % N;
                    v ^= cur[j];
                }
            }
            nxt[i] = (char)v;
        }
        swap(cur, nxt);
    }
    s = cur;
}

// Exactly mirrors solutions/greedy.cpp: naive one-tap-lag Jacobi backward
// substitution, T layers, then budget-capped index-order selection of 1-cells.
int simulateGreedyMatches(const vector<char>& target, int N, int M, int T, int Bc){
    vector<int> offs;
    for (int d = 2; d >= -2; d--) if (M & (1 << maskOf(d))) offs.push_back(d);
    int dstar = offs[0];
    vector<char> cur(N);
    for (int i = 0; i < N; i++) cur[i] = target[i];
    for (int step = 0; step < T; step++){
        vector<char> guess(N, 0);
        for (int i = 0; i < N; i++){
            int v = cur[i];
            for (int d : offs) if (d != dstar) v ^= cur[((i + d) % N + N) % N];
            int pos = ((i + dstar) % N + N) % N;
            guess[pos] = (char)v;
        }
        cur = guess;
    }
    vector<char> seed(N, 0);
    int cnt = 0;
    for (int i = 0; i < N && cnt < Bc; i++) if (cur[i]) { seed[i] = 1; cnt++; }
    vector<char> fin = seed;
    evolve(fin, N, M, T);
    int match = 0;
    for (int i = 0; i < N; i++) if (fin[i] == target[i]) match++;
    return match;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int N = (int)llround(12 + f * 188.0);      // 12 .. 200
    if (N < 8) N = 8;
    int T = (int)llround(3 + f * 27.0);        // 3 .. 30
    if (T < 2) T = 2;

    int wMin = max(1, N / 14);
    int wMax = max(wMin + 1, N / 6);

    vector<char> target, s0;
    int M = 0, w = 0, B = 0;
    const int MAXTRY = 800;
    bool ok = false;
    for (int tryNo = 0; tryNo < MAXTRY; tryNo++){
        // pick a mixing mask: popcount in {2,3,4}, i.e. never a bare shift/identity
        int pc = 2 + rnd.next(0, 2);
        vector<int> offs = {-2,-1,0,1,2};
        for (int i = 4; i > 0; i--) swap(offs[i], offs[rnd.next(0,i)]);
        int Mc = 0;
        for (int i = 0; i < pc; i++) Mc |= (1 << maskOf(offs[i]));

        w = wMin + rnd.next(0, wMax - wMin);
        int slack = max(2, N / 40 + 1);
        int Bc = w + slack;
        if (Bc > N) Bc = N;
        if (w > Bc) w = Bc;

        vector<int> idx(N); for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = N - 1; i > 0; i--) swap(idx[i], idx[rnd.next(0,i)]);
        vector<char> s0c(N, 0);
        for (int i = 0; i < w; i++) s0c[idx[i]] = 1;

        vector<char> tgt = s0c;
        evolve(tgt, N, Mc, T);

        int zeros = 0; for (int i = 0; i < N; i++) if (!tgt[i]) zeros++;
        int Bbase = zeros;
        // (a) keep Bbase in a band that leaves real headroom above a perfect (F=N)
        //     reconstruction: ratio_strong = min(1,N/(10*Bbase)) must stay <= ~0.7.
        if (Bbase < (int)ceil(N * 0.15) || Bbase > (int)floor(N * 0.55)) continue;

        int gm = simulateGreedyMatches(tgt, N, Mc, T, Bc);
        // (b) ladder sanity: naive greedy clearly beats do-nothing ...
        if (gm < (int)ceil(Bbase * 1.30)) continue;
        // ... but the trap: it never gets close to the exact N-match reconstruction.
        if (gm > (int)floor(N * 0.90)) continue;

        M = Mc; B = Bc; target = tgt; s0 = s0c; ok = true;
        break;
    }
    if (!ok){
        // extremely unlikely fallback: keep the last generated instance anyway
        M = M ? M : 0b00110; B = B ? B : max(2, N/6); if (target.empty()) target = s0;
    }

    printf("%d %d %d %d\n", N, T, B, M);
    for (int i = 0; i < N; i++) putchar(target[i] ? '1' : '0');
    putchar('\n');
    return 0;
}
