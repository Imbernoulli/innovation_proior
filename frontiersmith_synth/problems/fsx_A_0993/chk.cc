#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// -----------------------------------------------------------------------------
// Checker / scorer for "Sampler Quilt: No Two Blocks Repeat".
//
// Input:  B b c Q tol W1  (single line).
// Output: B*B*b*b triples "d c1 c2", in order block-row, block-col, cell-row,
//         cell-col (see statement).
//
// Feasibility (hard): d in {0,1}; c1,c2 in [1,c], c1!=c2; global color-balance
//   |count_k*c - 2*B*B*b*b| <= tol*c for every color k; exactly the right
//   token count (seekEof).
//
// Objective (MAX): F = W1*min(M,Q) + S, where
//   M = number of distinct dihedral-equivalence classes among the B*B block
//       patterns (a block's pattern = its b x b matrix of diagonal
//       orientations; equivalence = the 8 symmetries of the square),
//   S = number of border cell-pairs (horizontal + vertical seams between
//       adjacent blocks) whose diagonal orientations agree.
//
// Baseline: Bref = W1*min(1,Q) + Smax, Smax = 2*B*(B-1)*b (every seam
//   continuing, one block-pattern equivalence class) -- exactly what the
//   trivial reference construction (uniform pattern + balanced coloring)
//   achieves, so it scores ratio == 0.1 by construction.
// Score (max): sc = min(1000, 100*F/max(1,Bref)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N_; // block size, for the bit-packed canonical-form helpers below

// A physical rotation/reflection of the square doesn't just move a cell to a
// new position -- it turns the cell itself, so a "\" triangle-split becomes a
// "/" split (and vice versa) under any 90/270 rotation or either axis-aligned
// mirror, and stays "\" under 180 rotation or either diagonal mirror. Both
// helpers below flip the moved bit accordingly; composing them (as canonForm
// does) then reproduces the correct parity for all 8 symmetries.
u64 rot90(u64 m, int N){
    u64 r = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            int oldbit = (int)((m >> (i * N + j)) & 1ULL);
            int newbit = 1 - oldbit;             // 90-degree turn flips the diagonal
            if (newbit) {
                int ni = j, nj = N - 1 - i;
                r |= (1ULL << (ni * N + nj));
            }
        }
    return r;
}

u64 mirror(u64 m, int N){
    u64 r = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            int oldbit = (int)((m >> (i * N + j)) & 1ULL);
            int newbit = 1 - oldbit;             // axis-aligned mirror flips the diagonal
            if (newbit) {
                int nj = N - 1 - j;
                r |= (1ULL << (i * N + nj));
            }
        }
    return r;
}

u64 canonForm(u64 m, int N){
    u64 best = m;
    for (int rep = 0; rep < 2; rep++){
        u64 cur = (rep == 0) ? m : mirror(m, N);
        for (int k = 0; k < 4; k++){
            if (cur < best) best = cur;
            cur = rot90(cur, N);
        }
    }
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int B   = inf.readInt(2, 12, "B");
    int b   = inf.readInt(3, 7, "b");
    int c   = inf.readInt(3, 8, "c");
    int Q   = inf.readInt(1, B * B, "Q");
    int tol = inf.readInt(0, 10000, "tol");
    ll  W1  = inf.readLong(0LL, 100000LL, "W1");

    ll Smax = 2LL * B * (B - 1) * b;
    ll Bref = W1 * (ll)min(1, Q) + Smax;
    if (Bref <= 0) Bref = 1;

    int G = B * b; // global cell-grid dimension
    vector<vector<int>> dmat(G, vector<int>(G, 0));
    ll total = 2LL * B * B * b * b;
    vector<ll> colorCount(c + 1, 0);

    for (int br = 0; br < B; br++){
        for (int bc = 0; bc < B; bc++){
            for (int i = 0; i < b; i++){
                for (int j = 0; j < b; j++){
                    int d  = ouf.readInt(0, 1, "d");
                    int c1 = ouf.readInt(1, c, "c1");
                    int c2 = ouf.readInt(1, c, "c2");
                    if (c1 == c2)
                        quitf(_wa, "block (%d,%d) cell (%d,%d): c1==c2==%d (must differ)",
                              br, bc, i, j, c1);
                    colorCount[c1]++;
                    colorCount[c2]++;
                    dmat[br * b + i][bc * b + j] = d;
                }
            }
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the quilt");

    for (int k = 1; k <= c; k++){
        ll dev = llabs(colorCount[k] * (ll)c - total);
        if (dev > (ll)tol * c)
            quitf(_wa, "color %d imbalance: count=%lld target-share*c-deviates by %lld > tol*c=%lld",
                  k, colorCount[k], dev, (ll)tol * c);
    }

    // ---- M: distinct dihedral-equivalence classes among block patterns ----
    N_ = b;
    set<u64> classes;
    for (int br = 0; br < B; br++){
        for (int bc = 0; bc < B; bc++){
            u64 mask = 0;
            for (int i = 0; i < b; i++)
                for (int j = 0; j < b; j++)
                    if (dmat[br * b + i][bc * b + j])
                        mask |= (1ULL << (i * b + j));
            classes.insert(canonForm(mask, b));
        }
    }
    ll M = (ll)classes.size();

    // ---- S: continuing seam positions across block borders ----
    ll S = 0;
    for (int br = 0; br < B; br++){
        for (int bc = 0; bc + 1 < B; bc++){
            int leftCol  = bc * b + b - 1;
            int rightCol = (bc + 1) * b;
            for (int i = 0; i < b; i++){
                int gr = br * b + i;
                if (dmat[gr][leftCol] == dmat[gr][rightCol]) S++;
            }
        }
    }
    for (int bc = 0; bc < B; bc++){
        for (int br = 0; br + 1 < B; br++){
            int topRow = br * b + b - 1;
            int botRow = (br + 1) * b;
            for (int j = 0; j < b; j++){
                int gc = bc * b + j;
                if (dmat[topRow][gc] == dmat[botRow][gc]) S++;
            }
        }
    }

    ll F = W1 * min(M, (ll)Q) + S;

    double sc = min(1000.0, 100.0 * (double)F / (double)Bref);
    quitp(sc / 1000.0, "OK F=%lld Bref=%lld M=%lld Q=%d S=%lld Smax=%lld Ratio: %.6f",
          F, Bref, M, Q, S, Smax, sc / 1000.0);
    return 0;
}
