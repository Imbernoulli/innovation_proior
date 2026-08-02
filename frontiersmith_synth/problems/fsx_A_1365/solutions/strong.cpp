// TIER: strong
// The insight: misere sums of these (tame) cairns are governed by a COARSER
// equivalence than raw Grundy numbers -- Grundy values >=2 form one "wild"
// class (still combined by XOR), while Grundy values 0 and 1 need a separate
// odd/even parity rule once every cairn has fallen into {0,1}. Instead of
// re-deriving which single cairn-move to prefer, use this coarse classifier
// directly as a 1-ply oracle: try every legal move, keep the one whose
// resulting Grundy multiset the classifier calls a misere loss for the
// opponent. This is the working-in-the-quotient idea, not "greedy plus more
// search" -- the trivial/greedy tiers never consult this classifier at all.
#include <bits/stdc++.h>
using namespace std;

static int grundy(int r, int n) { return n % (r + 1); }

static bool theoremIsP(const vector<int>& gs) {
    bool anyWild = false;
    for (int g : gs) if (g >= 2) { anyWild = true; break; }
    if (anyWild) {
        int x = 0;
        for (int g : gs) x ^= g;
        return x == 0;
    }
    int c1 = 0;
    for (int g : gs) if (g == 1) c1++;
    return (c1 % 2) == 1;
}

int main() {
    int T;
    if (scanf("%d", &T) != 1) return 0;
    for (int k = 0; k < T; k++) {
        int M; scanf("%d", &M);
        vector<int> R(M), N(M);
        for (int i = 0; i < M; i++) scanf("%d %d", &R[i], &N[i]);
        vector<int> gs(M);
        for (int i = 0; i < M; i++) gs[i] = grundy(R[i], N[i]);

        int outIdx = -1, outNew = -1;
        for (int i = 0; i < M && outIdx < 0; i++) {
            for (int k2 = 1; k2 <= R[i] && k2 <= N[i]; k2++) {
                int newN = N[i] - k2;
                vector<int> ngs = gs;
                ngs[i] = grundy(R[i], newN);
                if (theoremIsP(ngs)) { outIdx = i; outNew = newN; break; }
            }
        }
        if (outIdx < 0) {
            // no misere-winning move found (shouldn't happen -- every input
            // duel is a first-player win by construction); stay feasible.
            for (int i = 0; i < M; i++) if (N[i] >= 1) { outIdx = i; outNew = N[i] - 1; break; }
        }
        printf("%d %d\n", outIdx, outNew);
    }
    return 0;
}
