// TIER: greedy
// The natural first instinct: sum of independent games -> Sprague-Grundy.
// Compute each cairn's normal-play Grundy number and play the textbook
// normal-play winning move (drive the XOR of Grundy numbers to 0). This is
// exactly correct for NORMAL play, but normal-play sums do not transfer to
// misere play, so this move is frequently wrong once the position nears the
// misere endgame (every cairn's Grundy number in {0,1}).
#include <bits/stdc++.h>
using namespace std;

static int grundy(int r, int n) { return n % (r + 1); }

int main() {
    int T;
    if (scanf("%d", &T) != 1) return 0;
    for (int k = 0; k < T; k++) {
        int M; scanf("%d", &M);
        vector<int> R(M), N(M);
        for (int i = 0; i < M; i++) scanf("%d %d", &R[i], &N[i]);
        vector<int> gs(M);
        for (int i = 0; i < M; i++) gs[i] = grundy(R[i], N[i]);
        int X = 0;
        for (int g : gs) X ^= g;

        int outIdx = -1, outNew = -1;
        if (X != 0) {
            // standard normal-play technique: find a cairn where the XOR
            // target is achievable via one legal move.
            for (int i = 0; i < M && outIdx < 0; i++) {
                int target = gs[i] ^ X;
                if (target >= gs[i]) continue;
                for (int k2 = 1; k2 <= R[i] && k2 <= N[i]; k2++) {
                    int newN = N[i] - k2;
                    if (grundy(R[i], newN) == target) {
                        outIdx = i; outNew = newN; break;
                    }
                }
            }
        }
        if (outIdx < 0) {
            // fallback: X==0 (greedy thinks it's already lost / has no
            // XOR-zeroing target) or no exact-target move exists -- take 1
            // stone from the first non-empty cairn.
            for (int i = 0; i < M; i++) if (N[i] >= 1) { outIdx = i; outNew = N[i] - 1; break; }
        }
        printf("%d %d\n", outIdx, outNew);
    }
    return 0;
}
