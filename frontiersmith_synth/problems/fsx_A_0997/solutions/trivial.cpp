// TIER: trivial
// Reproduces the checker's own weak-but-stable reference: chords + verticals +
// ONE diagonal per bay (type A only, per the documented candidate ordering),
// smallest area class everywhere. Always affordable (budget >= this cost by
// construction) and always kinematically stable (fully triangulated), so it is
// the honest "do the least you can get away with" baseline.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, K;
    if (!(cin >> W >> K)) return 0;
    // We only need W; consume nothing else.
    int S = 4 * W + 1; // bottom(W) + top(W) + vert(W+1) + diagA(W)
    printf("%d\n", S);
    for (int i = 0; i < W; i++) printf("%d 0\n", i);              // bottom chord
    for (int i = 0; i < W; i++) printf("%d 0\n", W + i);          // top chord
    for (int i = 0; i <= W; i++) printf("%d 0\n", 2 * W + i);     // vertical
    for (int i = 0; i < W; i++) printf("%d 0\n", 3 * W + 1 + i);  // diagonal A
    return 0;
}
