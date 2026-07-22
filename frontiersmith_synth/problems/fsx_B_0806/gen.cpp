// Generator for fsx_B_0806 "Mirror Hall Laserglyph".
// Prints: "n M group\nec\n"  (group in {C2,D4}; deterministic given testId via testlib rnd).
#include "testlib.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // Size ladder 8 -> 40 (even), alternating target group so both C2 and D4 are
    // exercised at every scale; testId 10 fills the stated envelope (n=40).
    static const int ns[10]      = {8, 10, 14, 18, 22, 26, 30, 34, 38, 40};
    static const char* grps[10]  = {"C2","D4","C2","D4","C2","D4","C2","D4","C2","D4"};

    int idx = testId - 1;
    if (idx < 0) idx = 0;
    if (idx > 9) idx = 9;
    int n = ns[idx];
    std::string group = grps[idx];

    // Mirror budget: comfortably above what a single closed loop needs (~5), but a
    // hard cap the trap solution (which spends it on symmetric-but-unreached mirrors)
    // will visibly squander. testId 9 uses a slightly tighter budget (needle case).
    int M = n / 2 + 8;
    if (testId == 9) M = n / 2 + 5;

    // ec: the emitter's column, kept away from the extreme edges (n/5 .. 4n/5) so a
    // reasonably sized centred shape can always reach it -- this is what makes every
    // test case a genuine trap for symmetric-hardware placement (the straight path
    // is always far from any grid-edge degeneracy), not just a chosen few.
    int lo = n / 5;
    int hi = (4 * n) / 5;
    if (lo < 1) lo = 1;
    if (hi > n - 2) hi = n - 2;
    if (hi < lo) hi = lo;
    int ec = rnd.next(lo, hi);

    printf("%d %d %s\n", n, M, group.c_str());
    printf("%d\n", ec);
    return 0;
}
