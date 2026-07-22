#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// No-Tell Round Robin -- generator.
//
// Emits a single integer n (odd, 3 <= n <= 1001) per test id. The 10 values
// are chosen to walk the *structural* ladder that the strong solution must
// navigate, not just a size ladder:
//   - testId 1-2:  tiny sanity + a prime q = 3 mod 4 (Paley applies directly).
//   - testId 3:    a clean product of two primes = 3 mod 4 (needs the
//                   lexicographic product, no patch needed).
//   - testId 4:    a prime = 1 mod 4 (Paley does NOT apply to n itself --
//                   this is the TRAP: a solver that only special-cases
//                   "n itself prime, 3 mod 4" gets nothing here, and must
//                   fall back to factoring/patching around n).
//   - testId 5-6:  clean products again, growing.
//   - testId 7:    product of two primes that are BOTH = 1 mod 4 (n=13*17):
//                   no direct factor at all is usable -- a hard TRAP that
//                   stresses the recursive patch fallback.
//   - testId 8:    a mixed product (one good prime, one bad prime).
//   - testId 9:    a large prime = 1 mod 4 (n=701) -- large-scale TRAP,
//                   greedy's circulant construction and the naive ranked
//                   baseline both degrade the same way here, so this test
//                   isolates whether the solver's patch strategy still
//                   salvages a real advantage at scale.
//   - testId 10:   n=1001=7*11*13 (two good primes, one bad) at the top of
//                   the stated envelope -- fills the constraint envelope.
// All values are odd and in [3, 1001] as required. Determinism: purely a
// fixed lookup by testId (no dependence on wall time, environment, or
// iteration order); `rnd` is still initialized via registerGen for contract
// compliance even though this generator does not need extra randomness.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int id = atoi(argv[1]);
    if (id < 1) id = 1;
    if (id > 10) id = 10;

    static const int table[11] = {
        0,   // unused
        11,  // 1: tiny, prime = 3 mod 4 (direct Paley)
        21,  // 2: 3*7, clean product of two good primes
        41,  // 3: prime = 1 mod 4 -- trap, no direct Paley on n
        99,  // 4: 3*3*11, clean product (repeated factor)
        151, // 5: prime = 3 mod 4, direct Paley at moderate scale
        161, // 6: 7*23, clean product
        221, // 7: 13*17, BOTH factors = 1 mod 4 -- hard trap
        323, // 8: 17*19, mixed (one bad, one good factor)
        701, // 9: prime = 1 mod 4, large-scale trap
        1001 // 10: 7*11*13, two good + one bad, fills the envelope
    };

    int n = table[id];
    // rnd is unused for these fixed structural picks, but touched once so the
    // RNG state is initialized deterministically per testId as the contract
    // requires for generators that do use it.
    rnd.next(1);

    printf("%d\n", n);
    return 0;
}
