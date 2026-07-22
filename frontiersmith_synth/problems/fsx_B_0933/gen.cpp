#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Combination-Lock Rewiring Against Nudge Attacks"  (generator)
// family: flat-differential-permutation
//
// Prints a single odd prime p. The ladder is a fixed list of 10 primes chosen so
// that, for MOST of them, p-1 is divisible by 3 (and often also 5 and 7): this
// means the exponent d=3 (which always gives the provably-best degree-2 difference
// polynomial when valid) is NOT a legal permutation exponent (gcd(3,p-1) != 1),
// which is the TRAP: a solver that naively reaches for the smallest textbook
// exponent WITHOUT actually scanning candidates and evaluating their peak leakage
// lands on a mediocre exponent, while a solver that scans several candidate
// exponents (using the closed-form, shift-invariant evaluation) finds a
// meaningfully better one. Sizes grow from a tiny sanity case (p=151) up to the
// largest case (p=7351), filling the stated constraint envelope.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const ll primes[11] = {
        0,
        151, 181, 271, 421, 571,
        811, 1051, 1531, 4201, 7351
    };

    ll p = primes[testId];
    printf("%lld\n", p);
    return 0;
}
