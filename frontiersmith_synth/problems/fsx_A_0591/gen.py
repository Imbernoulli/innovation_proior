#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE instance on stdout.
# Instance: line 1 = degree d ; line 2 = d+1 integer coefficients a_0 .. a_d (a_d != 0).
# Difficulty ladder: small dense polynomials -> large ones where Horner is far from the
# baby-step/giant-step reference (the trap cases). Everything seeded by testId only.
import sys, random

# degree per testId (1..10): ascending; the high-degree cases (>= testId 8) are the traps
# where the obvious Horner recipe lands far below the preconditioned strong solution.
DEG = {1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 16, 7: 20, 8: 24, 9: 27, 10: 30}


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d = DEG.get(tid, 8)
    rng = random.Random(9173 * tid + 41)
    # dense generic integer coefficients: no special sparse/composition structure,
    # so no evaluation scheme materially better than baby-step/giant-step is planted.
    coeffs = []
    for i in range(d + 1):
        c = rng.randint(-9, 9)
        if i < d and c == 0:
            # keep it dense (few zeros) so Horner cannot get lucky
            c = rng.choice([-1, 1]) * rng.randint(1, 9)
        coeffs.append(c)
    # leading coefficient nonzero
    lead = rng.randint(1, 9)
    coeffs[d] = lead * rng.choice([-1, 1])
    print(d)
    print(" ".join(str(c) for c in coeffs))


if __name__ == "__main__":
    main()
