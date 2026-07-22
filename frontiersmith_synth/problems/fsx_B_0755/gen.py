#!/usr/bin/env python3
"""
gen.py <testId> -- fsx_B_0755 "Carving a Radio Filter from Integer Gears"

Prints ONE instance to stdout:
    line 1:            N B M
    next M lines:       k type T      (type in {P,S,D}; T meaningful only for P)

The instance is a "comb" frequency mask on a ring of N gear positions: canonical
bin k=0..N//2 is PASS (target gain A) if (k-r) mod P == 0, DONTCARE if it is 1
or P-1 bins from a tooth, and STOP (target 0) otherwise. The L1 budget B is a
fixed fraction of the L1 norm of the exact real-valued filter that hits every
PASS/STOP bin exactly with the DONTCARE bins pinned to zero -- i.e. exactly the
budget a "textbook" designer would need if they (wrongly) treated don't-care as
an extra stopband. All randomness is seeded purely by testId (deterministic).
"""
import sys
import math
import numpy as np

# N, P (comb period), r (phase), A (pass-band target gain), xf (extra L1
# headroom ABOVE the exact DC=0 reference norm, as a fraction of it) --
# increasing size / shrinking headroom = difficulty ladder.
#
# N is deliberately chosen so that P does NOT divide N: if it did, the PASS
# bins (equally spaced by P, mirrored around the ring) would line up into a
# single period-(N/P) Dirac comb across the *whole* circle, whose exact
# real-valued filter is itself a trivial period-(N/P) impulse train -- a
# reachable, already-near-integer "known optimal" construction. Breaking the
# divisibility keeps the exact real filter genuinely non-integer everywhere.
TESTS = [
    (17, 4, 0,  6.130, 0.35),
    (21, 5, 1,  6.580, 0.32),
    (24, 5, 2,  7.290, 0.28),
    (28, 6, 0,  7.810, 0.24),
    (33, 6, 3,  8.060, 0.20),
    (40, 7, 1,  8.470, 0.17),
    (47, 7, 4,  9.140, 0.14),
    (65, 8, 2,  9.630, 0.11),
    (80, 9, 5, 10.220, 0.08),
    (97, 9, 0, 10.770, 0.06),
]


def build_roles(N, P, r):
    M = N // 2 + 1
    roles = []
    for k in range(M):
        d = (k - r) % P
        if d == 0:
            roles.append((k, 'P'))
        elif d == 1 or d == P - 1:
            roles.append((k, 'D'))
        else:
            roles.append((k, 'S'))
    return roles


def reference_l1(N, roles, A):
    """L1 norm of the exact real filter matching every PASS/STOP bin, with
    DONTCARE bins pinned to 0 (the 'treat unscored as stopband' reference)."""
    C = np.zeros(N, dtype=complex)
    for k, typ in roles:
        val = A if typ == 'P' else 0.0
        C[k] = val
        if k != 0 and 2 * k != N:
            C[N - k] = val
    h = np.fft.ifft(C).real
    return float(np.sum(np.abs(h)))


def main():
    test_id = int(sys.argv[1])
    N, P, r, A, xf = TESTS[(test_id - 1) % len(TESTS)]
    roles = build_roles(N, P, r)
    l1ref = reference_l1(N, roles, A)
    # B sits ABOVE the exact real-valued match (l1ref) so that scaling the
    # amplitude down is never necessary -- the only source of deviation is
    # the integer-rounding step itself, not a budget-forced amplitude cut.
    B = max(6, int(math.ceil(l1ref * (1.0 + xf))))
    M = len(roles)

    out = [f"{N} {B} {M}"]
    for k, typ in roles:
        T = A if typ == 'P' else 0.0
        out.append(f"{k} {typ} {T:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
