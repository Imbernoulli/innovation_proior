#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE powder-diffraction instance to stdout.

A hidden orthorhombic crystal (lattice constants a,b,c in angstrom, all mutually
perpendicular) with a hidden Bravais centering (P/C/I/F) produces Bragg peaks at
2*theta = 2*asin(lambda * sqrt((h/a)^2+(k/b)^2+(l/c)^2) / 2) for every integer
triple (h,k,l) (h,k,l >= 0, not all zero) that (a) is physically observable
(the argument of asin must be <= 1) and (b) satisfies the centering's reflection
condition (systematic-absence rule). Peaks closer than a resolution tolerance
merge into one observed line.

Only the peaks up to theta2_given_max are printed (STDOUT). Peaks between
theta2_given_max and theta2_full_max exist physically but are NEVER printed --
they are the held-out region the checker uses to grade extrapolation. The
hidden a,b,c and centering are NEVER printed; they are reconstructed inside
verify.py from the testId using the exact same seeded formula (see
`true_structure` below, duplicated verbatim in verify.py).
"""
import sys, math, random

LAMBDA = 1.5406          # Cu K-alpha1, angstrom -- fixed, known to the solver
MERGE_TOL_DEG = 0.05      # instrument resolution: peaks this close merge into one line


def allowed(h, k, l, cent):
    if h == 0 and k == 0 and l == 0:
        return False
    if cent == "P":
        return True
    if cent == "I":
        return (h + k + l) % 2 == 0
    if cent == "F":
        return (h % 2 == k % 2) and (k % 2 == l % 2)
    return False


def true_structure(t):
    """Hidden crystal for this test id (lives in gen AND verify, never printed)."""
    rng = random.Random(90000 + 97 * t)
    # hidden lattice constants are DISCLOSED to be distinct integers in [4,13] angstrom --
    # this keeps the challenge about combinatorial indexing / absence inference rather
    # than an open-ended continuous global optimization over the cell metric.
    a, b, c = (float(v) for v in rng.sample(range(4, 14), 3))
    centering_seq = {1: "P", 2: "P", 3: "I", 4: "F", 5: "I",
                      6: "F", 7: "I", 8: "F", 9: "I", 10: "F"}
    cent = centering_seq.get(t, "P")
    theta2_given_max = 33.0 + 2.0 * (t - 1)
    theta2_full_max = theta2_given_max + 22.0
    return a, b, c, cent, theta2_given_max, theta2_full_max


def gen_spectrum(a, b, c, cent, theta2_cutoff, lam=LAMBDA):
    """All allowed (h,k,l) with 2theta <= theta2_cutoff, merged into observed lines.
    Returns a list of [rep_theta2, [hkl, hkl, ...]] sorted ascending by rep_theta2."""
    q_max = (2.0 * math.sin(math.radians(theta2_cutoff / 2.0)) / lam) ** 2
    raw = []
    hb = int(a * math.sqrt(q_max)) + 2
    for h in range(0, hb + 1):
        qh = (h / a) ** 2
        if qh > q_max + 1e-12:
            break
        kb = int(b * math.sqrt(max(0.0, q_max - qh))) + 2
        for k in range(0, kb + 1):
            qhk = qh + (k / b) ** 2
            if qhk > q_max + 1e-12:
                break
            lb = int(c * math.sqrt(max(0.0, q_max - qhk))) + 2
            for l in range(0, lb + 1):
                q = qhk + (l / c) ** 2
                if q > q_max + 1e-12:
                    break
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed(h, k, l, cent):
                    continue
                sin_t = lam * math.sqrt(q) / 2.0
                if sin_t > 1.0:
                    continue
                theta2 = 2.0 * math.degrees(math.asin(sin_t))
                raw.append((theta2, (h, k, l)))
    raw.sort(key=lambda x: x[0])
    groups = []
    for theta2, hkl in raw:
        if groups and theta2 - groups[-1][0] <= MERGE_TOL_DEG:
            groups[-1][1].append(hkl)
        else:
            groups.append([theta2, [hkl]])
    return groups


def main():
    t = int(sys.argv[1])
    a, b, c, cent, theta2_given_max, theta2_full_max = true_structure(t)
    groups = gen_spectrum(a, b, c, cent, theta2_full_max)
    given = [g for g in groups if g[0] <= theta2_given_max]

    out = [str(t), "%.6f" % LAMBDA, "%.4f %.4f" % (theta2_given_max, theta2_full_max),
           str(len(given))]
    for rep_theta2, _hkls in given:
        out.append("%.6f" % rep_theta2)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
