# TIER: strong
"""Hammersley / Halton construction plus a deterministic best-of-shifts search.

Builds several low-discrepancy candidate point sets (Hammersley with (i+0.5)/M
first axis, Halton, and a small deterministic set of Cranley-Patterson digital
shifts of the Hammersley set), scores each with the SAME exact L2 star
discrepancy the checker uses, and emits the best. Different per-test winners
give behavior distinct from the greedy lattice."""
import sys
import math

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def radinv(i, b):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def hammersley(M, d):
    return [tuple([(i + 0.5) / M] + [radinv(i + 1, PRIMES[k]) for k in range(d - 1)])
            for i in range(M)]


def halton(M, d):
    return [tuple(radinv(i + 1, PRIMES[k]) for k in range(d)) for i in range(M)]


def shift(pts, sv):
    return [tuple((x[k] + sv[k]) % 1.0 for k in range(len(sv))) for x in pts]


def l2star_sq(pts, d, M):
    t1 = 3.0 ** (-d)
    s2 = 0.0
    for x in pts:
        p = 1.0
        for k in range(d):
            p *= (1.0 - x[k] * x[k])
        s2 += p
    t2 = (2.0 ** (1 - d) / M) * s2
    s3 = 0.0
    for i in range(M):
        xi = pts[i]
        for j in range(M):
            xj = pts[j]
            p = 1.0
            for k in range(d):
                a = xi[k]
                b = xj[k]
                p *= (1.0 - (a if a > b else b))
            s3 += p
    return t1 - t2 + s3 / (M * M)


def main():
    data = sys.stdin.read().split()
    d, M = int(data[0]), int(data[1])

    ham = hammersley(M, d)
    candidates = [ham, halton(M, d)]
    # a few deterministic Cranley-Patterson shifts of the Hammersley set
    for g in (0.5, 0.2, 0.7):
        sv = [((k + 1) * g) % 1.0 for k in range(d)]
        candidates.append(shift(ham, sv))

    best = None
    best_val = None
    for c in candidates:
        v = l2star_sq(c, d, M)
        if best_val is None or v < best_val:
            best_val = v
            best = c

    out = []
    for x in best:
        out.append(" ".join("%.10f" % xi for xi in x))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
