# TIER: greedy
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    C = int(next(it)); P = int(next(it)); H = int(next(it)); Lmax = int(next(it))
    A_full = int(next(it)); t = int(next(it))

    exposures = []
    ages = []
    mixes = []
    triangles = []   # cumulative reported, index by development age
    for _ in range(C):
        exposure = float(next(it)); age = int(next(it))
        mix = [float(next(it)) for _ in range(P)]
        K = int(next(it))
        rvals = [float(next(it)) for _ in range(K)]
        exposures.append(exposure); ages.append(age); mixes.append(mix)
        triangles.append(rvals)

    m = [min(ages[c], A_full) for c in range(C)]

    # Classic chain-ladder: pool ALL cohorts (ignore product mix entirely) to
    # get ONE aggregate age-to-age development factor per age, then chain
    # them into a cumulative-to-ultimate factor and apply it to each
    # cohort's own latest reported value.
    factors = [1.0] * A_full
    for a in range(A_full):
        num = 0.0
        den = 0.0
        for c in range(C):
            if m[c] >= a + 1:
                num += triangles[c][a + 1]
                den += triangles[c][a]
        factors[a] = (num / den) if den > 1e-9 else 1.0

    cdf = [1.0] * (A_full + 1)
    for a in range(A_full - 1, -1, -1):
        cdf[a] = cdf[a + 1] * factors[a]

    out = []
    for c in range(C):
        last = triangles[c][-1]
        ultimate = last * cdf[m[c]]
        reserve = max(0.0, ultimate - last)
        out.append("%.6f" % reserve)
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
