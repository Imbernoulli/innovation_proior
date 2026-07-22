# TIER: trivial
"""Uniform K-slice, single pass, sell everything -- exactly the checker's own
internal baseline construction. No attempt to read the actual overlap structure."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it)); G = int(next(it)); step1 = int(next(it))
    for _ in range(K):
        next(it)  # values
    for _ in range(K * G):
        next(it)  # masses (unused)
    # H, energyCost, bands: unused by this construction

    raw = [round(i * G / K / step1) * step1 for i in range(1, K)]
    cuts = sorted({c for c in raw if 1 <= c <= G - 1})
    out = []
    out.append(f"{len(cuts)} " + " ".join(str(c) for c in cuts))
    out.append(" ".join(["S"] * (len(cuts) + 1)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
