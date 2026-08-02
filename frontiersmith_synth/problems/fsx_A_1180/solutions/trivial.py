# TIER: trivial
import sys, math

LAMBDA = 1.5406


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    _t = int(next(it))
    lam = float(next(it))
    _gmax = float(next(it))
    _fmax = float(next(it))
    M = int(next(it))
    qs = [float(next(it)) for _ in range(M)]

    # laziest possible guess: treat the FIRST observed line as the (1,0,0) reflection of
    # a CUBIC cell (a=b=c) -- ignores that the crystal is orthorhombic entirely, and
    # never revisits this guess against any other peak.
    s = math.sin(math.radians(qs[0] / 2.0)) if M >= 1 else 0.5
    if s < 1e-9:
        s = 1e-9
    a = lam / (2.0 * s)
    a = min(max(a, 0.05), 39.9)
    b = c = a

    # fixed, non-adaptive canonical index sequence -- assigned in that order regardless
    # of how well it actually matches each subsequent peak (no search, no refitting).
    seq = []
    n = 0
    while len(seq) < M:
        n += 1
        for h in range(0, n + 1):
            for k in range(0, n + 1):
                for l in range(0, n + 1):
                    if max(h, k, l) != n:
                        continue
                    seq.append((h, k, l))
    seq = seq[:M]

    out = ["%.6f %.6f %.6f" % (a, b, c), "P"]
    for (h, k, l) in seq:
        out.append("%d %d %d" % (h, k, l))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
