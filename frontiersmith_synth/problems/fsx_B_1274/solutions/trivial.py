# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    C = int(next(it)); P = int(next(it)); H = int(next(it)); Lmax = int(next(it))
    A_full = int(next(it)); t = int(next(it))

    out = []
    for _ in range(C):
        exposure = float(next(it)); age = int(next(it))
        mix = [float(next(it)) for _ in range(P)]
        K = int(next(it))
        rvals = [float(next(it)) for _ in range(K)]
        # Naive: reported claims ARE the whole story -- reserve nothing more.
        out.append("0.0")
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
