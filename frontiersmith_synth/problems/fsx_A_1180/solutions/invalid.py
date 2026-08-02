# TIER: invalid
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    _t = int(next(it))
    _lam = float(next(it))
    _gmax = float(next(it))
    _fmax = float(next(it))
    M = int(next(it))
    # garbage: non-finite lattice constants, bogus centering, and h=k=l=0 indices
    out = ["nan nan nan", "X"]
    for _ in range(M):
        out.append("0 0 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
