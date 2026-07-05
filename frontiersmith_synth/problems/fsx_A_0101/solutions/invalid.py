# TIER: invalid
# Emits an explicit resonance triple {0, e0, 2*e0} (0+1+2 == 0 mod 3 on line 0)
# -> a resonance cascade -> checker must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); k = int(next(it))
    z = [0] * n
    a = [0] * n; a[0] = 1
    b = [0] * n; b[0] = 2
    S = [z, a, b]
    out = [str(len(S))] + [" ".join(map(str, v)) for v in S]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
