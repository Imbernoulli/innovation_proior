# TIER: trivial
# One singleton chain per village: (t, 1, 1) lights exactly {t}. This is the
# checker's own reference baseline B, reproduced exactly.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); alpha = int(next(it))
    m = int(next(it))
    T = [int(next(it)) for _ in range(m)]

    out = [str(m)]
    for t in T:
        out.append("%d 1 1" % t)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
