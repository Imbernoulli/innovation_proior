# TIER: trivial
# Conservative catalogue = the checker baseline: {0,1}^(n-1) x {0}, size 2^(n-1).
import sys, itertools


def main():
    n = int(sys.stdin.read().split()[0])
    rows = []
    for bits in itertools.product((0, 1), repeat=n - 1):
        rows.append(list(bits) + [0])
    out = [str(len(rows))]
    for r in rows:
        out.append(" ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
