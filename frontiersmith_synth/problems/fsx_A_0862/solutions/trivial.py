# TIER: trivial
# Dumbest-possible-but-always-feasible plan: move exactly one arm at a time,
# one distance unit per tick (speed 1). Never risks the cap (1^2=1<=P
# always) but is maximally slow and reproduces the checker's own baseline
# construction exactly.
import sys


def main():
    data = sys.stdin.read().split("\n")
    K, P, A = (int(x) for x in data[0].split())
    D = [int(x) for x in data[1].split()]

    out = []
    for i in range(K):
        row = [0] * K
        for _ in range(D[i]):
            row[i] = 1
            out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
