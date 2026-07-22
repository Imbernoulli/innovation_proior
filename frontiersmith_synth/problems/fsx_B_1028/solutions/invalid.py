# TIER: invalid
"""Deliberately infeasible: guard 2 never leaves the hub (period-1 walk
that just sits at room 0 forever), so unless guard 1 alone happens to
cover every room -- which it does not here, guard 1 only loops the first
petal -- most rooms are never visited by either guard. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0]); P = int(data[1])
    Ls = [int(data[2 + i]) for i in range(k)]

    # guard 1: loop only the FIRST petal (ignores the rest of the graph)
    L1 = Ls[0]
    priv1 = list(range(1, L1))
    w1 = [0] + priv1

    # guard 2: stays at the hub forever (period 1)
    w2 = [0]

    out = [str(len(w1)), " ".join(str(x) for x in w1),
           str(len(w2)), " ".join(str(x) for x in w2)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
