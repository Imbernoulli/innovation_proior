# TIER: invalid
"""
Deliberately infeasible: emits a duplicate-laden non-permutation and a
swap-event budget that exceeds K, plus a non-finite token thrown in.
Must score 0 under the checker's strict feasibility gate.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    n = int(data[ptr]); ptr += 1
    m = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    P = int(data[ptr]); ptr += 1
    ptr += 2 * P
    ptr += m

    out = []
    # not a permutation: item 1 repeated, item n missing
    bogus = [1] * n
    out.append(" ".join(str(x) for x in bogus))
    # declare far more swap events than the budget allows
    out.append(str(K + 5))
    for k in range(K + 5):
        out.append("1 1")
    out.append("nan inf")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
