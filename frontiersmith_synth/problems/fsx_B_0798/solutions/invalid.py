# TIER: invalid
"""
Deliberately infeasible: skips node 0 as the first amplifier and emits a
power level that is not in the allowed discrete set. Must score Ratio: 0.0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    out = []
    out.append("2")
    # first amplifier index is NOT 0 -> infeasible
    second = min(2, max(1, N - 1))
    out.append("1 %d" % second)
    # power value that is (almost certainly) not in the allowed set, and
    # non-finite noise besides -- either alone must trigger Ratio: 0.0
    out.append("999999.5 999999.5")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
