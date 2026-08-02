# TIER: invalid
# Every op crammed into (cycle=1, slot=0): violates the slot-type match for any
# op whose required type isn't slot_types[0], AND violates the no-two-ops-per-
# bundle-slot rule for every op that DOES match -- guaranteed infeasible.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    out = []
    for i in range(1, N + 1):
        out.append("1 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
