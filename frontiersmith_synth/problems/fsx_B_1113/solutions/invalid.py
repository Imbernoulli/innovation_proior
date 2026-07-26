# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    # Land everyone on runway 1 at fix_time=0, landing_time=0, ignoring readiness,
    # transit, fix-separation and wake-separation entirely. Infeasible on any
    # instance with N >= 2 (and even N==1 typically violates readiness if r_1>0).
    lines = ["1 0 0" for _ in range(N)]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
