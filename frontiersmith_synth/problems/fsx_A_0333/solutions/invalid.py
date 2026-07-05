# TIER: invalid
# Emit heavily overlapping disks that also ignore the platform-cover constraint:
# every station gets a huge disk centred at the city middle -> non-overlap fails
# and platforms are missed. Must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = []
    for _ in range(N):
        out.append("0.5 0.5 0.49")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
