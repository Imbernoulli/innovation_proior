# TIER: invalid
# Deliberately infeasible: claims one more split than the budget allows (K = S + 1), which
# the checker must reject outright.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    C = int(next(it)); S = int(next(it)); D = int(next(it))
    out = [str(S + 1)]
    for i in range(S + 1):
        out.append("0 0 0")   # nonsensical: re-splitting the same already-split root
    out.append("1")
    out.append("0 0 0 0.5")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
