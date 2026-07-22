# TIER: invalid
# Deliberately infeasible: claims to deliver zero energy to every bus (never meets the
# required charge), which must be rejected by the checker's sufficiency check.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); D = int(next(it)); C = int(next(it)); pmax = int(next(it))
    for _ in range(n):
        for _ in range(4):
            next(it)

    out_lines = []
    for _ in range(n):
        out_lines.append(" ".join(["0"] * D))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
